# benchmark_framework.py
import json
import torch
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any
import csv
from pathlib import Path

@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    sample_index: int
    export_time: float
    inference_time: float
    inject_time: Optional[float] = None
    is_correct: Optional[bool] = None
    max_difference: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

# ModelWrapper for static
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

# Add new wrapper for dynamic
class DynamicExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, inputs):
        input_ids, attention_mask = inputs
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

class BenchmarkBase(ABC):
    """Base class for all benchmarking experiments"""
    def __init__(self, model_name: str, model_path: str, device: str = "cuda"):
        self.model_name = model_name
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.uses_dynamic_wrapper = False

    @abstractmethod
    def run_experiment(self, sample_input: Tuple) -> BenchmarkResult:
        """Run a single experiment iteration"""
        pass

    def time_operation(self, operation, *args, **kwargs):
        """Time any operation with proper synchronization"""
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()
        result = operation(*args, **kwargs)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed_time = time.time() - start_time
        return result, elapsed_time

    def export_model(self, sample_input: Tuple):
        """Export model with proper input structure"""
        wrapped_model = ModelWrapper(self.model)
        
        # ✅ FIX: Extract input_ids and attention_mask from sample_input
        # Handle different input structures
        if len(sample_input) >= 2:
            input_ids, attention_mask = sample_input[0], sample_input[1]
        else:
            raise ValueError(f"Expected at least 2 elements in sample_input, got {len(sample_input)}")
        
        # Export with separate arguments (not as tuple)
        return self.time_operation(
            torch.export.export,
            wrapped_model,
            args=(input_ids, attention_mask)  # Pass as separate args
        )

    def run_inference(self, exported_model, sample_input: Tuple) -> Tuple[Any, float]:
        """Run inference with exported model"""
        if len(sample_input) >= 2:
            input_ids, attention_mask = sample_input[0], sample_input[1]
        else:
            raise ValueError(f"Expected at least 2 elements in sample_input, got {len(sample_input)}")
        
        if hasattr(self, 'uses_dynamic_wrapper') and self.uses_dynamic_wrapper:
            # For dynamic exports, call the exported model directly (no .module())
            return self.time_operation(
                exported_model.module(),
                input_ids, attention_mask
            )
        else:
            # Static export - use module() and pass as separate args
            return self.time_operation(
                exported_model.module(),
                input_ids, attention_mask
            )
   
    def save_results(self, results: List[BenchmarkResult], output_dir: str = "results"):
        """Save results to CSV"""
        Path(output_dir).mkdir(exist_ok=True)
        filename = f"{output_dir}/{self.model_name}_{self.__class__.__name__}_results.csv"
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            # Dynamic header based on what's in results
            if results:
                headers = ["Sample Index", "Export Time (s)", "Inference Time (s)"]
                if results[0].inject_time is not None:
                    headers.append("Inject Time (s)")
                if results[0].is_correct is not None:
                    headers.extend(["Is Correct", "Max Difference"])
                writer.writerow(headers)
                
                for r in results:
                    row = [r.sample_index, r.export_time, r.inference_time]
                    if r.inject_time is not None:
                        row.append(r.inject_time)
                    if r.is_correct is not None:
                        row.extend([r.is_correct, r.max_difference])
                    writer.writerow(row)


