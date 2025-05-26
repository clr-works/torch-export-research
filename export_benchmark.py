# export_benchmark.py
from benchmark_framework import BenchmarkBase, BenchmarkResult
import torch
from typing import Tuple

class ExportBenchmark(BenchmarkBase):
    """Benchmark for basic export, inference, and correctness checking"""
    
    def run_experiment(self, sample_input: Tuple) -> BenchmarkResult:
        try:
            # DEBUG: Print input structure
            print(f"Sample input length: {len(sample_input)}")
            print(f"Sample input types: {[type(x) for x in sample_input]}")
            if len(sample_input) >= 2:
                print(f"Input shapes: {[x.shape if hasattr(x, 'shape') else 'no shape' for x in sample_input[:2]]}")
            
            # Export model
            exported_model, export_time = self.export_model(sample_input)
            
            # Run inference with exported model
            exported_output, inference_time = self.run_inference(exported_model, sample_input)
            
            # Handle exported output
            if isinstance(exported_output, (tuple, list)):
                exported_logits = exported_output[0]
            elif hasattr(exported_output, 'logits'):
                exported_logits = exported_output.logits
            else:
                exported_logits = exported_output
            
            # Run original model inference consistently
            self.model.eval()
            with torch.no_grad():
                # Extract input_ids and attention_mask for original model
                if len(sample_input) >= 2:
                    input_ids, attention_mask = sample_input[0], sample_input[1]
                    # Call original model the same way as wrapper
                    original_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    raise ValueError(f"Expected at least 2 elements in sample_input, got {len(sample_input)}")
            
            # Handle original output
            if isinstance(original_output, (tuple, list)):
                original_logits = original_output[0]
            elif hasattr(original_output, 'logits'):
                original_logits = original_output.logits
            else:
                original_logits = original_output
            
            # Compare outputs
            is_close = torch.allclose(exported_logits, original_logits, atol=1e-5, rtol=1e-3)
            diff = (exported_logits - original_logits).abs().max().item()
            
            return BenchmarkResult(
                sample_index=0,  # Updated by runner
                export_time=export_time,
                inference_time=inference_time,
                is_correct=is_close,
                max_difference=diff
            )
            
        except Exception as e:
            print(f"❌ Error in run_experiment: {e}")
            print(f"Sample input structure: {sample_input}")
            raise

