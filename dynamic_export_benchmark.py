# dynamic_export_benchmark.py
from benchmark_framework import BenchmarkBase, BenchmarkResult
import torch
from torch.export import Dim, export
from typing import Tuple, Optional, List

# Define the constrained dimension
_seq_len = Dim('_seq_len', max=64)
seq_len = 8 * _seq_len

class DynamicExportBenchmark(BenchmarkBase):
    """Benchmark for dynamic shape export and inference"""
    
    def __init__(self, model_name: str, model_path: str, 
                 min_seq_len: int = 1, max_seq_len: int = 512, 
                 test_seq_lengths: Optional[List[int]] = None,
                 device: str = "cuda"):
        super().__init__(model_name, model_path, device)
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.test_seq_lengths = test_seq_lengths or [8, 32, 64, 128, 256]
        self.exported_model = None
    
    def export_model_dynamic(self, sample_input: Tuple):
        """Export model with dynamic sequence length support"""
        class ExportWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, input_ids, attention_mask):
                return self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        
        # Define dynamic dimension
        seq_len_dim = Dim("seq_len", min=self.min_seq_len, max=self.max_seq_len)
        
        # Extract sample inputs
        input_ids, attention_mask = sample_input[0], sample_input[1]
        
        # Define dynamic shapes for both inputs
        dynamic_shapes = {
            "input_ids": {1: seq_len_dim},
            "attention_mask": {1: seq_len_dim}
        }
        
        # Export with dynamic shapes
        wrapped_model = ExportWrapper(self.model)
        exported_program, export_time = self.time_operation(
            export,
            wrapped_model,
            args=(input_ids, attention_mask),
            dynamic_shapes=dynamic_shapes
        )
        
        return exported_program, export_time
    
    def test_multiple_lengths(self, exported_model):
        """Test exported model with different sequence lengths"""
        results = []
        
        for seq_len in self.test_seq_lengths:
            if seq_len < self.min_seq_len or seq_len > self.max_seq_len:
                print(f"Skipping seq_len {seq_len} (outside range [{self.min_seq_len}, {self.max_seq_len}])")
                continue
            
            # Create input of specific length
            input_text = " ".join(["test"] * seq_len)
            inputs = self.tokenizer(
                input_text, 
                return_tensors="pt", 
                padding="max_length", 
                max_length=seq_len, 
                truncation=True
            )
            
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            
            # Run inference
            try:
                output, inference_time = self.time_operation(
                    exported_model.module(),
                    input_ids, attention_mask
                )
                
                results.append({
                    "seq_len": seq_len,
                    "inference_time": inference_time,
                    "output_shape": output.shape,
                    "success": True
                })
                print(f"✓ Seq length {seq_len}: {inference_time:.4f}s, shape: {output.shape}")
                
            except Exception as e:
                results.append({
                    "seq_len": seq_len,
                    "error": str(e),
                    "success": False
                })
                print(f"✗ Seq length {seq_len}: Failed - {e}")
        
        return results
    
    def run_experiment(self, sample_input: Tuple) -> BenchmarkResult:
        """Run dynamic export experiment"""
        # Export model with dynamic shapes (only once)
        if self.exported_model is None:
            self.exported_model, export_time = self.export_model_dynamic(sample_input)
            print(f"Model exported with dynamic shapes in {export_time:.4f}s")
        else:
            export_time = 0.0  # Already exported
        
        # Test with multiple sequence lengths
        test_results = self.test_multiple_lengths(self.exported_model)
        
        # Calculate average inference time
        successful_results = [r for r in test_results if r["success"]]
        avg_inference_time = sum(r["inference_time"] for r in successful_results) / len(successful_results) if successful_results else 0
        
        return BenchmarkResult(
            sample_index=0,
            export_time=export_time,
            inference_time=avg_inference_time,
            metadata={
                "dynamic_shapes": f"[{self.min_seq_len}, {self.max_seq_len}]",
                "test_results": test_results,
                "successful_tests": len(successful_results),
                "total_tests": len(test_results)
            }
        )