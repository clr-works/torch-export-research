# dynamic_weight_injection_benchmark.py
from benchmark_framework import BenchmarkBase, BenchmarkResult
from dynamic_export_benchmark import DynamicExportBenchmark
import torch
from typing import Tuple, List
from torch.export import Dim

_seq_len = Dim('_seq_len', min=16, max=128)
seq_len = 4 * _seq_len  # Gives you 64, 128, 256, 512

class DynamicWeightInjectionBenchmark(DynamicExportBenchmark):
    """Test weight injection with dynamic shape models"""
    
    def inject_weights(self, model, weight_source="random"):
        """Inject weights with timing"""
        def _inject():
            with torch.no_grad():
                if weight_source == "random":
                    for param in model.parameters():
                        param.data.copy_(torch.randn_like(param.data))
        
        _, inject_time = self.time_operation(_inject)
        return inject_time
    
    def run_experiment(self, sample_input: Tuple) -> BenchmarkResult:
        """Test weight injection with dynamic shapes"""
        # Export model with dynamic shapes
        if self.exported_model is None:
            self.exported_model, export_time = self.export_model_dynamic(sample_input)
            print(f"Model exported with dynamic shapes in {export_time:.4f}s")
        else:
            export_time = 0.0
        
        # Test 1: Before weight injection
        print("\n--- Testing BEFORE weight injection ---")
        results_before = self.test_multiple_lengths(self.exported_model)
        
        # Inject weights
        inject_time = self.inject_weights(self.exported_model)
        print(f"\n💉 Weights injected in {inject_time:.4f}s")
        
        # Test 2: After weight injection
        print("\n--- Testing AFTER weight injection ---")
        results_after = self.test_multiple_lengths(self.exported_model)
        
        # Compare results
        comparison = []
        for before, after in zip(results_before, results_after):
            if before["success"] and after["success"]:
                comparison.append({
                    "seq_len": before["seq_len"],
                    "time_before": before["inference_time"],
                    "time_after": after["inference_time"],
                    "time_diff": after["inference_time"] - before["inference_time"]
                })
        
        # Calculate averages
        avg_before = sum(r["time_before"] for r in comparison) / len(comparison) if comparison else 0
        avg_after = sum(r["time_after"] for r in comparison) / len(comparison) if comparison else 0
        
        return BenchmarkResult(
            sample_index=0,
            export_time=export_time,
            inference_time=avg_after,
            inject_time=inject_time,
            metadata={
                "dynamic_shapes": f"[{self.min_seq_len}, {self.max_seq_len}]",
                "results_before": results_before,
                "results_after": results_after,
                "comparison": comparison,
                "avg_time_before": avg_before,
                "avg_time_after": avg_after,
                "performance_impact": (avg_after - avg_before) / avg_before * 100 if avg_before > 0 else 0
            }
        )