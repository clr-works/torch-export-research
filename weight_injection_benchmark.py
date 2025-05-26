from benchmark_framework import BenchmarkBase, BenchmarkResult
import torch
import json
import torch
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any
import csv
from pathlib import Path

class WeightInjectionBenchmark(BenchmarkBase):
    """Benchmark for weight injection experiments"""
    
    def inject_weights(self, model, weight_source="random"):
        """Inject weights with timing"""
        def _inject():
            with torch.no_grad():
                if weight_source == "random":
                    for param in model.parameters():
                        param.data.copy_(torch.randn_like(param.data))
                # Add other weight sources here
        
        _, inject_time = self.time_operation(_inject)
        return inject_time
    
    def run_experiment(self, sample_input: Tuple) -> BenchmarkResult:
        # Export model once
        exported_model, export_time = self.export_model(sample_input)
        
        # Inject weights
        inject_time = self.inject_weights(exported_model)
        
        # Run inference with new weights
        _, inference_time = self.run_inference(exported_model, sample_input)
        
        return BenchmarkResult(
            sample_index=0,
            export_time=export_time,
            inference_time=inference_time,
            inject_time=inject_time
        )

