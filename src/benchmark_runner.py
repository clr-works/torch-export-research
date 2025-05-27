import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import os
import json
import torch

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"

class BenchmarkRunner:
    """Orchestrates benchmark execution"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.dataset = None
        
    def load_dataset(self, dataset_name: str = "wikitext", split: str = "wikitext-2-raw-v1"):
        """Load dataset for benchmarking"""
        print(f"Loading dataset {dataset_name}...")
        dataset = load_dataset(dataset_name, split)
        self.dataset = dataset["train"]
        
    def run_benchmark(self, 
                     benchmark_class, 
                     model_name: str, 
                     num_samples: int = 100,
                     **kwargs):
        """Run a specific benchmark"""
        if model_name not in self.config.get("models", {}):
            raise ValueError(f"Model '{model_name}' not found in configuration")
        
        model_path = self.config["models"][model_name]
        
        # Initialize benchmark
        benchmark = benchmark_class(model_name, model_path, **kwargs)
        
        # Load model and tokenizer
        print(f"Loading model '{model_name}' from {model_path}...")
        benchmark.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="cuda" if torch.cuda.is_available() else "cpu"
        )
        benchmark.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Run experiments
        results = []
        valid_samples = 0
        
        print(f"Running {benchmark_class.__name__} for up to {num_samples} samples...")
        
        for i, row in enumerate(self.dataset):
            if valid_samples >= num_samples:
                break
                
            text = row["text"].strip()
            if not text:
                continue
                
            # Prepare input
            inputs = benchmark.tokenizer(text, return_tensors="pt")
            sample_input = (
                inputs["input_ids"].to(benchmark.device),
                inputs["attention_mask"].to(benchmark.device)
            )
            
            try:
                # Clear cache before each experiment
                if benchmark.device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                # Run experiment
                result = benchmark.run_experiment(sample_input)
                result.sample_index = valid_samples
                results.append(result)
                
                valid_samples += 1
                
                # Progress update
                if valid_samples % 10 == 0:
                    print(f"Processed {valid_samples}/{num_samples} samples")
                    
            except Exception as e:
                print(f"Sample {i}: Failed due to {e}")
                continue
        
        # Save results
        benchmark.save_results(results)
        print(f"Results saved for {benchmark_class.__name__}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Model Benchmark Suite")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON configuration")
    parser.add_argument("--model", type=str, required=True, help="Model name from config")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--benchmark", type=str, default="export", 
                       choices=["export", "inject", "both", "dynamic", "dynamic-inject"],  # ✅ Added dynamic-inject
                       help="Which benchmark to run")
    
    # Add dynamic shape arguments
    parser.add_argument("--min-seq-len", type=int, default=1, 
                       help="Minimum sequence length for dynamic export")
    parser.add_argument("--max-seq-len", type=int, default=512, 
                       help="Maximum sequence length for dynamic export")
    parser.add_argument("--test-seq-lengths", type=str, default="8,32,64,128,256",
                       help="Comma-separated sequence lengths to test")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = BenchmarkRunner(args.config)
    runner.load_dataset()
    
    # Import benchmarks
    from export_benchmark import ExportBenchmark
    from weight_injection_benchmark import WeightInjectionBenchmark
    from dynamic_export_benchmark import DynamicExportBenchmark
    from dynamic_weight_injection_benchmark import DynamicWeightInjectionBenchmark  # ✅ Add import
    
    # Parse test sequence lengths
    test_seq_lengths = [int(x) for x in args.test_seq_lengths.split(",")]
    
    # Run selected benchmarks
    if args.benchmark in ["export", "both"]:
        runner.run_benchmark(ExportBenchmark, args.model, args.samples)
    
    if args.benchmark in ["inject", "both"]:
        runner.run_benchmark(WeightInjectionBenchmark, args.model, args.samples)
    
    if args.benchmark == "dynamic":
        runner.run_benchmark(
            DynamicExportBenchmark, 
            args.model, 
            args.samples,
            min_seq_len=args.min_seq_len,
            max_seq_len=args.max_seq_len,
            test_seq_lengths=test_seq_lengths
        )
    
    if args.benchmark == "dynamic-inject":  # ✅ Add this handler
        runner.run_benchmark(
            DynamicWeightInjectionBenchmark, 
            args.model, 
            args.samples,
            min_seq_len=args.min_seq_len,
            max_seq_len=args.max_seq_len,
            test_seq_lengths=test_seq_lengths
        )

if __name__ == "__main__":
    main()