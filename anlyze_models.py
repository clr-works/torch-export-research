import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time as time_module 
from collections import Counter

"""
This script profiles the export process and graph structure of pre-trained language models.

Steps:
1. Load a model and tokenizer.
2. Ensure the tokenizer has a padding token.
3. Prepare a sample input for the model.
4. Wrap the model for Torch export.
5. Export the model using `torch.export` and measure export time.
6. Analyze the graph structure, including node types and count.

Outputs:
- Export duration.
- Total graph nodes and their types.
- Total model parameters.

Usage:
- Run the script with models specified via command-line arguments:
  python analyze_models.py --models llama-7b /path/to/llama-7b granite-2b /path/to/granite-2b
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter
import time as time_module  # Avoiding `time` conflicts

def analyze_graph_nodes(model_path, model_name):
    """
    Analyzes the export process and graph structure for a model.

    Args:
        model_path (str): Path to the model.
        model_name (str): Name of the model.

    Returns:
        tuple: Export duration (float) and node type distribution (Counter).
    """
    print(f"\nAnalyzing {model_name}...")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Fix padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create sample input
    text = "test"
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", max_length=32)
    
    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, inputs):
            input_ids, attention_mask = inputs
            return self.model(input_ids=input_ids, attention_mask=attention_mask).logits
    
    wrapper = Wrapper(model)
    
    # Export and analyze
    print("Starting export...")
    start_time = time_module.time()
    exported = torch.export.export(wrapper, args=((inputs["input_ids"], inputs["attention_mask"]),))
    export_duration = time_module.time() - start_time
    
    # Analyze node types
    node_types = Counter()
    for node in exported.graph_module.graph.nodes:
        node_types[node.op] += 1
    
    print(f"\nResults for {model_name}:")
    print(f"Export time: {export_duration:.2f}s")
    print(f"Total nodes: {len(exported.graph_module.graph.nodes)}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    print("\nNode types distribution:")
    for op_type, count in node_types.most_common():
        print(f"{op_type}: {count}")
    
    return export_duration, node_types

def main():
    # Load paths from config file
    with open("config.json", "r") as f:
        config = json.load(f)
    
    models = config["models"]
    results = {}

    for name, path in models.items():
        duration, nodes = analyze_graph_nodes(path, name)
        results[name] = {"time": duration, "nodes": nodes}

    # Print final results
    print("\nFinal Results:")
    for model_name, result in results.items():
        print(f"{model_name}: Time = {result['time']:.2f}s, Nodes = {result['nodes']}")

if __name__ == "__main__":
    main()
