import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time as time_module  # Rename the import
from collections import Counter

def analyze_graph_nodes(model_path, model_name):
    print(f"\nAnalyzing {model_name}")
    
    # Load model
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
    start_time = time_module.time()  # Use the renamed module
    exported = torch.export.export(wrapper, args=((inputs["input_ids"], inputs["attention_mask"]),))
    export_duration = time_module.time() - start_time  # Changed variable name
    
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

# Test models
models = {
        "llama-7b": "/home/corina_rios/.cache/huggingface/hub/models--huggyllama--llama-7b/snapshots/4782ad278652c7c71b72204d462d6d01eaaf7549",
        "granite-2b": "/home/corina_rios/.cache/huggingface/hub/models--ibm-granite--granite-3.0-2b-instruct/snapshots/69e41fe735f54cec1792de2ac4f124b6cc84638f"
    }
results = {}
for name, path in models.items():
    duration, nodes = analyze_graph_nodes(path, name) 
    results[name] = {"time": duration, "nodes": nodes}



