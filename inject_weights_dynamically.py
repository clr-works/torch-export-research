import csv
import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
from datasets import load_dataset
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"

"""
This script processes a dataset using a pre-trained causal language model.
It includes steps to:
1. Dynamically load a model based on a command-line argument from a JSON configuration file.
2. Export the model for each dataset sample.
3. Inject random weights into the exported model.
4. Run inference using the exported model.
5. Record inject and inference times in a CSV file for analysis.
"""

# Function to inject random weights in place
def inject_random_weights_inplace(model):
    """Overwrite state dictionary with random tensor."""
    start_time = time.time()
    with torch.no_grad():
        for param in model.parameters():
            param.data.copy_(torch.randn_like(param.data))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inject_time = time.time() - start_time
    return model, inject_time

# Function to export model
def export_model(model, sample_input):
    """Wraps the model for export with a specific input size."""
    class ExportWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, inputs):
            input_ids, attention_mask = inputs
            return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    export_wrapper = ExportWrapper(model)
    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    exported_program = torch.export.export(export_wrapper, args=(sample_input,))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    export_time = time.time() - start_time
    return exported_program, export_time

# Function to run inference
def run_exported_inference(exported_model, sample_input):
    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    output = exported_model.module()(sample_input)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_time = time.time() - start_time
    return output, inference_time

# Main function
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Model Export, Inject, and Inference Benchmark")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON configuration file.")
    parser.add_argument("--model", type=str, required=True, help="Model name to load from the configuration.")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to process (default: 100).")
    args = parser.parse_args()

    # Load JSON configuration
    with open(args.config, "r") as f:
        config = json.load(f)

    models = config.get("models", {})
    if args.model not in models:
        print(f"Error: Model '{args.model}' not found in the configuration file.")
        return

    model_name = args.model
    model_path = models[model_name]

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_dataset = dataset["train"]

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    print(f"Loading model '{model_name}' from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Open CSV file for results
    csv_filename = f"{model_name}_inject_inference_times.csv"
    with open(csv_filename, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Sample Index", "Inject Time (s)", "Inference Time (s)"])

        # Process dataset samples
        print(f"Processing up to {args.samples} samples...")
        for i, row in enumerate(train_dataset):
            if i == args.samples:  # Stop after the specified number of samples
                print(f"Early stopping after processing {args.samples} samples.")
                break

            text = row["text"].strip()
            if not text:  # Skip empty rows
                continue

            # Tokenize the text
            inputs = tokenizer(text, return_tensors="pt")
            sample_input = (inputs["input_ids"].to(device), inputs["attention_mask"].to(device))

            # Export the model
            try:
                torch.cuda.empty_cache()
                exported_model, _ = export_model(model, sample_input)
            except Exception as e:
                print(f"Sample {i}: Failed to export model due to {e}")
                continue

            # Inject random weights
            try:
                torch.cuda.empty_cache()
                exported_model, inject_time = inject_random_weights_inplace(exported_model)
                print(f"Sample {i}: Random weights injected in {inject_time:.4f} seconds.")
            except Exception as e:
                print(f"Sample {i}: Failed to inject random weights due to {e}")
                continue

            # Run inference
            try:
                torch.cuda.empty_cache()
                _, inference_time = run_exported_inference(exported_model, sample_input)
                print(f"Sample {i}: Inference completed in {inference_time:.4f} seconds.")
            except Exception as e:
                print(f"Sample {i}: Failed to run inference due to {e}")
                continue

            # Write results to CSV
            csv_writer.writerow([i, inject_time, inference_time])

if __name__ == "__main__":
    main()








