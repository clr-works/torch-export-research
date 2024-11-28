from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import time
from datasets import load_dataset
import os
import csv

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"

"""
This script processes a dataset using a pre-trained causal language model.
It includes steps to:
1. Export the model for each dataset sample.
2. Run inference using the exported model.
3. Record export and inference times in a CSV file for analysis.

The code is designed for performance profiling on GPU, with support for early stopping after 100 samples.
"""

# Function to export model
# a wrapper is defined in order to export the model
def export_model(model, sample_input):
    class ExportWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, inputs):
            input_ids, attention_mask = inputs
            return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    export_wrapper = ExportWrapper(model) # a new instance of the model is instantiated with each sample
    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    exported_program = torch.export.export(export_wrapper, args=(sample_input,))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    export_time = time.time() - start_time
    return exported_program, export_time

# run inference using the exported model
# run inference once per sample
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
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_dataset = dataset["train"]

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model configuration and architecture
    # load will be managed from gpu due to cpu memory restrictions
    model_name = "ibm-granite/granite-3.0-2b-instruct"
    print("Loading model and tokenizer directly onto GPU...")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Open CSV file for writing (persist resulsts)
    with open("export_inference_times.csv", mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Sample Index", "Export Time (s)", "Inference Time (s)"])

        # Loop through the dataset
        print("Processing dataset samples...")
        for i, row in enumerate(train_dataset):
            if i == 100:  # break after 100 samples
                print("Early stopping after processing 100 samples.")
                break

            text = row["text"].strip()
            if not text:  # skip empty rows
                continue

            # tokenize the text
            inputs = tokenizer(text, return_tensors="pt")
            sample_input = (inputs["input_ids"].to(device), inputs["attention_mask"].to(device))

            # Export the model
            try:
                torch.cuda.empty_cache()
                exported_model, export_time = export_model(model, sample_input)
                print(f"Sample {i}: Model exported in {export_time:.4f} seconds.")
            except Exception as e:
                print(f"Sample {i}: Failed to export model due to {e}")
                continue

            # Run inference over exported model:
            try:
                torch.cuda.empty_cache()
                _, inference_time = run_exported_inference(exported_model, sample_input)
                print(f"Sample {i}: Inference completed in {inference_time:.4f} seconds.")
            except Exception as e:
                print(f"Sample {i}: Failed to run inference due to {e}")
                continue

            # Total time per sample
            total_time = export_time + inference_time
            print(f"Sample {i}: Total time = {total_time:.4f} seconds.")

            # writing results to CSV
            csv_writer.writerow([i, export_time, inference_time])

if __name__ == "__main__":
    main()




