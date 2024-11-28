import csv
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import time
from datasets import load_dataset
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"

"""
This script processes a dataset using a pre-trained causal language model.
It includes steps to:
1. Export the model for each dataset sample.
2. Inject weights in the state dictionary of the exported model
2. Run inference using the exported model.
3. Record inject and inference times in a CSV file for analysis.

The code is designed for performance profiling on GPU, with support for early stopping after 100 samples.
"""


# Function to inject random weights in place
# this will overwrite the state dictionary of hte exported model
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

# function to export model
def export_model(model, sample_input):
    """
    Wraps the model for export with a specific input size.
    Assumes each dataset sample defines a unique input signature.
    """
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

# Run inference using the exported model
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
    # loading hte dataset
    print("Loading dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_dataset = dataset["train"]

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model configuration and architecture
    model_name = "ibm-granite/granite-3.0-2b-instruct"
    print("Loading model and tokenizer directly onto GPU...")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Open CSV file for writing
    with open("inject_inference_times.csv", mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Sample Index", "Inject Time (s)", "Inference Time (s)"])

        # loop through the dataset
        print("Processing dataset samples...")
        for i, row in enumerate(train_dataset):
            if i == 100:
                print("Early stopping after processing 100 samples.")
                break

            text = row["text"].strip()
            if not text:  # there are some empty rows that were found
                continue

            # tokenize the text
            inputs = tokenizer(text, return_tensors="pt")
            sample_input = (inputs["input_ids"].to(device), inputs["attention_mask"].to(device))

            # export the model
            try:
                torch.cuda.empty_cache()
                exported_model, _ = export_model(model, sample_input)
            except Exception as e:
                print(f"Sample {i}: Failed to export model due to {e}")
                continue

            # inject random weights into the exported model
            try:
                torch.cuda.empty_cache()
                exported_model, inject_time = inject_random_weights_inplace(exported_model)
                print(f"Sample {i}: Random weights injected in {inject_time:.4f} seconds.")
            except Exception as e:
                print(f"Sample {i}: Failed to inject random weights due to {e}")
                continue

            #Run inference
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








