from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import time
from datasets import load_dataset
import os
import transformers
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from fms.models import get_model
from fms.models.hf import to_hf_api
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
            return self.model(input_ids=input_ids, attention_mask=attention_mask)

    export_wrapper = ExportWrapper(model)  # a new instance of the model is instantiated with each sample
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
    logits = output.logits
    print("finished calculating logits")
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

    # PathsLLaMA 7B model and tokenizer
    model_path = "/home/corina_rios/.cache/huggingface/hub/models--huggyllama--llama-7b/snapshots/4782ad278652c7c71b72204d462d6d01eaaf7549"
    # model_path = "/home/corina/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08"

    # Configuration for the model
    architecture = "llama"  # Specify the architecture
    variant = "7b"  # Specify the variant (e.g., 7B for the smallest model)

    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.half)

    model = get_model(architecture, variant, model_path=model_path, source="hf", device_type="cuda", norm_eps=1e-6)
    model = to_hf_api(model)

    # tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Open CSV file for writing (persist resulsts)
    with open("export_inference_times.csv", mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Sample Index", "Export Time (s)", "Inference Time (s)"])

        # Loop through the dataset
        print("Processing dataset samples...")
        for i, row in enumerate(train_dataset):
            if i ==10: # break after certain num of  samples
                print(f"Early stopping after processing {i} samples.")
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

            # inference over exported model handling out of mem:
            try:
                torch.cuda.empty_cache()
                _, inference_time = run_exported_inference(exported_model, sample_input)
                print(f"Sample {i}: Inference completed in {inference_time:.4f} seconds.")
            except Exception as e:
                print(f"Sample {i}: Failed to run inference due to {e}")
                continue

            # total time per sample
            total_time = export_time + inference_time
            print(f"Sample {i}: Total time = {total_time:.4f} seconds.")

            # writing results to CSV
            csv_writer.writerow([i, export_time, inference_time])


if __name__ == "__main__":
    main()