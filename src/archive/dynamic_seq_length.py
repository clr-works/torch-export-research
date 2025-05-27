import json
import argparse
from torch.export import Dim, export
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fms.models import get_model
from fms.models.hf import to_hf_api

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"

"""
This script does the dynamic export and testing of causal language models 
with varying sequence lengths. It supports two model loading mechanisms:
1. For the `llama-7b` model, it uses IBM's Foundational Model Stack (FMS) wrapper 
   to load the model and tokenizer.
2. For other models, it relies on Hugging Face's `transformers` library.

The script performs the following steps:
1. Loads the model and tokenizer based on the model name provided via a JSON configuration file.
2. Dynamicaly exporst the model using `torch.export`, allowing for flexible input shapes.
3. Tests the exported model by running inference on random inputs of varying sequence lengths.

Key Features:
- Handles `llama-7b` with FMS-specific logic and loads other models using standard Hugging Face APIs.
- Ensures compatibility with GPU or CPU environments.
- Supports dynamic sequence lengths to test model flexibility and performance.

Command-Line Arguments:
- `--config`: Path to the JSON configuration file containing model paths.
- `--model`: Name of the model to load (must match a key in the JSON configuration).
- `--seq_lens`: Comma-separated list of sequence lengths to test (default: 8,32,128).

Example Usage:
1. Create a JSON file `models.json` with the following structure:
   {
       "models": {
           "llama-7b": "/path/to/llama-7b",
           "granite-2b": "/path/to/granite-2b"
       }
   }
2. Run the script:
   python script.py --config models.json --model llama-7b --seq_lens 8,32,128

Outputs:
- The script logs model export and inference results for each sequence length.
"""


# Function to export the model with dynamic sequence length
def export_model(model, tokenizer, max_seq_len=512):
    class ExportWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, inputs):
            input_ids, attention_mask = inputs
            return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    #seq_len_dim = Dim("seq_len", min=1, max=max_seq_len)
    _seq_len = Dim('_seq_len', min=1, max=64)
    seq_len_dim = 8*_seq_len

    # Generate a sample input
    sample_text = "This is a test sequence for exporting the model."
    inputs = tokenizer(sample_text, return_tensors="pt", padding="max_length", max_length=max_seq_len, truncation=True)
    sample_input_ids = inputs["input_ids"]
    sample_attention_mask = inputs["attention_mask"]

    dynamic_shapes = (
        {1: seq_len_dim},
        {1: seq_len_dim},
    )

    export_wrapper = ExportWrapper(model)

    exported_program = export(
        export_wrapper,
        args=((sample_input_ids, sample_attention_mask),),
        dynamic_shapes=(dynamic_shapes,),
    )

    return exported_program

# Function to test the exported model with varying sequence lengths
def test_exported_model(exported_program, tokenizer, seq_lens):
    for seq_len in seq_lens:
        print(f"Running Inference with Sequence Length: {seq_len}")
        input_text = " ".join(["word"] * seq_len)
        inputs = tokenizer(input_text, return_tensors="pt", padding="max_length", max_length=seq_len, truncation=True)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        outputs = exported_program.module()((input_ids, attention_mask))
        print(f"Sequence Length: {seq_len}, Logits Shape: {outputs.shape}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Model Export with Dynamic Sequence Lengths")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON configuration file.")
    parser.add_argument("--model", type=str, required=True, help="Model name to load from the configuration.")
    parser.add_argument("--seq_lens", type=str, required=False, default="8,32,128",
                        help="Comma-separated sequence lengths to test (default: 8,32,128).")
    args = parser.parse_args()

    # Load JSON configuration
    with open(args.config, "r") as f:
        config = json.load(f)

    models = config.get("models", {})
    if args.model not in models:
        print(f"Error: Model '{args.model}' not found in the configuration file.")
        exit(1)

    model_path = models[args.model]

    # Parse sequence lengths
    seq_lens = [int(x) for x in args.seq_lens.split(",")]

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    print(f"Loading model '{args.model}' from {model_path}...")
    if args.model == "llama-7b":
        # Use FMS wrapper for llama-7b
        architecture, variant = "llama", "7b"
        torch.set_default_device("cuda")
        torch.set_default_dtype(torch.half)

        model = get_model(architecture, variant, model_path=model_path, source="hf", device_type="cuda", norm_eps=1e-6)
        model = to_hf_api(model)
    else:
        # Load other models directly
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))

    # Export the model
    exported_program = export_model(model, tokenizer, max_seq_len=512)
    print("Model exported successfully.")

    # Test the exported model with specified sequence lengths
    test_exported_model(exported_program, tokenizer, seq_lens)
