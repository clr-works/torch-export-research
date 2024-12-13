from torch.export import Dim, export
import os
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from fms.models import get_model
from fms.models.hf import to_hf_api

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"


# This function exports the model with dynamic sequence length
def export_model(model, tokenizer, max_seq_len=512, batch_size=1):
    class ExportWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, inputs):
            input_ids, attention_mask = inputs
            print("Input IDs Shape:", input_ids.shape)
            print("Attention Mask Shape:", attention_mask.shape)
            print("Input IDs Device:", input_ids.device)
            print("Attention Mask Device:", attention_mask.device)

            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            print("Model Output:", output)

            return output.logits

    # Define the Dynamic Dimension for Sequence Length
    #seq_len_dim = Dim("seq_len", min=1, max=max_seq_len)
    _seq_len = Dim('_seq_len', min=1, max=64) #gpu requirements
    seq_len_dim = 8*_seq_len

    # Generate a sample input with max sequence length
    sample_text = "This is a test sequence for exporting the Granite model."
    inputs = tokenizer(sample_text, return_tensors="pt", padding="max_length", max_length=max_seq_len, truncation=True)
    sample_input_ids = inputs["input_ids"]
    sample_attention_mask = inputs["attention_mask"]

    # some printings for debugging
    print("Input IDs Shape:", sample_input_ids.shape)
    print("Attention Mask Shape:", sample_attention_mask.shape)
    print("Sample Input IDs:", sample_input_ids)
    print("Sample Attention Mask:", sample_attention_mask)
    print("Input IDs Device:", sample_input_ids.device)
    print("Attention Mask Device:", sample_attention_mask.device)

    dynamic_shapes = (
        {1: seq_len_dim},
        {1: seq_len_dim},
    )
    print("Dynamic Shapes Config:", dynamic_shapes)

    export_wrapper = ExportWrapper(model)

    exported_program = export(
        export_wrapper,
        args=((sample_input_ids, sample_attention_mask),),
        dynamic_shapes=(dynamic_shapes,),
    )

    return exported_program


#inference on the exported model  with random inputs
def test_exported_model(exported_program, tokenizer, seq_lens):
    for seq_len in seq_lens:
        print(f"Running Inference with Sequence length: {seq_len}")
        #gen random input text and tokenize it
        input_text = " ".join(["word"] * seq_len)
        inputs = tokenizer(input_text, return_tensors="pt", padding="max_length", max_length=seq_len, truncation=True)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Perform inference with varying sequence lengths
        outputs = exported_program.module()((input_ids, attention_mask))

        print(f"Sequence Length: {seq_len}, Logits Shape: {outputs.shape}")


if __name__ == "__main__":

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths on instance toLLaMA 7B model weights and tokenizer
    model_path = "/home/corina_rios/.cache/huggingface/hub/models--huggyllama--llama-7b/snapshots/4782ad278652c7c71b72204d462d6d01eaaf7549"
    # model_path = "/home/corina/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08"

    # Configuration for the model
    architecture = "llama"  # Specify the architecture
    variant = "7b"  # Specify the variant (e.g., 7B for the smallest model)

    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.half)

    model = get_model(architecture, variant, model_path=model_path, source="hf", device_type="cuda", norm_eps=1e-6)
    model = to_hf_api(model)

    #tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)  # Use Hugging Face's AutoTokenizer

    # Handle the padding token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            print("Setting pad_token to eos_token.")
            tokenizer.pad_token = tokenizer.eos_token
        else:
            print("Adding a new [PAD] token as pad_token.")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))

            # Export the model with dynamic sequence length
    exported_program = export_model(model, tokenizer, max_seq_len=512, batch_size=1)

    print("Exported Model with Dynamic Sequence Length")
    print(exported_program)

    # Test exported model inference with different sequence lengths
    seq_lens = [8,32, 128]  # Sequence lengths to test
    test_exported_model(exported_program, tokenizer, seq_lens)