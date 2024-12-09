from transformers import AutoModelForCausalLM
import torch
from torch.export import Dim, export
import torch.nn as nn

# Define a wrapper to align inputs with Granite's forward method
class GraniteWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


if __name__ == "__main__":
    # Load IBM Granite Model
    model_name = "ibm-granite/granite-3.0-2b-instruct"
    print("Loading IBM Granite model directly onto GPU...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",   
        torch_dtype=torch.float16   
    )

    # Wrap the model
    granite_wrapper = GraniteWrapper(model)

    # Define the Dynamic Dimension for Sequence Length
    max_seq_len = 512  # Cap for the sequence length
    seq_len_dim = Dim("seq_len", min=1, max=max_seq_len)

    # Generate random inputs
    batch_size = 1
    seq_len = 64 
    vocab_size = 50257 

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.int64).to("cuda")
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.int64).to("cuda")

    print("Input IDs Shape:", input_ids.shape)
    print("Attention Mask Shape:", attention_mask.shape)
    print("Input IDs Device:", input_ids.device)
    print("Attention Mask Device:", attention_mask.device)

    # define the dynamic shapes
    dynamic_shapes = {
        "input_ids": {1: seq_len_dim},     
        "attention_mask": {1: seq_len_dim}
    }

    # Export the Model
    try:
        print("Exporting the Granite model with dynamic sequence length...")
        exported_program = export(
            granite_wrapper,
            args=(input_ids, attention_mask),
            dynamic_shapes=dynamic_shapes
        )
        print("Model successfully exported with dynamic sequence length!")
        print(exported_program)

    except Exception as e:
        print(f"Failed to export the model: {e}")
