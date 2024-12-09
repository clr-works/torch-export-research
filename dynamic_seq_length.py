from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.export import Dim, export
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"


# Function to export the model with dynamic sequence length
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

    #Define the Dynamic Dimension for Sequence Length
    seq_len_dim = Dim("seq_len", min=1, max=max_seq_len)

    # Generate a sample input with max sequence length
    sample_text = "This is a test sequence for exporting the Granite model."
    inputs = tokenizer(sample_text, return_tensors="pt", padding="max_length", max_length=max_seq_len, truncation=True)
    sample_input_ids = inputs["input_ids"].to("cuda")
    sample_attention_mask = inputs["attention_mask"].to("cuda")

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

    # Wrap the model for export
    export_wrapper = ExportWrapper(model)

    # Export the model
    exported_program = export(
        export_wrapper,
        args=((sample_input_ids, sample_attention_mask),),
        dynamic_shapes=(dynamic_shapes,),
    )

    return exported_program


# Test inference on the exported model
def test_exported_model(exported_program, tokenizer, seq_lens):
    for seq_len in seq_lens:
        # Generate random input text and tokenize it
        input_text = " ".join(["word"] * seq_len)
        inputs = tokenizer(input_text, return_tensors="pt", padding="max_length", max_length=seq_len, truncation=True)

        input_ids = inputs["input_ids"].to("cuda")
        attention_mask = inputs["attention_mask"].to("cuda")

        # Perform inference with varying sequence lengths
        outputs = exported_program.module()((input_ids, attention_mask))

        logits = outputs.logits
        print(f"Sequence Length: {seq_len}, Logits Shape: {logits.shape}")



if __name__ == "__main__":
    
    model_name = "ibm-granite/granite-3.0-2b-instruct"

    print("Loading model and tokenizer directly onto GPUs with reduced memory usage...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",          
        torch_dtype=torch.float16   # 
        use half-precision to reduce memory usage
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    #Export the model with dynamic sequence length
    exported_program = export_model(model, tokenizer, max_seq_len=512, batch_size=1)

    print("Exported Model with Dynamic Sequence Length")
    print(exported_program)

    # Test exported model inference with different sequence lengths
    #seq_lens = [32]  # Sequence lengths to test
    #test_exported_model(exported_program, tokenizer, seq_lens)

