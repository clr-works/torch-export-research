from transformers import AutoModelForCausalLM
import torch
import traceback

# Function to test Granite model with torch.compile
def test_granite_with_compile():
    # Load IBM Granite Model
    model_name = "ibm-granite/granite-3.0-2b-instruct"
    print("Loading IBM Granite model onto GPU...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16 
    )

    # Compile the model
    try:
        print("Compiling the model with torch.compile...")
        compiled_model = torch.compile(model)
        print("Model successfully compiled!")
    except Exception as e:
        print("Failed to compile the model:")
        traceback.print_exc()
        return

    '''# Generate random inputs
    batch_size = 1
    seq_len = 64
    vocab_size = 50257  # GPT-like vocab size typically used in Granite models

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.int64).to("cuda")
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.int64).to("cuda")

    # Test the compiled model
    try:
        print("Testing the compiled model with random inputs...")
        outputs = compiled_model(input_ids=input_ids, attention_mask=attention_mask)
        print("Model outputs:")
        print(outputs.logits.shape)  # Logits shape should match (batch_size, seq_len, vocab_size)
    except Exception as e:
        print("Failed to run inference with the compiled model:")
        traceback.print_exc()'''

if __name__ == "__main__":
    test_granite_with_compile()
