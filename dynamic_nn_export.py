import torch
from torch.export import Dim, export

'''The program simulates a language model workflow where a blueprint of a  model is
exported for inference, and then tested for flexibility with dynamic input shapes.'''

# Simplified Model Definition
class SimpleLM(torch.nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        self.linear = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        logits = self.linear(x.mean(dim=1))  # implified; just mean pooling for demo
        return logits # Each value in the output is a raw score (logit) representing the unnormalized
                        # likelihood of each token in the vocabulary for the given input sample.

# Generate inputs of varying sequence lengths
def generate_dynamic_inputs(batch_size, seq_len, vocab_size):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len))
    return input_ids, attention_mask

# Test inference on the exported model
def test_exported_model(exported_program, vocab_size, seq_lens):

    for seq_len in seq_lens:
        # Generate dynamic inputs
        input_ids, attention_mask = generate_dynamic_inputs(batch_size=1, seq_len=seq_len, vocab_size=vocab_size)

        # Perform inference with varying sequence lengths
        outputs = exported_program.module()(input_ids, attention_mask)

        # Output the results
        print(f"Sequence Length: {seq_len}, Logits Shape: {outputs.shape}")

if __name__ == "__main__":

    # Hardcoded Vocabulary Size and Hidden Dimension
    vocab_size = 10000
    hidden_dim = 128
    model = SimpleLM(vocab_size, hidden_dim)

    # Example Inputs with a Fixed Sequence Length
    batch_size = 1
    seq_len = 64
    example_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    example_attention_mask = torch.ones((batch_size, seq_len))

    # Define the Dynamic Dimension for Sequence Length
    seq_len_dim = Dim("seq_len", min=1, max=512) # the dynamic sequence length capped at 512

    # Annotate Dynamic Shapes
    dynamic_shapes = {
        "input_ids": {1: seq_len_dim},       # sequence length in `input_ids`
        "attention_mask": {1: seq_len_dim}  # equence length in `attention_mask`
    }

    # Wrap Inputs for Export
    sample_input = (example_input_ids, example_attention_mask)

    # Export the Model with Dynamic Sequence Length
    exported_program = export(
        model,
        args=sample_input,
        dynamic_shapes=dynamic_shapes,
    )

    # Print Export Summary
    print("Exported Model with Dynamic Sequence Length")
    print(exported_program)

    # test exported model inference with different sequence lenghts
    inputs_ids, attention_mask = generate_dynamic_inputs(batch_size=1, seq_len=32, vocab_size=vocab_size)

    seq_lens = [32, 64, 128, 256, 512]  # Sequence lengths to test

    test_exported_model(exported_program, vocab_size, seq_lens)



