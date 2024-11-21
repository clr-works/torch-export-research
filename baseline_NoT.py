import torch
from torch import nn
from torch.export import export

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 5)  # Linear layer with 10 inputs, 5 outputs

    def forward(self, x):
        return self.linear(x)

# Init. the model
model = SimpleModel()

print("Initial weights and biases in the model:\n")
for name, param in model.state_dict().items():
    print(f"{name}: {param}\n")

# Example input for exporting the model
sample_input = torch.randn(1, 10)

#Export the model
exported_program = export(model, (sample_input,))

# Perform inference with the exported model using `module()`
new_input = torch.randn(1, 10)
output = exported_program.module()(new_input)

print("Inference output:", output)
