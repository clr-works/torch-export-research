import torch
from torch import nn
from torch.export import export

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)

# Initialize and export the model
model = SimpleModel()
sample_input = torch.randn(1, 10)
exported_program = export(model, (sample_input,))

# Get the state_dict from the exported program's module
exported_state_dict = exported_program.module().state_dict()

# Print each parameter name and its tensor values
print("Parameters in Exported Graph's state_dict:")
for name, param in exported_state_dict.items():
    print(f"{name}: {param}\n")

# Inspect weights and biases using parameters()
print("Parameters in Exported Graph's parameters():")
for param in exported_program.module().parameters():
    print(param)

# Access the state_dict of the exported model's module
exported_state_dict = exported_program.module().state_dict()

# Inject new weights into the exported state_dict
with torch.no_grad():
    exported_state_dict['linear.weight'].copy_(torch.randn(5, 10))  # New weights
    exported_state_dict['linear.bias'].copy_(torch.randn(5))  # New biases

# Verify that the weights were updated
print("Injected weights and biases:")
for name, param in exported_state_dict.items():
    print(f"{name}: {param}\n")

# New input for inference
new_input = torch.randn(1, 10)

# Run inference with the modified weights
output = exported_program.module()(new_input)
print("Inference output with injected weights:", output)
