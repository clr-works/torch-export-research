import torch
from torch.export import export

# Define the model
class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, padding=1
        )
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3)

    def forward(self, x: torch.Tensor, *, constant=None) -> torch.Tensor:
        a = self.conv(x)
        a.add_(constant)
        return self.maxpool(self.relu(a))

# Create the model and export it
example_args = (torch.randn(1, 3, 256, 256),)
example_kwargs = {"constant": torch.ones(1, 16, 256, 256)}
exported_program: torch.export.ExportedProgram = export(
    M(), args=example_args, kwargs=example_kwargs
)

# Access the state_dict
state_dict = exported_program.module().state_dict()

# Generate new weights and biases
new_weights = torch.randn_like(state_dict['conv.weight'])
new_biases = torch.randn_like(state_dict['conv.bias'])

# Update the state_dict
state_dict['conv.weight'] = new_weights
state_dict['conv.bias'] = new_biases

# Verify the updated state_dict
print("Updated state_dict:")
print(state_dict)

# Perform inference to ensure everything works with updated weights
sample_input = torch.randn(1, 3, 256, 256)
new_constant = torch.ones(1, 16, 256, 256)

# Perform inference with the updated state_dict
exported_program.module().load_state_dict(state_dict)
output = exported_program.module()(sample_input, constant=new_constant)
print("Output after updating weights:", output)
