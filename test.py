import torch
from torch.export import export

class Mod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)  # Example layer with weight and bias

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        a = torch.sin(x)
        b = torch.cos(y)
        return self.linear(a + b)

# Define a sample model
model = Mod()

# Export the model
example_args = (torch.randn(10, 10), torch.randn(10, 10))
print(type(example_args))

exported_program = export(
    Mod(), args=example_args
)

print("Before modification, state_dict:", exported_program.module().state_dict())

# Create random tensors for weight and bias
new_state_dict = {
    "linear.weight": torch.randn_like(exported_program.module().state_dict()["linear.weight"]),
    "linear.bias": torch.randn_like(exported_program.module().state_dict()["linear.bias"]),
}

# Overwrite the state_dict
exported_program.module().load_state_dict(new_state_dict)

print("After modification, state_dict:", exported_program.module().state_dict())

