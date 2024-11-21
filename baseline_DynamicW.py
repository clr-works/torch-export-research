import torch
from torch import nn
from torch.export import export

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 5)  # Linear layer with 10 inputs, 5 outputs

    def forward(self, x):
        return self.linear(x)

# initialize the model
model = SimpleModel()

# Inject custom weights into the model's state_dict
with torch.no_grad():  #run gradients off
    model.state_dict()['linear.weight'].copy_(torch.randn(5, 10))
    model.state_dict()['linear.bias'].copy_(torch.randn(5))

#Verify that the weights were injected
print("Injected weights and biases:\n")
for name, param in model.state_dict().items():
    print(f"{name}: {param}\n")

sample_input = torch.randn(1, 10)

exported_program = export(model, (sample_input,))

new_input = torch.randn(1, 10)

output = exported_program.module()(new_input)
print("Inference output with injected weights:", output)

