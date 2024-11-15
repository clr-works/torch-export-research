import os
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 16)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(16, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

with torch.no_grad():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model().to(device=device)
    example_inputs=(torch.randn(256, 10, device=device),)
    batch_dim = torch.export.Dim("batch", min=1, max=1024)
    so_path = torch._export.aot_compile(
        model,
        example_inputs,
        # Specify the first dimension of the input x as dynamic
        dynamic_shapes={"x": {0: batch_dim}},
        # Specify the generated shared library path
        options={"aot_inductor.output_path": os.path.join(os.getcwd(), "model.so")},
    )