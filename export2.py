import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torch.export import export
#import torch._dynamo
#torch._dynamo.config.suppress_errors = True


# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def test_exported_program(exported_program, weights, input_tensor):
    # Map model's state_dict keys to the graph's expected parameter names
    inputs = {}
    for param in exported_program.graph_signature.input_specs:
        # Check if the parameter's target matches a key in the weights
        if param.target in weights:
            inputs[param.arg.name] = weights[param.target]

    # Use the provided input tensor directly
    inputs["input"] = input_tensor

    # Run the exported program using module() to access the callable
    with torch.no_grad():
        # explanation of the error
        # type error: 'ExportedProgram' got an unexpected keyword argument 'p_conv1_weight
        # so these parameters names are not expected to be passed as arguments.
        # ExportedProgram.module already holds references to these weights internally, so they
        # are not designed to be passed as inputs
        output = exported_program.module(**inputs)  # Use .module() here
        print("Output with loaded weights:", output)

if __name__ == "__main__":

    # Load the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Initialize the model
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()

    # Set up example inputs for export
    example_inputs = (torch.randn(1, 1, 28, 28),)
    exported_program = export(model, example_inputs)

    for param in exported_program.graph_signature.input_specs:
        print(f"Exported parameter name: {param.arg.name}")

    # Different optimizers to try
    optimizers = {
        "SGD": optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
        "Adam": optim.Adam(model.parameters(), lr=0.001),
    }

    # Dictionary to store different sets of weights
    trained_weights = {}
    for opt_name, optimizer in optimizers.items():
        # Reset model parameters for each optimizer's training session
        def reset_params(module):
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()


        model.apply(reset_params)

        for epoch in range(2):  #it is just a demo
            for images, labels in train_loader:
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

        # Print to confirm that all model layers are captured
        print(f"Trained weights keys for {opt_name}: {list(model.state_dict().keys())}")
        trained_weights[opt_name] = model.state_dict().copy()

    # Test each set of trained weights with the exported program
    input_tensor = torch.ones(1, 1, 28, 28)
    for opt_name, weights in trained_weights.items():
        print(f"\nTesting with weights from optimizer: {opt_name}")
        test_exported_program(exported_program, weights, input_tensor)
