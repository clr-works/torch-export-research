import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.export import export

# Define the model with packed tensor input
class PackedInputSimpleCNN(nn.Module):
    def __init__(self):
        super(PackedInputSimpleCNN, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, packed_tensor):
        # Calculate and unpack each component based on its offset and shape
        x = packed_tensor[:3 * 32 * 32].view(1, 3, 32, 32)

        start = 3 * 32 * 32
        conv1_weight = packed_tensor[start:start + 32 * 3 * 3 * 3].view(32, 3, 3, 3)
        start += 32 * 3 * 3 * 3

        conv1_bias = packed_tensor[start:start + 32].view(32)
        start += 32

        conv2_weight = packed_tensor[start:start + 64 * 32 * 3 * 3].view(64, 32, 3, 3)
        start += 64 * 32 * 3 * 3

        conv2_bias = packed_tensor[start:start + 64].view(64)
        start += 64

        fc1_weight = packed_tensor[start:start + 128 * 64 * 8 * 8].view(128, 64 * 8 * 8)
        start += 128 * 64 * 8 * 8

        fc1_bias = packed_tensor[start:start + 128].view(128)
        start += 128

        fc2_weight = packed_tensor[start:start + 10 * 128].view(10, 128)
        start += 10 * 128

        fc2_bias = packed_tensor[start:start + 10].view(10)

        # Apply the layers using the unpacked weights
        x = self.pool(self.relu(nn.functional.conv2d(x, conv1_weight, conv1_bias, stride=1, padding=1)))
        x = self.pool(self.relu(nn.functional.conv2d(x, conv2_weight, conv2_bias, stride=1, padding=1)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(nn.functional.linear(x, fc1_weight, fc1_bias))
        return nn.functional.linear(x, fc2_weight, fc2_bias)


# Function to pack inputs into a single tensor
def pack_inputs(input_tensor, weights):
    packed_input = torch.cat([
        input_tensor.view(-1),
        weights['conv1.weight'].view(-1),
        weights['conv1.bias'].view(-1),
        weights['conv2.weight'].view(-1),
        weights['conv2.bias'].view(-1),
        weights['fc1.weight'].view(-1),
        weights['fc1.bias'].view(-1),
        weights['fc2.weight'].view(-1),
        weights['fc2.bias'].view(-1)
    ])
    return packed_input

# Testing function with dynamically injected weights
def test_exported_program_with_dynamic_weights(exported_program, input_data, weights):
    packed_input = pack_inputs(input_data, weights)
    with torch.no_grad():
        output = exported_program.module(packed_input)
        print("Output with dynamically injected weights:", output)

if __name__ == "__main__":
    # Load CIFAR-10 dataset from the local data folder without redownloading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Added std values
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.relu = nn.ReLU()
            self.fc1 = nn.Linear(64 * 8 * 8, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(-1, 64 * 8 * 8)
            x = self.relu(self.fc1(x))
            return self.fc2(x)

    # Initialize the model and set up training components
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizers = {
        "SGD": optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
        "Adam": optim.Adam(model.parameters(), lr=0.001),
    }
    trained_weights = {}

    # Train with each optimizer and save weights
    for opt_name, optimizer in optimizers.items():
        model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
        for epoch in range(2):  # demo
            for images, labels in train_loader:
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
        trained_weights[opt_name] = model.state_dict().copy()

    # Export the model with packed inputs
    dynamic_model = PackedInputSimpleCNN()
    input_tensor = torch.randn(1, 3, 32, 32)
    example_weights = {
        'conv1.weight': torch.randn(32, 3, 3, 3),
        'conv1.bias': torch.randn(32),
        'conv2.weight': torch.randn(64, 32, 3, 3),
        'conv2.bias': torch.randn(64),
        'fc1.weight': torch.randn(128, 64 * 8 * 8),
        'fc1.bias': torch.randn(128),
        'fc2.weight': torch.randn(10, 128),
        'fc2.bias': torch.randn(10),
    }
    packed_example_input = pack_inputs(input_tensor, example_weights)
    exported_program = export(dynamic_model, (packed_example_input,))

    input_data = torch.randn(1, 3, 32, 32)
    for opt_name, weights in trained_weights.items():
        print(f"\nTesting with weights from optimizer: {opt_name}")
        test_exported_program_with_dynamic_weights(exported_program, input_data, weights)

