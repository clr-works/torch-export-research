import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.export import export


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
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


class DynamicWeightCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x, weights):
        # Extract weights in the correct shapes
        start_idx = 0

        # Unpacking conv1 weights and bias
        conv1_w = weights[start_idx:start_idx + 32 * 3 * 3 * 3].view(32, 3, 3, 3)
        start_idx += 32 * 3 * 3 * 3
        conv1_b = weights[start_idx:start_idx + 32].view(32)
        start_idx += 32

        # Unpackingin
        conv2_w = weights[start_idx:start_idx + 64 * 32 * 3 * 3].view(64, 32, 3, 3)
        start_idx += 64 * 32 * 3 * 3
        conv2_b = weights[start_idx:start_idx + 64].view(64)
        start_idx += 64

        # Unpaking
        fc1_w = weights[start_idx:start_idx + 128 * 64 * 8 * 8].view(128, 64 * 8 * 8)
        start_idx += 128 * 64 * 8 * 8
        fc1_b = weights[start_idx:start_idx + 128].view(128)
        start_idx += 128

        # Unpacking
        fc2_w = weights[start_idx:start_idx + 10 * 128].view(10, 128)
        start_idx += 10 * 128
        fc2_b = weights[start_idx:start_idx + 10].view(10)

        # Now the forward pass with unpacked weights
        x = self.pool(self.relu(nn.functional.conv2d(x, conv1_w, conv1_b, padding=1)))
        x = self.pool(self.relu(nn.functional.conv2d(x, conv2_w, conv2_b, padding=1)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(nn.functional.linear(x, fc1_w, fc1_b))
        x = nn.functional.linear(x, fc2_w, fc2_b)
        return x


def flatten_state_dict(state_dict):
    """Convert a state dict into a single flattened tensor"""
    return torch.cat([
        state_dict['conv1.weight'].view(-1),
        state_dict['conv1.bias'].view(-1),
        state_dict['conv2.weight'].view(-1),
        state_dict['conv2.bias'].view(-1),
        state_dict['fc1.weight'].view(-1),
        state_dict['fc1.bias'].view(-1),
        state_dict['fc2.weight'].view(-1),
        state_dict['fc2.bias'].view(-1)
    ])


def export_model_with_dynamic_weights():
    model = DynamicWeightCNN()

    # Create dummy input and weights
    dummy_input = torch.randn(1, 3, 32, 32)
    dummy_weights = torch.randn(
        32 * 3 * 3 * 3 +  # conv1.weight
        32 +  # conv1.bias
        64 * 32 * 3 * 3 +  # conv2.weight
        64 +  # conv2.bias
        128 * 64 * 8 * 8 +  # fc1.weight
        128 +  # fc1.bias
        10 * 128 +  # fc2.weight
        10  # fc2.bias
    )

    # Export the model with separate input and weight arguments
    return export(model, (dummy_input, dummy_weights))


def main():
    # Training setup
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Train the standard model
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training base model...")
    for epoch in range(2):
        for images, labels in train_loader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

    # Export model with dynamic weight support
    exported_program = export_model_with_dynamic_weights()

    # Convert trained weights to flattened tensor
    flattened_weights = flatten_state_dict(model.state_dict())

    # Test with dynamic weights
    print("\nTesting exported model with trained weights...")
    test_input = torch.randn(1, 3, 32, 32)

    with torch.no_grad():
        output = exported_program.module(test_input, flattened_weights)
        print("Output shape:", output.shape)
        print("Output:", output)


if __name__ == "__main__":
    main()