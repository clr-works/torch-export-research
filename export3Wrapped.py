import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.export import export

# Define the wrapped model
class WrappedSimpleCNN(nn.Module):
    def __init__(self):
        super(WrappedSimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # Adjusted for 3 input channels (CIFAR)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # Adjusted for CIFAR image size (32x32 with two pooling layers)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x, conv1_weight, conv1_bias, conv2_weight, conv2_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias):
        x = self.pool(self.relu(nn.functional.conv2d(x, conv1_weight, conv1_bias, stride=1, padding=1)))
        x = self.pool(self.relu(nn.functional.conv2d(x, conv2_weight, conv2_bias, stride=1, padding=1)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(nn.functional.linear(x, fc1_weight, fc1_bias))
        return nn.functional.linear(x, fc2_weight, fc2_bias)

# Function to test the exported model with dynamic weights
def test_exported_program_with_weights(exported_program, input_tensor, weights):
    weight_inputs = (
        weights['conv1.weight'], weights['conv1.bias'],
        weights['conv2.weight'], weights['conv2.bias'],
        weights['fc1.weight'], weights['fc1.bias'],
        weights['fc2.weight'], weights['fc2.bias']
    )

    with torch.no_grad():
        output = exported_program.module(input_tensor, *weight_inputs)
        print("Output with dynamically passed weights:", output)


if __name__ == "__main__":
    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Initialize and export the wrapped model
    wrapped_model = WrappedSimpleCNN()
    input_tensor = torch.randn(1, 3, 32, 32)  # Updated for 3 channels and CIFAR's 32x32 image size
    example_weights = (
        wrapped_model.conv1.weight, wrapped_model.conv1.bias,
        wrapped_model.conv2.weight, wrapped_model.conv2.bias,
        wrapped_model.fc1.weight, wrapped_model.fc1.bias,
        wrapped_model.fc2.weight, wrapped_model.fc2.bias
    )
    exported_program = export(wrapped_model, (input_tensor, *example_weights))

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Train with different optimizers and save the trained weights
    trained_weights = {}
    optimizers = {
        "SGD": optim.SGD(wrapped_model.parameters(), lr=0.01, momentum=0.9),
        "Adam": optim.Adam(wrapped_model.parameters(), lr=0.001),
    }

    for opt_name, optimizer in optimizers.items():
        # Reset parameters before each training session
        wrapped_model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
        for epoch in range(2):
            for images, labels in train_loader:
                optimizer.zero_grad()
                output = wrapped_model(images, *example_weights)  # Pass example weights for training
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
        trained_weights[opt_name] = wrapped_model.state_dict().copy()

    # Test the exported model with each optimizer's weights
    for opt_name, weights in trained_weights.items():
        print(f"\nTesting with weights from optimizer: {opt_name}")
        test_exported_program_with_weights(exported_program, input_tensor, weights)




