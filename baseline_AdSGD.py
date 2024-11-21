import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.export import export

#Define a simple CNN model for CIFAR-10
class CIFAR10Model(nn.Module):
    def __init__(self):
        super(CIFAR10Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

#defune a data loader for cifar10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

#Train (using different optimizers) and save the trained weights
def train_model(model, optimizer):
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(1):  # 1 epoch, it is just a demo
        for images, labels in train_loader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
    return model.state_dict()

#Init model and train with SGD
model = CIFAR10Model()
optimizer_sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
sgd_weights = train_model(model, optimizer_sgd)

#Reinitialize model and train with Adam
model = CIFAR10Model()
optimizer_adam = optim.Adam(model.parameters(), lr=0.001)
adam_weights = train_model(model, optimizer_adam)

# Function to inject weights, export, and run inference
def export_and_infer_with_weights(weights):
    # Inject weights into model's state_dict
    model.load_state_dict(weights)

    # Sample input for exporting the model
    sample_input = torch.randn(1, 3, 32, 32)

    # Export the model with the injected weights
    exported_program = export(model, (sample_input,))

    # New input for inference
    new_input = torch.randn(1, 3, 32, 32)

    # Perform inference with the exported model using `module()`, get raw logits
    output = exported_program.module()(new_input)
    print("Inference output with injected weights:", output)

# Run inference with both sets of weights
print("Testing with SGD-trained weights:")
export_and_infer_with_weights(sgd_weights)

print("\nTesting with Adam-trained weights:")
export_and_infer_with_weights(adam_weights)
