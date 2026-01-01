import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model.mnist_model import MNISTCNN

# MNIST preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST dataset
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST("data", train=True, download=True, transform=transform),
    batch_size=64,
    shuffle=True
)

# Model
model = MNISTCNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Train ONLY 2 epochs
for epoch in range(2):
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

# Save pretrained weights
torch.save(model.state_dict(), "model/mnist_cnn.pth")
print("✅ Saved model/mnist_cnn.pth")
