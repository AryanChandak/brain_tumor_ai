import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to dataset
DATA_DIR = "data/mri_images/"

# Image transformations (with augmentation)
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

# Load full dataset (no validation split)
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)

# Get class balance info
targets = [label for _, label in dataset]
class_counts = np.bincount(targets)
class_weights = 1. / class_counts
sample_weights = [class_weights[label] for label in targets]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# DataLoader with class balance
train_loader = DataLoader(dataset, batch_size=32, sampler=sampler)

# Define simple CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 150→75
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 75→37
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 37→18
            nn.Flatten(),
            nn.Linear(64 * 18 * 18, 128), nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        return self.net(x)

# Initialize model, loss, optimizer
model = CNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
print("🔁 Training on full dataset...")
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss, correct = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    accuracy = correct / len(dataset)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")

# Save model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/cnn_brain_tumor.pth")
print("✅ Model saved to: models/cnn_brain_tumor.pth")
