import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from train_image_classifier import CNNModel  # reuse your trained model class
import os

# Paths
MODEL_PATH = 'models/cnn_brain_tumor.pth'
DATA_DIR = 'data/mri_images'

# Transform must match training
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load validation data
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
class_names = dataset.classes

# Recreate the validation split (same seed)
torch.manual_seed(42)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
_, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
val_loader = DataLoader(val_ds, batch_size=1, shuffle=True)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNModel(num_classes=4)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Visualize 5 predictions
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))  # CHW -> HWC
    inp = (inp * 0.5) + 0.5  # unnormalize
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.axis('off')

count = 0
with torch.no_grad():
    for images, labels in val_loader:
        if count == 5:
            break
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        label_name = class_names[labels.item()]
        pred_name = class_names[preds.item()]
        title = f"True: {label_name} | Pred: {pred_name}"
        
        plt.figure(figsize=(3, 3))
        imshow(images.cpu().squeeze(0), title)
        plt.show()

        count += 1
