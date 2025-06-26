import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import gradio as gr

# 👇 This model MUST match the training script's model exactly
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # → 75x75
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # → 37x37
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # → 18x18
            nn.Flatten(),
            nn.Linear(64 * 18 * 18, 128), nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        return self.net(x)


# Initialize model
cnn_model = CNNModel()
cnn_model.load_state_dict(torch.load("models/cnn_brain_tumor.pth", map_location="cpu"))
cnn_model.eval()

# Labels (make sure this matches ImageFolder's class_to_idx order)
image_labels = ["glioma", "meningioma", "no_tumor", "pituitary"]

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
])

# Prediction function
def predict_tumor(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = cnn_model(image)
        predicted = output.argmax(1).item()
    return f"🧠 Tumor Type: {image_labels[predicted]}"

# Gradio UI
gr.Interface(
    fn=predict_tumor,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Brain Tumor MRI Classifier"
).launch()
