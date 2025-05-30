import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ]
)

# Dataset and loader
test_dataset = datasets.ImageFolder("data/testing", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32)

# Load model
model = CNN().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# Evaluation loop
correct = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()

accuracy = correct / len(test_loader.dataset)
print(f"Test Accuracy: {accuracy:.4f}")
