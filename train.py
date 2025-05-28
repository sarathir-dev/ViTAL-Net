# Training pipeline

import torch
import torch.nn as nn
from torchvision import transforms
from models.vitalnet import ViTALNet
from utils.dataset_loader import get_dataloaders
from utils.metrics import evaluate_model
from config import DATA_PATH, NUM_EPOCHS, LR, BATCH_SIZE, DEVICE

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_loader, val_loader = get_dataloaders(
    root_dir=DATA_PATH,
    batch_size=BATCH_SIZE,
    transform=transform
)

model = ViTALNet().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    acc = correct / total * 100
    print(
        f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%")

torch.save(model.state_dict(), "./models/vitalnet.pth")
print("Model training complete.")

evaluate_model(model, val_loader, DEVICE)
