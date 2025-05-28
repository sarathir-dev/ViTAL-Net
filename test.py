# Evaluation pipeline

import torch
from torchvision import transforms
from models.vitalnet import ViTALNet
from utils.dataset_loader import get_dataloaders
from utils.metrics import evaluate_model
from config import DEVICE, DATA_PATH, BATCH_SIZE

model = ViTALNet().to(DEVICE)
model.load_state_dict(torch.load("./models/vitalnet.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

_, test_loader = get_dataloaders(
    root_dir=DATA_PATH,
    batch_size=BATCH_SIZE,
    transform=transform,
    train=False
)

evaluate_model(model, test_loader, DEVICE)
