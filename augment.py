# Run GAN to generate synthetic data

import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from models.gan_generator import Generator
from config import DEVICE, DATA_PATH, BATCH_SIZE

GAN_OUTPUT_DIR = "./gan_output"


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for label_name in ['fight', 'nonfight']:
            label_dir = os.path.join(root_dir, label_name)
            label = 1 if label_name == 'fight' else 0

            # Recursively walk through all subdirectories
            for root, _, files in os.walk(label_dir):
                for file in files:
                    if file.endswith('.jpg') or file.endswith('.png'):
                        self.samples.append((os.path.join(root, file), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

dataset = CustomImageDataset(root_dir=DATA_PATH, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

generator = Generator().to(DEVICE)
generator.load_state_dict(torch.load("./models/generator.pth"))
generator.eval()

os.makedirs(GAN_OUTPUT_DIR, exist_ok=True)
synthetic_count = 0

with torch.no_grad():
    for i, (images, labels) in enumerate(dataloader):
        noise = torch.randn(images.size(0), 100).to(DEVICE)
        fake_images = generator(noise)

        for j in range(fake_images.size(0)):
            label = labels[j].item()
            label_dir = os.path.join(
                GAN_OUTPUT_DIR, "fight" if label == 1 else "nonfight")
            os.makedirs(label_dir, exist_ok=True)
            save_path = os.path.join(
                label_dir, f"synthetic_{synthetic_count}.png")
            transforms.ToPILImage()(fake_images[j].cpu()).save(save_path)
            synthetic_count += 1

print(
    f"Synthetic augmentation completed. Total generated images: {synthetic_count}")
