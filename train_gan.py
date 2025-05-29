import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
from torchvision.datasets import ImageFolder
from models.gan_generator import Generator, Discriminator
from config import DEVICE, DATA_PATH

# Training Configs
BATCH_SIZE = 64
Z_DIM = 100
EPOCHS = 50
LR = 2e-4
IMAGE_SIZE = 64
MODEL_SAVE_PATH = "./models/generator.pth"


# Prepare Dataset
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = ImageFolder(root=DATA_PATH, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize Models
generator = Generator(z_dim=Z_DIM).to(DEVICE)
discriminator = Discriminator().to(DEVICE)

# Loss & Optimizers
criterion = nn.BCELoss()
opt_g = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
opt_d = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

# Labels
real_label = 1.
fake_label = 0.

print("Starting GAN Training...")

for epoch in range(EPOCHS):
    for i, (imgs, _) in enumerate(dataloader):
        real_imgs = imgs.to(DEVICE)
        batch_size = real_imgs.size(0)

        # === Train Discriminator ===
        noise = torch.randn(batch_size, Z_DIM).to(DEVICE)
        fake_imgs = generator(noise)

        discriminator.zero_grad()
        output_real = discriminator(real_imgs).view(-1)
        loss_real = criterion(output_real, torch.full(
            (batch_size,), real_label, device=DEVICE))

        output_fake = discriminator(fake_imgs.detach()).view(-1)
        loss_fake = criterion(output_fake, torch.full(
            (batch_size,), fake_label, device=DEVICE))

        loss_d = loss_real + loss_fake
        loss_d.backward()
        opt_d.step()

        # === Train Generator ===
        generator.zero_grad()
        output = discriminator(fake_imgs).view(-1)
        loss_g = criterion(output, torch.full(
            (batch_size,), real_label, device=DEVICE))
        loss_g.backward()
        opt_g.step()

    print(
        f"Epoch [{epoch+1}/{EPOCHS}]  Loss D: {loss_d.item():.4f}, Loss G: {loss_g.item():.4f}")

# Save Generator
torch.save(generator.state_dict(), MODEL_SAVE_PATH)
print(f"Generator saved to {MODEL_SAVE_PATH}")
