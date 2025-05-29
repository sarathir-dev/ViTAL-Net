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
    # Use IMAGE_SIZE for consistency
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
    # Initialize accuracy tracking for the epoch
    total_d_correct_real = 0
    total_d_correct_fake = 0
    total_samples = 0

    for i, (imgs, _) in enumerate(dataloader):
        real_imgs = imgs.to(DEVICE)
        batch_size = real_imgs.size(0)

        # === Train Discriminator ===
        noise = torch.randn(batch_size, Z_DIM).to(DEVICE)
        fake_imgs = generator(noise)

        discriminator.zero_grad()

        # Train with real images
        output_real = discriminator(real_imgs).view(-1)
        loss_real = criterion(output_real, torch.full(
            (batch_size,), real_label, device=DEVICE))

        # Calculate discriminator accuracy on real images
        # Round output to 0 or 1, then compare with real_label
        # Using 0.5 as threshold for binary classification
        predictions_real = (output_real > 0.5).float()
        correct_real = (predictions_real == real_label).sum().item()
        total_d_correct_real += correct_real

        # Train with fake images
        output_fake = discriminator(fake_imgs.detach()).view(-1)
        loss_fake = criterion(output_fake, torch.full(
            (batch_size,), fake_label, device=DEVICE))

        # Calculate discriminator accuracy on fake images
        # Round output to 0 or 1, then compare with fake_label
        # Fake images should ideally be classified as 0
        predictions_fake = (output_fake <= 0.5).float()
        correct_fake = (predictions_fake == fake_label).sum().item()
        total_d_correct_fake += correct_fake

        # Total discriminator loss and update
        loss_d = loss_real + loss_fake
        loss_d.backward()
        opt_d.step()

        # Update total samples for overall accuracy
        # Each batch contains real images, which is half the input to D
        total_samples += batch_size

        # === Train Generator ===
        generator.zero_grad()
        # Discriminator's output on generator's latest fake images
        output = discriminator(fake_imgs).view(-1)
        # Generator wants discriminator to classify fakes as real
        loss_g = criterion(output, torch.full(
            (batch_size,), real_label, device=DEVICE))
        loss_g.backward()
        opt_g.step()

    # Calculate and print epoch-level discriminator accuracy
    # Discriminator accuracy is (correctly classified real + correctly classified fake) / (total real + total fake)
    # Since each batch has 'batch_size' real and 'batch_size' fake images, total samples for D is 2 * total_samples_from_dataloader
    discriminator_accuracy = (
        (total_d_correct_real + total_d_correct_fake) / (2 * total_samples)) * 100

    print(
        f"Epoch [{epoch+1}/{EPOCHS}]  Loss D: {loss_d.item():.4f}, Loss G: {loss_g.item():.4f}, D Accuracy: {discriminator_accuracy:.2f}%")

# Save Generator
torch.save(generator.state_dict(), MODEL_SAVE_PATH)
print(f"Generator saved to {MODEL_SAVE_PATH}")
