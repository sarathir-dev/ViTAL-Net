# GAN module (cGAN or TGAN)

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, noise_dim=100, class_dim=2, img_size=128):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(noise_dim + class_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3 * img_size * img_size),
            nn.Tanh()
        )
        self.img_size = img_size

    def forward(self, noise, labels):
        x = torch.cat((noise, labels), dim=1)
        x = self.fc(x)
        return x.view(-1, 3, self.img_size, self.img_size)


class Discriminator(nn.Module):
    def __init__(self, class_dim=2, img_size=128):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3 * img_size * img_size + class_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.img_size = img_size

    def forward(self, img, labels):
        x = torch.cat((img.view(img.size(0), -1), labels), dim=1)
        return self.fc(x)


class GANTrainer:
    def __init__(self, img_size=128, device='cuda'):
        self.G = Generator(img_size=img_size).to(device)
        self.D = Discriminator(img_size=img_size).to(device)
        self.loss_fn = nn.BCELoss()
        self.device = device
        self.optim_G = torch.optim.Adam(self.G.parameters(), lr=0.0002)
        self.optim_D = torch.optim.Adam(self.D.parameters(), lr=0.0002)

    def train_step(self, real_imgs, labels):
        batch_size = real_imgs.size(0)
        valid = torch.ones(batch_size, 1).to(self.device)
        fake = torch.zeros(batch_size, 1).to(self.device)

        # ---------------------
        #  Train Generator
        # ---------------------
        noise = torch.randn(batch_size, 100).to(self.device)
        gen_labels = labels
        gen_imgs = self.G(noise, gen_labels)

        g_loss = self.loss_fn(self.D(gen_imgs, gen_labels), valid)

        self.optim_G.zero_grad()
        g_loss.backward()
        self.optim_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        real_loss = self.loss_fn(self.D(real_imgs, labels), valid)
        fake_loss = self.loss_fn(self.D(gen_imgs.detach(), gen_labels), fake)
        d_loss = (real_loss + fake_loss) / 2

        self.optim_D.zero_grad()
        d_loss.backward()
        self.optim_D.step()

        return d_loss.item(), g_loss.item()
