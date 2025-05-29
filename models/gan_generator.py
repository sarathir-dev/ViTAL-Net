# GAN module (cGAN or TGAN)

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim=100, img_channels=3, feature_maps=64):
        super().__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(z_dim, feature_maps * 8, 4, 1,
                               0, bias=False),   # 1x1 -> 4x4
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4,
                               4, 2, 1, bias=False),  # 4x4 -> 8x8
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2,
                               4, 2, 1, bias=False),  # 8x8 -> 16x16
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps * 2, feature_maps,
                               4, 2, 1, bias=False),  # 16x16 -> 32x32
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps, img_channels, 4,
                               2, 1, bias=False),  # 32x32 -> 64x64
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x.view(x.size(0), x.size(1), 1, 1))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),      # 64x64 -> 32x32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),    # 32x32 -> 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),   # 16x16 -> 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),   # 8x8 -> 4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4),           # 4x4 -> 1x1
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)


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
