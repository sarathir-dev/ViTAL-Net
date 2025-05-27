# Model architecture (CNN + GRU + Attention)
# CNN + GRU + Self-Attention for Temporal Violence Detection

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialCNN(nn.Module):
    def __init__(self):
        super(SpatialCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # [B, 32, H, W]
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),  # [B, 64, H/2, W/2]
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),  # [B, 128, H/4, W/4]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # Reduce to [128, 4, 4]
        )

    def forward(self, x):
        return self.features(x).view(x.size(0), -1)  # [B, 128*4*4]


class TemporalGRUAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TemporalGRUAttention, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        gru_out, _ = self.gru(x)  # [B, T, H]
        attn_weights = F.softmax(self.attn(gru_out), dim=1)  # [B, T, 1]
        context = torch.sum(attn_weights * gru_out, dim=1)  # [B, H]
        return context


class ViTALNet(nn.Module):
    def __init__(self, cnn_feat_size=2048, gru_hidden_size=256, num_classes=2):
        super(ViTALNet, self).__init__()
        self.cnn = SpatialCNN()
        self.temporal_module = TemporalGRUAttention(
            input_dim=2048, hidden_dim=gru_hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):  # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        cnn_out = self.cnn(x)  # [B*T, 2048]
        cnn_out = cnn_out.view(B, T, -1)  # [B, T, 2048]
        context = self.temporal_module(cnn_out)  # [B, 256]
        out = self.fc(context)  # [B, num_classes]
        return out
