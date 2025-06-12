# Dataset & DataLoader creation
# This handles loading the frame-extrated dataset and preparing it as PyTorch Dataset DataLoader.

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random


class ViolenceVideoDataset(Dataset):
    def __init__(self, data_path, seq_len=16, image_size=(128, 128), mode='train'):
        self.data_path = data_path
        self.seq_len = seq_len
        self.image_size = image_size
        self.mode = mode
        self.samples = []

        for label_name in ['fight', 'nonfight']:
            label_dir = os.path.join(data_path, label_name)
            label = 1 if label_name == 'fight' else 0

            for video_folder in os.listdir(label_dir):
                video_path = os.path.join(label_dir, video_folder)
                frames = sorted([f for f in os.listdir(
                    video_path) if f.endswith('.jpg')])
                if len(frames) >= seq_len:
                    self.samples.append((video_path, frames, label))

        random.shuffle(self.samples)

        # Split train/test
        split_idx = int(len(self.samples) * 0.8)
        if mode == 'train':
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, frames, label = self.samples[idx]
        start = random.randint(0, len(frames) - self.seq_len)
        selected_frames = frames[start:start+self.seq_len]

        images = []
        for frame_file in selected_frames:
            img_path = os.path.join(video_path, frame_file)
            img = Image.open(img_path).convert('RGB')
            images.append(self.transform(img))

        clip = torch.stack(images)  # shape: (T, C, H, W)
        return clip, label


def get_dataloaders(data_path, batch_size, seq_len, image_size):
    train_set = ViolenceVideoDataset(
        data_path, seq_len, image_size, mode='train')
    val_set = ViolenceVideoDataset(data_path, seq_len, image_size, mode='val')

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=2)
    return train_loader, val_loader
