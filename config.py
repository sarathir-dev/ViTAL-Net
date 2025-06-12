# Configurations & hyperparameters

import os
import torch

DATA_PATH = "/home/sarathi/Documents/Projects/violence-detection/data/violence_dataset/hockey_fight"
BATCH_SIZE = 16
NUM_EPOCHS = 30
LR = 1e-4
IMG_SIZE = (128, 128)
SEQ_LEN = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
