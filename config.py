# Configurations & hyperparameters

import os
import torch

DATA_PATH = "datasets/hockey_fight"
BATCH_SIZE = 8
NUM_EPOCHS = 30
LR = 1e-4
IMG_SIZE = (128, 129)
SEQ_LEN = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
