import os
import torch

SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 128
WIN_LENGTH = 512

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "unet_best.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
