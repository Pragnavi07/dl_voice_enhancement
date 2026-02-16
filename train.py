import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# ================= CONFIG =================
SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 128
WIN_LENGTH = 512

EPOCHS = 40
BATCH_SIZE = 4
LR = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLEAN_DIR = "data/clean"
NOISY_DIR = "data/noisy"

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "unet_best.pt")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ================= UTILS =================
def wav_to_mag(wav):
    window = torch.hann_window(WIN_LENGTH).to(wav.device)
    spec = torch.stft(
        wav,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        return_complex=True
    )
    return torch.abs(spec)  # [freq, time]

# ================= DATASET =================
class SpeechDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.files = sorted(
            set(os.listdir(clean_dir)).intersection(os.listdir(noisy_dir))
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        clean, sr = torchaudio.load(os.path.join(self.clean_dir, fname))
        noisy, _ = torchaudio.load(os.path.join(self.noisy_dir, fname))

        clean = clean.mean(dim=0)
        noisy = noisy.mean(dim=0)

        if sr != SAMPLE_RATE:
            clean = torchaudio.functional.resample(clean, sr, SAMPLE_RATE)
            noisy = torchaudio.functional.resample(noisy, sr, SAMPLE_RATE)

        clean_mag = wav_to_mag(clean)
        noisy_mag = wav_to_mag(noisy)

        return noisy_mag, clean_mag

# ================= COLLATE =================
def collate_fn(batch):
    noisy, clean = zip(*batch)

    max_time = max(x.shape[1] for x in noisy)

    def pad(x):
        return F.pad(x, (0, max_time - x.shape[1]))

    noisy = torch.stack([pad(x) for x in noisy])
    clean = torch.stack([pad(x) for x in clean])

    return noisy.unsqueeze(1), clean.unsqueeze(1)

# ================= MODEL =================
class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()  # mask ∈ [0,1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ================= TRAIN =================
def train():
    dataset = SpeechDataset(CLEAN_DIR, NOISY_DIR)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    model = SimpleUNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_loss = float("inf")

    for epoch in range(EPOCHS):
        total_loss = 0.0

        for noisy_mag, clean_mag in loader:
            noisy_mag = noisy_mag.to(DEVICE)
            clean_mag = clean_mag.to(DEVICE)

            optimizer.zero_grad()

            mask = model(noisy_mag)
            enhanced = mask * noisy_mag

            loss = torch.mean(
                torch.abs(torch.log1p(enhanced) - torch.log1p(clean_mag))
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print("✔ Best model saved")

    print("✅ Training finished successfully")

# ================= RUN =================
if __name__ == "__main__":
    train()
