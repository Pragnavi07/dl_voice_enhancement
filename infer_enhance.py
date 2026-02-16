import os
import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F

from config import SAMPLE_RATE, N_FFT, HOP_LENGTH, WIN_LENGTH, CHECKPOINT_PATH

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= MODEL (MUST MATCH TRAINING) =================
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
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# ================= ENHANCE =================
def enhance_audio(input_path):
    model = SimpleUNet().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    wav, sr = torchaudio.load(input_path)
    wav = wav.mean(dim=0)

    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)

    window = torch.hann_window(WIN_LENGTH).to(DEVICE)

    spec = torch.stft(
        wav.to(DEVICE),
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        return_complex=True
    )

    mag = torch.abs(spec)
    phase = torch.angle(spec)

    with torch.no_grad():
        mask = model(mag.unsqueeze(0).unsqueeze(0))
        enhanced_mag = mask.squeeze() * mag

    enhanced_spec = enhanced_mag * torch.exp(1j * phase)

    enhanced_wav = torch.istft(
        enhanced_spec,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window
    )

    out_path = os.path.join(OUTPUT_DIR, "enhanced.wav")
    torchaudio.save(out_path, enhanced_wav.unsqueeze(0).cpu(), SAMPLE_RATE)

    print(f"✅ Enhanced audio saved at: {out_path}")
