import numpy as np
import soundfile as sf
import os

os.makedirs("inputs", exist_ok=True)

sr = 16000
duration = 3  # seconds

# Generate fake speech-like signal
t = np.linspace(0, duration, int(sr * duration))
clean = 0.5 * np.sin(2 * np.pi * 220 * t)

# Add noise
noise = np.random.normal(0, 0.05, clean.shape)
noisy = clean + noise

sf.write("inputs/noisy.wav", noisy, sr)
print("inputs/noisy.wav created")
