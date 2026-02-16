import torch
from config import Config

def stft_waveform(waveform):
    return torch.stft(
        waveform,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH,
        win_length=Config.WIN_LENGTH,
        return_complex=True
    )

def istft_spectrogram(spec):
    return torch.istft(
        spec,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH,
        win_length=Config.WIN_LENGTH
    )

def mag_phase(spec):
    return torch.abs(spec), torch.angle(spec)

def mag_phase_to_complex(mag, phase):
    return torch.polar(mag, phase)
