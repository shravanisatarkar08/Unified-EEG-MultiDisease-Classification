"""
models/spectral_features.py - Differentiable Log Relative Band Power Feature Extraction.

Extracts standard clinical EEG band powers (Delta, Theta, Alpha, Beta, Gamma)
from multi-channel raw EEG windows [B, C, T] using discrete Fourier transform.

Bands (assuming fs = 256 Hz):
- Delta: 0.5 - 4.0 Hz
- Theta: 4.0 - 8.0 Hz
- Alpha: 8.0 - 13.0 Hz
- Beta:  13.0 - 30.0 Hz
- Gamma: 30.0 - 45.0 Hz

Normalisation:
- Relative power = Band Power / Total Power (cancels arbitrary amplitude / impedance differences across sites)
- Log relative power = log(Relative Power + epsilon)
"""

import math
import torch
import torch.nn as nn


class LogRelativeBandPowerExtractor(nn.Module):
    """
    Computes log relative band power for 19-channel EEG signals.
    Input:  [B, 19, 1280] float32 tensor (5s at 256 Hz)
    Output: [B, 95] float32 tensor (19 channels x 5 frequency bands)
    """

    def __init__(self, srate: float = 256.0, n_samples: int = 1280, eps: float = 1e-6):
        super().__init__()
        self.srate = srate
        self.n_samples = n_samples
        self.eps = eps

        # Frequency bin resolution: df = srate / n_samples = 256 / 1280 = 0.2 Hz
        df = srate / n_samples
        self.bands = [
            (max(1, int(round(0.5 / df))), int(round(4.0 / df))),   # Delta: 0.5 - 4 Hz
            (int(round(4.0 / df)), int(round(8.0 / df))),           # Theta: 4 - 8 Hz
            (int(round(8.0 / df)), int(round(13.0 / df))),          # Alpha: 8 - 13 Hz
            (int(round(13.0 / df)), int(round(30.0 / df))),         # Beta: 13 - 30 Hz
            (int(round(30.0 / df)), int(round(45.0 / df))),         # Gamma: 30 - 45 Hz
        ]
        self.n_bands = len(self.bands)
        self.out_dim = 19 * self.n_bands  # 95

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T] where C=19, T=1280
        Returns:
            [B, C * 5] log relative band powers
        """
        B, C, T = x.shape
        # Real FFT along temporal dimension
        fft_vals = torch.fft.rfft(x, dim=-1)  # [B, C, T // 2 + 1]
        psd = (fft_vals.abs() ** 2) / float(T)

        band_powers = []
        for low_bin, high_bin in self.bands:
            # Sum power within frequency band
            p_b = psd[:, :, low_bin:high_bin].sum(dim=-1)  # [B, C]
            band_powers.append(p_b)

        powers = torch.stack(band_powers, dim=-1)  # [B, C, 5]
        total_p = powers.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # [B, C, 1]
        rel_powers = powers / total_p  # [B, C, 5]
        log_rel_powers = torch.log(rel_powers.clamp(min=self.eps))  # [B, C, 5]

        return log_rel_powers.view(B, -1)  # [B, 95]
