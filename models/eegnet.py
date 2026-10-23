"""
models/eegnet.py - Compact EEGNet Architecture (Lawhern et al., 2018).

Reference:
----------
Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Chou, C. A., & Lance, B. J. (2018).
EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces.
Journal of Neural Engineering, 15(5), 056013.

Architecture:
-------------
- Block 1:
    - 2D Temporal Convolution: Conv2d(1, F1, (1, kernel_length))
    - Batch Normalization
    - Depthwise Spatial Convolution: Conv2d(F1, F1*D, (C, 1), groups=F1) across all electrodes
    - Batch Normalization + ELU activation
    - Average Pooling (1, 4)
    - Dropout
- Block 2:
    - Separable Convolution:
        - Depthwise: Conv2d(F1*D, F1*D, (1, 16), groups=F1*D)
        - Pointwise: Conv2d(F1*D, F2, (1, 1))
    - Batch Normalization + ELU activation
    - Average Pooling (1, 8)
    - Dropout
- Classification Block:
    - Flatten: F2 * (T // 32)
    - Dense Linear Classifier: Linear(F2 * (T // 32), num_classes)

Total Parameters: ~4,613 (for F1=8, D=2, F2=16, C=19, T=1280, num_classes=5).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """
    Compact EEGNet architecture for raw EEG signal classification.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels (C=19).
    n_samples : int
        Number of time samples per window (T=1280 for 5s @ 256 Hz).
    num_classes : int
        Number of target classification categories (5 classes).
    F1 : int
        Number of temporal filters in Block 1 (default 8).
    D : int
        Depth multiplier for spatial depthwise convolution (default 2).
    F2 : int
        Number of pointwise filters in Block 2 (default 16 = F1 * D).
    kernel_length : int
        Temporal convolution kernel length (default 64, corresponding to 250 ms at 256 Hz).
    dropout_rate : float
        Dropout probability for regularization (default 0.25).
    """

    def __init__(
        self,
        n_channels: int = 19,
        n_samples: int = 1280,
        num_classes: int = 5,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        dropout_rate: float = 0.25,
    ) -> None:
        super().__init__()

        self.n_channels = n_channels
        self.n_samples = n_samples
        self.num_classes = num_classes
        self.F1 = F1
        self.D = D
        self.F2 = F2

        # ------------------------------------------------------------------
        # Block 1: Temporal Conv + Depthwise Spatial Conv
        # ------------------------------------------------------------------
        # Temporal convolution across time samples
        self.conv1 = nn.Conv2d(
            1, F1, (1, kernel_length),
            padding=(0, kernel_length // 2), bias=False
        )
        self.bn1 = nn.BatchNorm2d(F1)

        # Depthwise spatial convolution across electrodes
        self.depthwise = nn.Conv2d(
            F1, F1 * D, (n_channels, 1),
            groups=F1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.act1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout_rate)

        # ------------------------------------------------------------------
        # Block 2: Separable Convolution
        # ------------------------------------------------------------------
        # Depthwise temporal convolution
        self.sep_depthwise = nn.Conv2d(
            F1 * D, F1 * D, (1, 16),
            padding=(0, 8), groups=F1 * D, bias=False
        )
        # Pointwise 1x1 convolution
        self.sep_pointwise = nn.Conv2d(
            F1 * D, F2, (1, 1), bias=False
        )
        self.bn3 = nn.BatchNorm2d(F2)
        self.act2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout_rate)

        # ------------------------------------------------------------------
        # Classification Block
        # ------------------------------------------------------------------
        # Dimension calculation after two pooling layers (pool1: /4, pool2: /8 -> /32)
        self.n_time_tokens = n_samples // 32
        self.flatten_dim = F2 * self.n_time_tokens

        self.classifier = nn.Linear(self.flatten_dim, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract flattened latent features before linear classifier.

        Parameters
        ----------
        x : torch.Tensor
            Input EEG window of shape [B, C, T] or [B, 1, C, T].

        Returns
        -------
        torch.Tensor
            Flattened latent feature vector of shape [B, flatten_dim].
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)  # [B, 1, C, T]

        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2
        x = self.sep_depthwise(x)
        x = self.sep_pointwise(x)
        x = self.bn3(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        # Flatten spatial & temporal feature maps
        features = x.flatten(start_dim=1)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification.

        Parameters
        ----------
        x : torch.Tensor
            Input EEG window of shape [B, C, T].

        Returns
        -------
        torch.Tensor
            Classification logits of shape [B, num_classes].
        """
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits


if __name__ == "__main__":
    B, C, T = 4, 19, 1280
    dummy = torch.randn(B, C, T)

    model = EEGNet(n_channels=C, n_samples=T, num_classes=5)
    out = model(dummy)
    feats = model.extract_features(dummy)

    n_params = sum(p.numel() for p in model.parameters())
    print("=" * 60)
    print("EEGNet Model Architecture Test")
    print("=" * 60)
    print(f"Input shape:        {dummy.shape}")
    print(f"Features shape:     {feats.shape}")
    print(f"Logits shape:       {out.shape}")
    print(f"Total Parameters:   {n_params:,}")
    print("=" * 60)
    assert out.shape == (B, 5), f"Unexpected output shape: {out.shape}"
    assert feats.shape == (B, 16 * 40), f"Unexpected features shape: {feats.shape}"
    print("[OK] Compact EEGNet smoke test passed!")
