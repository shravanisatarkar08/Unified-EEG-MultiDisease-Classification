"""
Combined CNN + Transformer EEG Classifier.

Pipeline: EEG [B,C,T] -> CNN -> [B,N,D] -> Transformer -> [B, num_classes]
"""

import torch
import torch.nn as nn

from .eeg_cnn import EEGFeatureExtractor
from .transformer import EEGTransformerEncoder


class EEGClassifier(nn.Module):
    """
    End-to-end EEG classification model.

    Combines EEGNet-style CNN feature extractor with a Transformer encoder.
    """

    def __init__(self, n_channels=19, n_samples=1280, num_classes=2,
                 # CNN params
                 F1=16, D=2, F2=32, cnn_dropout=0.25, embed_dim=64,
                 # Transformer params
                 num_heads=4, num_layers=2, ff_dim=128, trans_dropout=0.1):
        super().__init__()

        self.cnn = EEGFeatureExtractor(
            n_channels=n_channels,
            n_samples=n_samples,
            n_classes=num_classes,
            F1=F1, D=D, F2=F2,
            dropout_rate=cnn_dropout,
            embed_dim=embed_dim
        )

        self.transformer = EEGTransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=trans_dropout,
            num_classes=num_classes
        )

    def forward(self, x):
        """
        Full pipeline: EEG -> CNN features -> Transformer -> logits.

        Args:
            x: [B, C, T] raw EEG window

        Returns:
            [B, num_classes] classification logits
        """
        features = self.cnn.extract_features(x)  # [B, N, D]
        logits = self.transformer(features)        # [B, num_classes]
        return logits

    def get_features(self, x):
        """Extract CNN features without classification.

        Returns:
            [B, N, D] feature tokens
        """
        return self.cnn.extract_features(x)

    def get_cnn_logits(self, x):
        """CNN-only classification (no Transformer).

        Returns:
            [B, num_classes] logits from CNN classifier head
        """
        return self.cnn(x)


if __name__ == '__main__':
    B, C, T = 4, 19, 1280
    x = torch.randn(B, C, T)

    model = EEGClassifier(n_channels=C, n_samples=T, num_classes=5)

    # Full pipeline
    logits = model(x)
    print(f"Input:           {x.shape}")
    print(f"Full logits:     {logits.shape}")

    # Feature extraction
    feats = model.get_features(x)
    print(f"CNN features:    {feats.shape}")

    # CNN-only
    cnn_logits = model.get_cnn_logits(x)
    print(f"CNN-only logits: {cnn_logits.shape}")

    total = sum(p.numel() for p in model.parameters())
    print(f"Total params:    {total:,}")

    print("\n[OK] EEGClassifier smoke test passed!")
