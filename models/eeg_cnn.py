"""
EEGNet-style CNN Feature Extractor for EEG signals.

Architecture:
    Input [B, C, T] -> Conv2d temporal -> DepthwiseConv2d spatial ->
    SeparableConv2d -> Feature projection -> [B, N, D]

Supports both feature extraction (for Transformer) and direct classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv2d(nn.Module):
    """Depthwise separable convolution."""
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class EEGFeatureExtractor(nn.Module):
    """
    EEGNet-inspired CNN that extracts features from EEG windows.

    Input:  [B, C, T] where C=n_channels, T=n_samples
    Output: extract_features() -> [B, N, D] feature tokens for Transformer
            forward()          -> [B, num_classes] classification logits
    """

    def __init__(self, n_channels=19, n_samples=1280, n_classes=2,
                 F1=16, D=2, F2=32, dropout_rate=0.25, embed_dim=64):
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.embed_dim = embed_dim

        # Block 1: Temporal convolution + Depthwise spatial convolution
        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        # Depthwise conv across channels
        self.depthwise = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.act1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout_rate)

        # Block 2: Separable convolution
        self.separable = SeparableConv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.act2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout_rate)

        # Calculate feature dimensions after conv blocks
        # After block1: T -> T//4 (pool1)
        # After block2: T//4 -> T//32 (pool2)
        # Spatial dim: n_channels -> 1 after depthwise, stays 1
        self._n_temporal = n_samples // 32  # temporal tokens
        self._n_features = F2  # features per token

        # Feature projection: [B, F2, 1, T//32] -> [B, T//32, embed_dim]
        self.feature_proj = nn.Linear(F2, embed_dim)

        # Classification head (from features)
        self.classifier = nn.Linear(embed_dim, n_classes)

    def _cnn_forward(self, x):
        """Run CNN blocks, return feature maps.

        Args:
            x: [B, C, T] raw EEG

        Returns:
            [B, F2, 1, T//32] feature maps
        """
        # Reshape: [B, C, T] -> [B, 1, C, T]
        x = x.unsqueeze(1)  # [B, 1, C, T]

        # Block 1: temporal + spatial
        x = self.conv1(x)      # [B, F1, C, T]
        x = self.bn1(x)
        x = self.depthwise(x)  # [B, F1*D, 1, T]
        x = self.bn2(x)
        x = self.act1(x)
        x = self.pool1(x)      # [B, F1*D, 1, T//4]
        x = self.drop1(x)

        # Block 2: separable
        x = self.separable(x)  # [B, F2, 1, T//4]
        x = self.bn3(x)
        x = self.act2(x)
        x = self.pool2(x)      # [B, F2, 1, T//32]
        x = self.drop2(x)

        return x

    def extract_features(self, x):
        """Extract feature tokens for Transformer encoder.

        Args:
            x: [B, C, T] raw EEG input

        Returns:
            [B, N, D] where N=temporal tokens, D=embed_dim
        """
        feat = self._cnn_forward(x)      # [B, F2, 1, T//32]
        B = feat.size(0)

        # Reshape: [B, F2, 1, N] -> [B, N, F2]
        feat = feat.squeeze(2)            # [B, F2, N]
        feat = feat.permute(0, 2, 1)      # [B, N, F2]

        # Project to embedding dim
        feat = self.feature_proj(feat)     # [B, N, embed_dim]
        return feat

    def forward(self, x):
        """Full forward pass with classification.

        Args:
            x: [B, C, T] raw EEG input

        Returns:
            [B, num_classes] classification logits
        """
        features = self.extract_features(x)  # [B, N, D]

        # Global average pooling over tokens
        pooled = features.mean(dim=1)        # [B, D]

        logits = self.classifier(pooled)     # [B, num_classes]
        return logits


if __name__ == '__main__':
    # Smoke test with dummy data
    B, C, T = 4, 19, 1280  # batch=4, channels=19, samples=1280 (5s @ 256Hz)
    x = torch.randn(B, C, T)

    model = EEGFeatureExtractor(n_channels=C, n_samples=T, n_classes=5)

    # Test feature extraction
    features = model.extract_features(x)
    print(f"Input shape:   {x.shape}")          # [4, 19, 1280]
    print(f"Feature shape: {features.shape}")    # [4, N, 64]

    # Test classification
    logits = model(x)
    print(f"Logits shape:  {logits.shape}")      # [4, 5]

    # Parameter count
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters:    {params:,}")

    print("\n[OK] EEGFeatureExtractor smoke test passed!")
