"""
Transformer Encoder for EEG classification.

Takes CNN feature tokens [B, N, D] and produces classification logits.
Uses sinusoidal positional encoding and standard PyTorch TransformerEncoder.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence data."""

    def __init__(self, embed_dim, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, embed_dim]

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: [B, N, D] input sequence

        Returns:
            [B, N, D] with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class EEGTransformerEncoder(nn.Module):
    """
    Transformer encoder for EEG feature classification.

    Input:  [B, N, D] feature tokens from CNN
    Output: [B, num_classes] classification logits

    Uses a learnable CLS token prepended to the sequence.
    """

    def __init__(self, embed_dim=64, num_heads=4, num_layers=2,
                 ff_dim=128, dropout=0.1, num_classes=2, max_len=512):
        super().__init__()

        self.embed_dim = embed_dim

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Positional encoding
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_len + 1, dropout=dropout)

        # Layer normalization (pre-norm)
        self.input_norm = nn.LayerNorm(embed_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: [B, N, D] feature tokens from CNN

        Returns:
            [B, num_classes] classification logits
        """
        B = x.size(0)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat([cls, x], dim=1)           # [B, N+1, D]

        # Input normalization + positional encoding
        x = self.input_norm(x)
        x = self.pos_encoder(x)                   # [B, N+1, D]

        # Transformer encoding
        x = self.transformer(x)                    # [B, N+1, D]

        # CLS token output for classification
        cls_output = x[:, 0, :]                    # [B, D]

        # Classify
        logits = self.classifier(cls_output)       # [B, num_classes]
        return logits


if __name__ == '__main__':
    # Smoke test
    B, N, D = 4, 40, 64
    x = torch.randn(B, N, D)

    model = EEGTransformerEncoder(embed_dim=D, num_classes=5)
    logits = model(x)

    print(f"Input shape:  {x.shape}")       # [4, 40, 64]
    print(f"Output shape: {logits.shape}")   # [4, 5]

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters:   {params:,}")

    print("\n[OK] EEGTransformerEncoder smoke test passed!")
