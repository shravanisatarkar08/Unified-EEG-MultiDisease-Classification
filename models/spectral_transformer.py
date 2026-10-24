"""
models/spectral_transformer.py - Spectral Hybrid CNN-Transformer with DANN.

Architecture:
- Raw waveform [B, 19, 1280] -> EEGNet CNN feature extractor -> [B, 40, 64]
- Raw waveform [B, 19, 1280] -> Log Relative Band Power (Delta, Theta, Alpha, Beta, Gamma) -> [B, 95] -> Linear projection -> [B, 1, 64]
- Transformer sequence: [CLS_token (1), SPECTRAL_token (1), CNN_tokens (40)] -> [B, 42, 64]
- Transformer self-attention cross-attends spectral biomarkers with spatio-temporal conv features
- CLS output [B, 64] -> Disease Classifier -> [B, 5]
- Training-only: GRL(CLS output, alpha) -> Domain Classifier -> [B, 5 domains]
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn

from .eeg_classifier import EEGClassifier
from .spectral_features import LogRelativeBandPowerExtractor


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        output = grad_output.neg() * ctx.alpha
        return output, None


def grad_reverse(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return GradReverse.apply(x, alpha)


class SpectralHybridCNNTransformer(nn.Module):
    """
    Hybrid model combining time-domain CNN features, frequency-domain log relative
    band power features, and Transformer self-attention with optional DANN head.
    """

    def __init__(
        self,
        base_model: EEGClassifier,
        num_domains: int = 5,
        spectral_hidden_dim: int = 64,
        domain_hidden_dim: int = 32,
        domain_dropout: float = 0.20,
    ):
        super().__init__()
        self.base_model = base_model
        embed_dim = base_model.transformer.embed_dim  # 64

        # Spectral feature extractor (95 features = 19 channels x 5 bands)
        self.spectral_extractor = LogRelativeBandPowerExtractor(srate=256.0, n_samples=1280)
        self.spectral_proj = nn.Sequential(
            nn.Linear(self.spectral_extractor.out_dim, spectral_hidden_dim),
            nn.LayerNorm(spectral_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(spectral_hidden_dim, embed_dim),
        )

        # Domain adversarial classifier head
        self.domain_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, domain_hidden_dim),
            nn.GELU(),
            nn.Dropout(domain_dropout),
            nn.Linear(domain_hidden_dim, num_domains),
        )

    def extract_cls_embedding(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract CLS embedding and spectral features.
        Returns:
            cls_embedding: [B, 64]
            spectral_feat: [B, 95]
        """
        B = x.size(0)

        # 1. Temporal-spatial CNN features: [B, 40, 64]
        cnn_features = self.base_model.cnn.extract_features(x)

        # 2. Spectral relative band powers: [B, 95] -> [B, 1, 64]
        spectral_raw = self.spectral_extractor(x)
        spectral_token = self.spectral_proj(spectral_raw).unsqueeze(1)  # [B, 1, 64]

        # 3. Transformer sequence: [CLS (1), SPECTRAL (1), CNN (40)] -> [B, 42, 64]
        cls_token = self.base_model.transformer.cls_token.expand(B, -1, -1)  # [B, 1, 64]
        tokens = torch.cat([cls_token, spectral_token, cnn_features], dim=1)  # [B, 42, 64]

        # 4. Normalization + Positional encoding + Self-attention
        tokens = self.base_model.transformer.input_norm(tokens)
        tokens = self.base_model.transformer.pos_encoder(tokens)
        tokens = self.base_model.transformer.transformer(tokens)

        # 5. Extract output CLS token: [B, 64]
        cls_embedding = tokens[:, 0, :]
        return cls_embedding, spectral_raw

    def forward(
        self,
        x: torch.Tensor,
        alpha: float = 0.0,
        return_domain: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Standard forward pass.
        Returns disease_logits [B, 5] (and domain_logits [B, num_domains] if return_domain=True).
        """
        cls_embedding, _ = self.extract_cls_embedding(x)
        disease_logits = self.base_model.transformer.classifier(cls_embedding)

        if return_domain:
            rev_cls = grad_reverse(cls_embedding, alpha)
            domain_logits = self.domain_head(rev_cls)
            return disease_logits, domain_logits

        return disease_logits, None
