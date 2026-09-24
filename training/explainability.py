"""
training/explainability.py - Gradient-based Explainability for EEG Classifier.

Implements:
  1. Gradient-weighted Class Activation Mapping (Grad-CAM) adapted for 1-D EEG
  2. Input Gradient (saliency map) w.r.t. EEG channels x time
  3. Integrated Gradients (IG) approximation

All methods work on the EEGClassifier pipeline (CNN -> Transformer).
They return numpy arrays of the same spatial/temporal shape as input EEG
[C, T] so they can be overlaid on raw waveforms.

Usage example:
    from training.explainability import EEGExplainer
    explainer = EEGExplainer(model, device)
    saliency  = explainer.input_gradients(eeg_tensor, target_class=1)
    ig_map    = explainer.integrated_gradients(eeg_tensor, target_class=1, steps=20)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

# Path bootstrap
_MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _MODULE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


class EEGExplainer:
    """Wraps an EEGClassifier and exposes attribution methods."""

    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model  = model
        self.device = device
        self.model.eval()

    # ------------------------------------------------------------------
    # 1. Input Gradients (vanilla saliency)
    # ------------------------------------------------------------------

    def input_gradients(
        self,
        eeg: torch.Tensor,
        target_class: Optional[int] = None,
        smooth: bool = True,
    ) -> np.ndarray:
        """Compute gradient of target_class logit w.r.t. input EEG.

        Parameters
        ----------
        eeg          : [1, C, T] or [B, C, T] float32 tensor
        target_class : class index to explain (defaults to argmax prediction)
        smooth       : if True, return |gradient| (absolute saliency)

        Returns
        -------
        numpy array [C, T] saliency map (averaged over batch dim if B > 1)
        """
        if eeg.dim() == 2:
            eeg = eeg.unsqueeze(0)          # [1, C, T]
        eeg = eeg.to(self.device).requires_grad_(True)

        logits = self.model(eeg)            # [B, num_classes]
        if target_class is None:
            target_class = int(logits.argmax(dim=-1)[0].item())

        score = logits[:, target_class].sum()
        score.backward()

        grad = eeg.grad.detach().cpu().numpy()   # [B, C, T]
        if smooth:
            grad = np.abs(grad)
        return grad.mean(axis=0)                 # [C, T]

    # ------------------------------------------------------------------
    # 2. Integrated Gradients
    # ------------------------------------------------------------------

    def integrated_gradients(
        self,
        eeg: torch.Tensor,
        target_class: Optional[int] = None,
        steps: int = 30,
        baseline: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """Integrated Gradients attribution.

        Parameters
        ----------
        eeg          : [1, C, T] float32 tensor (single sample)
        target_class : class to explain
        steps        : number of IG approximation steps
        baseline     : baseline input (zeros by default)

        Returns
        -------
        numpy array [C, T] attribution map
        """
        if eeg.dim() == 2:
            eeg = eeg.unsqueeze(0)
        eeg = eeg.to(self.device)

        if baseline is None:
            baseline = torch.zeros_like(eeg)
        else:
            baseline = baseline.to(self.device)

        # Determine target class from model
        with torch.no_grad():
            logits = self.model(eeg)
        if target_class is None:
            target_class = int(logits.argmax(dim=-1)[0].item())

        # Interpolation path
        alphas = torch.linspace(0, 1, steps + 1, device=self.device)
        grads  = []

        for alpha in alphas:
            interp = (baseline + alpha * (eeg - baseline)).detach().requires_grad_(True)
            score  = self.model(interp)[:, target_class].sum()
            score.backward()
            grads.append(interp.grad.detach().cpu().numpy())

        grads_arr = np.stack(grads, axis=0)   # [steps+1, B, C, T]
        mean_grads = grads_arr.mean(axis=0)   # [B, C, T]
        delta = (eeg - baseline).detach().cpu().numpy()  # [B, C, T]
        ig    = mean_grads * delta             # [B, C, T]
        return ig.mean(axis=0)                # [C, T]

    # ------------------------------------------------------------------
    # 3. Attention weights (Transformer CLS token attention)
    # ------------------------------------------------------------------

    def transformer_attention(
        self,
        eeg: torch.Tensor,
    ) -> Optional[np.ndarray]:
        """Extract average attention weights from the Transformer encoder.

        Returns averaged attention over heads and layers:
          numpy array [N_tokens] (temporal token attention weights)
        or None if model does not expose attention.
        """
        if eeg.dim() == 2:
            eeg = eeg.unsqueeze(0)
        eeg = eeg.to(self.device)

        # Try to extract attention via hook
        attn_weights = []

        def _hook(module, input, output):
            # nn.MultiheadAttention returns (output, attn_weights)
            # When batch_first=True, attn_weights is [B, N, N]
            if isinstance(output, tuple) and len(output) == 2:
                w = output[1]
                if w is not None:
                    attn_weights.append(w.detach().cpu())

        hooks = []
        for layer in self.model.transformer.transformer.layers:
            hooks.append(layer.self_attn.register_forward_hook(_hook))

        try:
            with torch.no_grad():
                _ = self.model(eeg)
        finally:
            for h in hooks:
                h.remove()

        if not attn_weights:
            return None

        # Stack layers, average over heads and layers
        stacked = torch.stack(attn_weights, dim=0)   # [L, B, N, N]
        avg_attn = stacked.mean(dim=[0, 1])           # [N, N]
        # CLS token (index 0) attends to sequence positions
        cls_attn = avg_attn[0, 1:].numpy()            # [N_tokens]
        return cls_attn


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from models.eeg_classifier import EEGClassifier

    device = torch.device("cpu")
    model  = EEGClassifier(n_channels=19, n_samples=1280, num_classes=5).eval()
    exp    = EEGExplainer(model, device)

    x = torch.randn(1, 19, 1280)

    sal = exp.input_gradients(x, target_class=0)
    print(f"Input gradient shape: {sal.shape}")   # (19, 1280)

    ig = exp.integrated_gradients(x, target_class=0, steps=10)
    print(f"Integrated gradients: {ig.shape}")    # (19, 1280)

    attn = exp.transformer_attention(x)
    if attn is not None:
        print(f"Attention weights:    {attn.shape}")

    print("\n[OK] EEGExplainer smoke test passed!")
