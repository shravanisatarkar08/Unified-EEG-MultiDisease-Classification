"""
tests/test_publication_system.py - Unit and regression tests for Phase 2 & Phase 3.
"""

import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.eeg_classifier import EEGClassifier
from models.spectral_features import LogRelativeBandPowerExtractor
from models.spectral_transformer import SpectralHybridCNNTransformer
from scripts.run_exp7_publication_system import subject_level_bootstrap


class TestPublicationSystem(unittest.TestCase):
    """Verify spectral features, hybrid model, bootstrap, and baseline preservation."""

    def test_log_relative_band_power_extractor(self):
        """Extractor must produce [B, 95] without NaNs or Infs."""
        extractor = LogRelativeBandPowerExtractor(srate=256.0, n_samples=1280)
        x = torch.randn(4, 19, 1280)
        features = extractor(x)

        self.assertEqual(features.shape, (4, 95))
        self.assertFalse(torch.isnan(features).any(), "Spectral features contain NaNs")
        self.assertFalse(torch.isinf(features).any(), "Spectral features contain Infs")

    def test_spectral_hybrid_model_shapes(self):
        """Hybrid model must preserve [B, 5] disease classification interface."""
        base = EEGClassifier()
        model = SpectralHybridCNNTransformer(base, num_domains=5)

        x = torch.randn(2, 19, 1280)

        # Inference mode
        model.eval()
        d_out, dom_out = model(x, return_domain=False)
        self.assertEqual(d_out.shape, (2, 5))
        self.assertIsNone(dom_out)

        # Training mode with GRL domain head
        model.train()
        d_train, dom_train = model(x, alpha=0.15, return_domain=True)
        self.assertEqual(d_train.shape, (2, 5))
        self.assertEqual(dom_train.shape, (2, 5))

    def test_subject_level_bootstrap(self):
        """Bootstrap CI must have lower <= upper bound."""
        dummy_preds = []
        for subj in range(20):
            for _ in range(10):
                dummy_preds.append({
                    "dataset_name": "ds1",
                    "subject_id": f"sub-{subj:03d}",
                    "true_label_idx": 0,
                    "pred_label_idx": 0 if subj % 2 == 0 else 1,
                })
        ci = subject_level_bootstrap(dummy_preds, n_resamples=50, seed=42)
        self.assertIn("accuracy_ci_95", ci)
        low, high = ci["accuracy_ci_95"]
        self.assertLessEqual(low, high)

    def test_canonical_baseline_files_intact(self):
        """Ensure checkpoints/best_model.pt and results/metrics.json are unmodified in git."""
        result = subprocess.run(
            ["git", "status", "--porcelain", "checkpoints/best_model.pt", "results/metrics.json"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        self.assertEqual(result.stdout.strip(), "", "Canonical baseline files must remain untouched")


if __name__ == "__main__":
    unittest.main()
