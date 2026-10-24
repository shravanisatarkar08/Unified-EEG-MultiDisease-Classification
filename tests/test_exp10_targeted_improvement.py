"""
tests/test_exp10_targeted_improvement.py - Unit and regression tests for Exp 10.
"""

import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.eeg_classifier import EEGClassifier
from scripts.run_exp10_targeted_improvement import compute_hmd_loss, EXP6_CKPT_PATH

EXPECTED_CANONICAL_MODEL_SHA256 = "e8cbd5bd9ec79b6e46524fbc2cad7b84a59c22d83c07a5fe2d335f3021441111"
EXPECTED_CANONICAL_METRICS_SHA256 = "5f0a10003b0fe06b383614c097ab5287ae526a073a90dd420dda2bec1862ff3a"


class TestExp10(unittest.TestCase):
    """Test HMD loss, Exp 6 checkpoint accessibility, and canonical file safety."""

    def test_canonical_file_hashes_intact(self):
        """Verify that checkpoints/best_model.pt and results/metrics.json are bit-for-bit unchanged."""
        def hash_file(p):
            h = hashlib.sha256()
            with open(p, "rb") as f:
                while chunk := f.read(8192):
                    h.update(chunk)
            return h.hexdigest()

        model_path = PROJECT_ROOT / "checkpoints" / "best_model.pt"
        metrics_path = PROJECT_ROOT / "results" / "metrics.json"

        self.assertTrue(model_path.exists())
        self.assertTrue(metrics_path.exists())
        self.assertEqual(hash_file(model_path), EXPECTED_CANONICAL_MODEL_SHA256)
        self.assertEqual(hash_file(metrics_path), EXPECTED_CANONICAL_METRICS_SHA256)

    def test_hmd_loss_computation(self):
        """Verify HMD loss computes penalties and propagates gradients."""
        logits = torch.randn(4, 5, requires_grad=True)
        labels = torch.tensor([0, 2, 0, 3])
        metas = [
            {"dataset_name": "alzheimer"},
            {"dataset_name": "alzheimer"},
            {"dataset_name": "parkinson"},
            {"dataset_name": "parkinson"},
        ]

        loss = compute_hmd_loss(logits, labels, metas, margin=0.5)
        self.assertGreaterEqual(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertGreater(logits.grad.norm().item(), 0.0)

    def test_exp6_checkpoint_exists(self):
        """Verify Exp 6 champion checkpoint is accessible."""
        self.assertTrue(EXP6_CKPT_PATH.exists(), f"Exp 6 checkpoint missing at {EXP6_CKPT_PATH}")


if __name__ == "__main__":
    unittest.main()
