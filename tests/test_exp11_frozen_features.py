"""
tests/test_exp11_frozen_features.py - Unit and regression tests for Exp 11 Frozen Features.
"""

import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.eeg_classifier import EEGClassifier
from scripts.run_exp11_frozen_features import (
    compute_comprehensive_metrics,
    extract_cls_from_exp6,
    EXPECTED_CANONICAL_MODEL_SHA256,
    EXPECTED_CANONICAL_METRICS_SHA256,
)


class TestExp11FrozenFeatures(unittest.TestCase):
    """Test feature extraction, scaling isolation, metrics, and canonical integrity."""

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

        self.assertTrue(model_path.exists(), "checkpoints/best_model.pt missing!")
        self.assertTrue(metrics_path.exists(), "results/metrics.json missing!")

        self.assertEqual(hash_file(model_path), EXPECTED_CANONICAL_MODEL_SHA256,
                         "Canonical model checkpoint hash changed!")
        self.assertEqual(hash_file(metrics_path), EXPECTED_CANONICAL_METRICS_SHA256,
                         "Canonical metrics hash changed!")

    def test_exp6_cls_extraction_shape(self):
        """extract_cls_from_exp6 must return [B, 64] float tensor."""
        model = EEGClassifier(
            n_channels=19, n_samples=1280, num_classes=5,
            cnn_dropout=0.1, embed_dim=64, num_heads=4,
            num_layers=2, ff_dim=128, trans_dropout=0.1,
        )
        model.eval()
        dummy_x = torch.randn(4, 19, 1280)
        with torch.no_grad():
            emb = extract_cls_from_exp6(model, dummy_x)
        self.assertEqual(emb.shape, (4, 64))
        self.assertEqual(emb.dtype, torch.float32)

    def test_scaler_isolation(self):
        """StandardScaler must fit strictly on train and transform val without leakage."""
        np.random.seed(42)
        X_tr = np.random.randn(100, 64) * 2.0 + 5.0
        X_va = np.random.randn(20, 64) * 3.0 + 10.0

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_va_scaled = scaler.transform(X_va)

        # Train mean should be close to 0, std close to 1
        self.assertTrue(np.allclose(X_tr_scaled.mean(axis=0), 0.0, atol=1e-5))
        self.assertTrue(np.allclose(X_tr_scaled.std(axis=0), 1.0, atol=1e-5))
        # Val mean should NOT be forced to 0 (isolated)
        self.assertFalse(np.allclose(X_va_scaled.mean(axis=0), 0.0, atol=0.1))

    def test_metrics_computation(self):
        """compute_comprehensive_metrics outputs correct types and keys."""
        y_true = np.array([0, 0, 1, 2, 3, 4, 0, 1, 2, 3])
        y_pred = np.array([0, 3, 1, 2, 3, 4, 0, 1, 2, 0])
        datasets = np.array(["alzheimer", "parkinson", "modma", "alzheimer", "parkinson",
                             "modma", "alzheimer", "parkinson", "modma", "alzheimer"])

        m = compute_comprehensive_metrics(y_true, y_pred, datasets)
        self.assertIn("accuracy", m)
        self.assertIn("balanced_accuracy", m)
        self.assertIn("macro_f1", m)
        self.assertIn("per_class", m)
        self.assertIn("healthy_recall_by_dataset", m)
        self.assertIn("confusion_matrix", m)
        self.assertTrue(0.0 <= m["accuracy"] <= 1.0)
        self.assertTrue(0.0 <= m["balanced_accuracy"] <= 1.0)
        self.assertEqual(len(m["confusion_matrix"]), 5)


if __name__ == "__main__":
    unittest.main()
