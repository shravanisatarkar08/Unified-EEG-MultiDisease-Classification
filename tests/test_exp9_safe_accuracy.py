"""
tests/test_exp9_safe_accuracy.py - Unit and regression tests for Exp 9 Safe Accuracy Protocol.
"""

import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.eeg_classifier import EEGClassifier
from models.spectral_transformer import SpectralHybridCNNTransformer
from scripts.run_exp9_safe_accuracy import compute_metrics, subject_level_bootstrap

EXPECTED_CANONICAL_MODEL_SHA256 = "e8cbd5bd9ec79b6e46524fbc2cad7b84a59c22d83c07a5fe2d335f3021441111"
EXPECTED_CANONICAL_METRICS_SHA256 = "5f0a10003b0fe06b383614c097ab5287ae526a073a90dd420dda2bec1862ff3a"


class TestExp9SafeAccuracy(unittest.TestCase):
    """Test ensembling, metric computation, bootstrap CI, and absolute file safety."""

    def test_canonical_file_hashes_intact(self):
        """Verify that checkpoints/best_model.pt and results/metrics.json are bit-for-bit identical to backups."""
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

    def test_probability_blending_normalization(self):
        """Probability blending must yield valid probability distributions that sum to 1.0."""
        p1 = np.array([[0.1, 0.2, 0.3, 0.2, 0.2], [0.5, 0.1, 0.1, 0.1, 0.2]])
        p2 = np.array([[0.2, 0.1, 0.2, 0.3, 0.2], [0.2, 0.2, 0.2, 0.2, 0.2]])
        w = 0.6
        p_ens = w * p1 + (1.0 - w) * p2

        self.assertEqual(p_ens.shape, (2, 5))
        self.assertTrue(np.allclose(p_ens.sum(axis=1), np.ones(2)),
                        "Ensemble probabilities must sum to 1 across classes")

    def test_metric_computation(self):
        """Verify compute_metrics produces expected accuracy, balanced accuracy, and macro-F1."""
        y_true = np.array([0, 0, 1, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 1, 1, 2, 3, 4])  # 6/7 correct
        metas = [
            {"dataset_name": "ds1"},
            {"dataset_name": "ds1"},
            {"dataset_name": "ds2"},
            {"dataset_name": "ds2"},
            {"dataset_name": "ds3"},
            {"dataset_name": "ds4"},
            {"dataset_name": "ds5"},
        ]
        res = compute_metrics(y_true, y_pred, metas)
        self.assertAlmostEqual(res["accuracy"], 6 / 7, places=4)
        self.assertIn("balanced_accuracy", res)
        self.assertIn("macro_f1", res)
        self.assertIn("healthy_recall_by_dataset", res)

    def test_subject_level_bootstrap(self):
        """Subject-level bootstrap must return valid 95% confidence intervals."""
        dummy_preds = pd.DataFrame([
            {"dataset_name": "d1", "subject_id": f"s{i}", "true_label_idx": 0, "pred_label_idx": 0}
            for i in range(15)
        ] + [
            {"dataset_name": "d1", "subject_id": f"s{i}", "true_label_idx": 1, "pred_label_idx": 1}
            for i in range(15, 30)
        ])
        ci = subject_level_bootstrap(dummy_preds, n_resamples=50, seed=42)
        self.assertEqual(ci["accuracy_ci_95"], [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
