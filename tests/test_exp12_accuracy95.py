"""
tests/test_exp12_accuracy95.py - Unit and regression tests for Exp 12 Tri-Model Heterogeneous Ensemble.
"""

import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_exp12_accuracy95 import (
    compute_metrics,
    subject_level_bootstrap,
    EXPECTED_CANONICAL_MODEL_SHA256,
    EXPECTED_CANONICAL_METRICS_SHA256,
)


class TestExp12Accuracy95(unittest.TestCase):
    """Test ensembling validity, bootstrap bounds, metric schema, and canonical file safety."""

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

    def test_tri_model_probability_simplex(self):
        """Tri-model convex combination must sum to 1.0 across classes."""
        p1 = np.array([[0.3, 0.2, 0.1, 0.2, 0.2], [0.1, 0.5, 0.1, 0.1, 0.2]])
        p2 = np.array([[0.2, 0.3, 0.1, 0.2, 0.2], [0.2, 0.4, 0.2, 0.1, 0.1]])
        p3 = np.array([[0.1, 0.1, 0.4, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1, 0.2]])

        w1, w2, w3 = 0.20, 0.50, 0.30
        self.assertAlmostEqual(w1 + w2 + w3, 1.0)

        p_tri = w1 * p1 + w2 * p2 + w3 * p3
        self.assertEqual(p_tri.shape, (2, 5))
        self.assertTrue(np.allclose(p_tri.sum(axis=-1), np.ones(2)),
                        "Tri-ensemble probabilities must sum to 1 across classes")
        self.assertTrue((p_tri >= 0.0).all() and (p_tri <= 1.0).all(),
                        "Probabilities must be within [0, 1]")

    def test_metrics_schema_and_ranges(self):
        """compute_metrics returns proper keys, ranges [0, 1], and healthy breakdown."""
        y_true = np.array([0, 0, 1, 2, 3, 4, 0, 1, 2, 3])
        y_pred = np.array([0, 3, 1, 2, 3, 4, 0, 1, 2, 0])
        datasets = np.array(["alzheimer", "parkinson", "modma", "alzheimer", "parkinson",
                             "modma", "alzheimer", "parkinson", "modma", "alzheimer"])

        m = compute_metrics(y_true, y_pred, datasets)
        self.assertIn("accuracy", m)
        self.assertIn("balanced_accuracy", m)
        self.assertIn("macro_f1", m)
        self.assertIn("healthy_recall_by_dataset", m)
        self.assertTrue(0.0 <= m["accuracy"] <= 1.0)
        self.assertTrue(0.0 <= m["balanced_accuracy"] <= 1.0)

    def test_bootstrap_ci_monotonicity(self):
        """Bootstrap confidence intervals must have lower <= upper."""
        y_true = np.array([0, 1, 2, 3, 4] * 10)
        y_pred = np.array([0, 1, 2, 3, 4] * 10)
        subjects = np.array([f"sub_{i}" for i in range(10)] * 5)

        ci = subject_level_bootstrap(y_true, y_pred, subjects, n_resamples=50, seed=42)
        self.assertLessEqual(ci["accuracy_ci_95"][0], ci["accuracy_ci_95"][1])
        self.assertLessEqual(ci["balanced_accuracy_ci_95"][0], ci["balanced_accuracy_ci_95"][1])
        self.assertLessEqual(ci["macro_f1_ci_95"][0], ci["macro_f1_ci_95"][1])


if __name__ == "__main__":
    unittest.main()
