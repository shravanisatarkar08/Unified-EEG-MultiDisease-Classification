"""
tests/test_exp6_dann_refinement.py - Unit and regression tests for Exp 6 DANN refinement.
"""

import json
import subprocess
import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_exp5_dann_transformer import DANNConfig


class TestExp6Refinement(unittest.TestCase):
    """Verify Exp 5 preservation, Exp 6 candidate configurations, and canonical baseline safety."""

    def test_exp5_champion_artifacts_intact(self):
        """Verify Exp 5 metrics, checkpoint, and predictions exist and record 77.02% val acc."""
        exp5_dir = PROJECT_ROOT / "results" / "experiments" / "exp5_dann" / "dann_alpha0.10_linear"
        metrics_file = exp5_dir / "val_metrics.json"
        self.assertTrue(metrics_file.exists(), "Exp 5 val_metrics.json must exist")

        with open(metrics_file, "r", encoding="utf-8") as f:
            m = json.load(f)

        self.assertEqual(m["validation_accuracy"], 0.7702, "Exp 5 val_acc must be 77.02%")
        self.assertEqual(m["validation_balanced_accuracy"], 0.8438)
        self.assertEqual(m["validation_macro_f1"], 0.8043)
        self.assertEqual(m["selected_epoch"], 6)

        preds_file = exp5_dir / "val_predictions.csv"
        self.assertTrue(preds_file.exists(), "Exp 5 val_predictions.csv must exist")

    def test_exp6_candidate_configs(self):
        """Verify Candidate A has max_alpha=0.05 and Candidate B has max_alpha=0.15."""
        cfg_a = DANNConfig(run_name="dann_alpha0.05_linear", max_alpha=0.05, alpha_schedule="linear", domain_loss_weight=0.10)
        cfg_b = DANNConfig(run_name="dann_alpha0.15_linear", max_alpha=0.15, alpha_schedule="linear", domain_loss_weight=0.10)

        self.assertEqual(cfg_a.max_alpha, 0.05)
        self.assertEqual(cfg_b.max_alpha, 0.15)
        self.assertEqual(cfg_a.domain_loss_weight, 0.10)
        self.assertEqual(cfg_b.domain_loss_weight, 0.10)
        self.assertEqual(cfg_a.lr, 5e-4)
        self.assertEqual(cfg_b.lr, 5e-4)

    def test_canonical_baseline_files_unmodified(self):
        """Verify checkpoints/best_model.pt and results/metrics.json are unmodified in git."""
        result = subprocess.run(
            ["git", "status", "--porcelain", "checkpoints/best_model.pt", "results/metrics.json"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        self.assertEqual(result.stdout.strip(), "", "Canonical baseline files must remain untouched")


if __name__ == "__main__":
    unittest.main()
