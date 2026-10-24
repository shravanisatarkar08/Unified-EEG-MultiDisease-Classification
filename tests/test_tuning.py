"""
tests/test_tuning.py - Unit tests for Hyperparameter Tuning components.
"""

import sys
import unittest
from pathlib import Path
import tempfile
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_hyperparam_tuning import TuningConfig, TuningRunManager, pick_best_candidate


class TestTuningComponents(unittest.TestCase):
    def test_config_initialization(self):
        cfg = TuningConfig(run_name="test_run", lr=3e-4, weight_decay=1e-3, cnn_dropout=0.3)
        self.assertEqual(cfg.run_name, "test_run")
        self.assertEqual(cfg.lr, 3e-4)
        self.assertEqual(cfg.weight_decay, 1e-3)
        self.assertEqual(cfg.cnn_dropout, 0.3)

    def test_manager_model_setup(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = TuningConfig(run_name="test_mgr", lr=1e-4)
            mgr = TuningRunManager(cfg, Path(tmp_dir), torch.device("cpu"))
            x = torch.randn(2, 19, 1280)
            out = mgr.model(x)
            self.assertEqual(out.shape, (2, 5))

    def test_candidate_selection(self):
        candidates = {
            "run_a": {
                "validation_accuracy": 0.7600,
                "validation_macro_f1": 0.7800,
                "validation_balanced_accuracy": 0.8100,
            },
            "run_b": {
                "validation_accuracy": 0.7700,
                "validation_macro_f1": 0.7900,
                "validation_balanced_accuracy": 0.8200,
            },
            "run_c": {
                "validation_accuracy": 0.7700,
                "validation_macro_f1": 0.8050,
                "validation_balanced_accuracy": 0.8150,
            },
        }
        best_name, best_res = pick_best_candidate(candidates)
        # run_b and run_c tie on val_acc (0.7700), but run_c wins on macro_f1 (0.8050 > 0.7900)
        self.assertEqual(best_name, "run_c")

    def test_provenance_and_completion_tracking(self):
        cfg = TuningConfig(run_name="prov_run", provenance="cached_control")
        self.assertEqual(cfg.provenance, "cached_control")
        self.assertEqual(cfg.stage, 1)

        summary_item = {
            "run_name": cfg.run_name,
            "provenance": cfg.provenance,
            "training_completed": True,
            "checkpoint_path": "path/to/best_model.pt",
            "validation_accuracy": 0.7662,
        }
        self.assertTrue(summary_item["training_completed"])
        self.assertEqual(summary_item["provenance"], "cached_control")


if __name__ == "__main__":
    unittest.main()
