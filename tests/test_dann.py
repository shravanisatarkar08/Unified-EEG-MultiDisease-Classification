"""
tests/test_dann.py - Unit and regression tests for Experiment 5 (DANN with GRL).
"""

import sys
import unittest
from pathlib import Path
import subprocess

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.eeg_classifier import EEGClassifier
from scripts.run_exp5_dann_transformer import (
    GradReverse,
    grad_reverse,
    DANNCNNTransformer,
    DOMAIN_TO_IDX,
    IDX_TO_DOMAIN,
    NUM_DOMAINS,
    get_alpha,
    DANNConfig,
)


class TestDANNComponents(unittest.TestCase):
    """Test GRL, model wrapper, domain mapping, and safety constraints."""

    def test_gradient_reversal_forward_backward(self):
        """Forward should be identity; backward should invert and scale gradient by alpha."""
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        alpha = 0.5

        # Forward pass
        y = grad_reverse(x, alpha)
        self.assertTrue(torch.equal(x, y), "GRL forward must be exact identity")

        # Backward pass: dLoss/dy = [1.0, 1.0, 1.0]
        # Expected dLoss/dx = -alpha * [1.0, 1.0, 1.0] = [-0.5, -0.5, -0.5]
        loss = y.sum()
        loss.backward()

        expected_grad = torch.tensor([-0.5, -0.5, -0.5])
        self.assertTrue(torch.allclose(x.grad, expected_grad),
                        f"Expected grad {expected_grad}, got {x.grad}")

    def test_dann_wrapper_shapes_and_modes(self):
        """DANN wrapper should produce [B, 5] disease logits and [B, 5] domain logits."""
        base_model = EEGClassifier(
            n_channels=19,
            n_samples=1280,
            num_classes=5,
            cnn_dropout=0.25,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            ff_dim=128,
            trans_dropout=0.10,
        )
        dann = DANNCNNTransformer(base_model, num_domains=NUM_DOMAINS)

        x = torch.randn(2, 19, 1280)

        # 1. Training mode with domain head active
        dann.train()
        d_logits, dom_logits = dann(x, alpha=0.1, return_domain=True)
        self.assertEqual(d_logits.shape, (2, 5))
        self.assertEqual(dom_logits.shape, (2, NUM_DOMAINS))

        # 2. Evaluation / inference mode (pure canonical interface)
        dann.eval()
        out_eval, dom_eval = dann(x, return_domain=False)
        self.assertEqual(out_eval.shape, (2, 5))
        self.assertIsNone(dom_eval)

        # 3. Embedding extraction shape
        emb = dann.extract_cls_embedding(x)
        self.assertEqual(emb.shape, (2, 64))

    def test_domain_label_mapping(self):
        """Verify 5 domains are mapped 1-to-1 without collisions."""
        expected_sources = {"alzheimer", "chbmit", "modma", "parkinson", "srm_healthy"}
        self.assertEqual(set(DOMAIN_TO_IDX.keys()), expected_sources)
        self.assertEqual(len(DOMAIN_TO_IDX), 5)
        for name, idx in DOMAIN_TO_IDX.items():
            self.assertEqual(IDX_TO_DOMAIN[idx], name)

    def test_alpha_schedules(self):
        """Verify dynamic GRL coefficient ramp-up."""
        cfg_linear = DANNConfig(run_name="test", max_alpha=0.20, alpha_schedule="linear")
        # Step 0 should be 0.0
        a0 = get_alpha(epoch=1, step=0, steps_per_epoch=10, total_epochs=10, cfg=cfg_linear)
        self.assertAlmostEqual(a0, 0.0)

        # Midpoint should be ~0.10
        amid = get_alpha(epoch=6, step=0, steps_per_epoch=10, total_epochs=10, cfg=cfg_linear)
        self.assertAlmostEqual(amid, 0.10, places=2)

        # Final step should be max_alpha
        afin = get_alpha(epoch=10, step=10, steps_per_epoch=10, total_epochs=10, cfg=cfg_linear)
        self.assertAlmostEqual(afin, 0.20, places=2)

    def test_canonical_baseline_files_untouched(self):
        """Verify checkpoints/best_model.pt and results/metrics.json are unmodified in git."""
        result = subprocess.run(
            ["git", "status", "--porcelain", "checkpoints/best_model.pt", "results/metrics.json"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        self.assertEqual(result.stdout.strip(), "", "Canonical baseline files must remain untouched")


if __name__ == "__main__":
    unittest.main()
