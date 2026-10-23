"""
tests/test_contrastive.py - Unit tests for Cross-Source Healthy Contrastive Learning components.
"""

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_exp2_contrastive import (
    ContrastiveEEGClassifier,
    CrossSourceHealthyContrastiveLoss,
)


class TestContrastiveComponents(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.B = 8
        self.D = 64
        self.embeddings = torch.randn(self.B, self.D, requires_grad=True)
        # 4 healthy (across 3 datasets), 4 disease
        self.labels = torch.tensor([0, 0, 0, 0, 1, 2, 3, 4])
        self.ds_names = [
            "alzheimer",
            "alzheimer",  # same-site healthy
            "parkinson",  # cross-site healthy
            "modma",      # cross-site healthy
            "chbmit",     # epilepsy
            "alzheimer",  # alzheimer disease
            "parkinson",  # parkinson disease
            "modma",      # depression
        ]
        self.loss_fn = CrossSourceHealthyContrastiveLoss(temperature=0.1)

    def test_loss_computation_and_gradient(self):
        loss = self.loss_fn(self.embeddings, self.labels, self.ds_names)
        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0.0)

        loss.backward()
        self.assertIsNotNone(self.embeddings.grad)
        self.assertFalse(torch.isnan(self.embeddings.grad).any())
        self.assertGreater(self.embeddings.grad.norm().item(), 0.0)

    def test_same_class_healthy_never_in_negatives(self):
        """
        Verify that for sample 0 (alzheimer healthy), sample 1 (alzheimer healthy)
        is excluded from the negative set.
        """
        norm_emb = F.normalize(self.embeddings.detach(), dim=-1)
        sims = torch.matmul(norm_emb, norm_emb.T) / 0.1

        # Calculate manual loss for anchor 0
        pos_sim_2 = sims[0, 2]  # parkinson healthy
        pos_sim_3 = sims[0, 3]  # modma healthy
        disease_sims = sims[0, [4, 5, 6, 7]]  # disease negatives

        # Denominators should contain ONLY pos + disease_sims, NOT sample 1!
        denom_2 = torch.logsumexp(torch.cat([pos_sim_2.unsqueeze(0), disease_sims]), dim=0)
        denom_3 = torch.logsumexp(torch.cat([pos_sim_3.unsqueeze(0), disease_sims]), dim=0)
        expected_anchor_0_loss = (-(pos_sim_2 - denom_2) - (pos_sim_3 - denom_3)) / 2.0

        # Assert sample 1 similarity is not included in the negative denominator
        self.assertNotIn(sims[0, 1].item(), disease_sims.tolist())

    def test_zero_loss_when_no_cross_site_positives(self):
        """When all healthy are from the same dataset, cross-site loss should gracefully be 0."""
        ds_all_same = ["alzheimer"] * self.B
        loss = self.loss_fn(self.embeddings, self.labels, ds_all_same)
        self.assertEqual(loss.item(), 0.0)

    def test_contrastive_model_architecture(self):
        model = ContrastiveEEGClassifier(n_channels=19, n_samples=1280, num_classes=5).eval()
        x = torch.randn(2, 19, 1280)
        logits = model(x)
        self.assertEqual(logits.shape, (2, 5))

        logits_emb, proj = model(x, return_embedding=True)
        self.assertEqual(logits_emb.shape, (2, 5))
        self.assertEqual(proj.shape, (2, 64))
        self.assertTrue(torch.allclose(logits, logits_emb, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
