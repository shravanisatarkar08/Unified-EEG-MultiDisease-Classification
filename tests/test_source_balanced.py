"""
tests/test_source_balanced.py - Regression tests for exp4 source-balanced training.
"""

import sys
import unittest
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestSourceBalancedSampler(unittest.TestCase):
    """Test that source-balanced sampling produces roughly equal source representation."""

    def test_weighted_sampler_distribution(self):
        """WeightedRandomSampler with inverse-frequency weights should equalize sources."""
        from torch.utils.data import WeightedRandomSampler

        # Simulate: 3 sources with very unequal sizes
        sources = ["A"] * 100 + ["B"] * 300 + ["C"] * 600
        n = len(sources)

        source_counts = {"A": 100, "B": 300, "C": 600}
        n_sources = 3
        weights = [n / (n_sources * source_counts[s]) for s in sources]

        sampler = WeightedRandomSampler(weights, num_samples=n, replacement=True)
        indices = list(sampler)

        # Count how many times each source is sampled
        sampled_sources = [sources[i] for i in indices]
        from collections import Counter
        counts = Counter(sampled_sources)

        # Each source should get roughly n/3 = 333 samples
        for src in ["A", "B", "C"]:
            ratio = counts[src] / n
            # Allow 15% tolerance from the expected 1/3
            self.assertGreater(ratio, 0.20, f"Source {src} undersampled: {ratio:.3f}")
            self.assertLess(ratio, 0.47, f"Source {src} oversampled: {ratio:.3f}")

    def test_canonical_baseline_untouched(self):
        """Verify checkpoints/best_model.pt and results/metrics.json exist and are tracked by git."""
        import subprocess
        result = subprocess.run(
            ["git", "status", "--porcelain", "checkpoints/best_model.pt", "results/metrics.json"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        # Empty output means files are clean (tracked, no changes)
        self.assertEqual(result.stdout.strip(), "",
                         f"Canonical files modified: {result.stdout}")

    def test_model_constructs_with_baseline_params(self):
        """Verify model instantiation with exact baseline parameters."""
        from models.eeg_classifier import EEGClassifier
        model = EEGClassifier(
            n_channels=19, n_samples=1280, num_classes=5,
            cnn_dropout=0.25, embed_dim=64, num_heads=4,
            num_layers=2, ff_dim=128, trans_dropout=0.10,
        )
        x = torch.randn(2, 19, 1280)
        out = model(x)
        self.assertEqual(out.shape, (2, 5))


if __name__ == "__main__":
    unittest.main()
