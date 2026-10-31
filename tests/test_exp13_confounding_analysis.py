"""
tests/test_exp13_confounding_analysis.py - Unit tests for Exp 13 diagnostic analysis.
"""

import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_exp13_confounding_analysis import (
    verify_canonical_hashes,
    EXPECTED_CANONICAL_MODEL_SHA256,
    EXPECTED_CANONICAL_METRICS_SHA256,
)


class TestExp13ConfoundingAnalysis(unittest.TestCase):
    """Test hash verification and subject voting consistency."""

    def test_canonical_file_hashes_intact(self):
        """Verify checkpoints/best_model.pt and results/metrics.json are bit-for-bit unchanged."""
        self.assertTrue(verify_canonical_hashes(), "Canonical file hash mismatch!")

    def test_majority_voting_aggregation(self):
        """Verify majority voting logic aggregates correctly per subject."""
        data = {
            "subject_id": ["s1", "s1", "s1", "s2", "s2"],
            "pred": [0, 0, 2, 3, 3],
            "true": [0, 0, 0, 3, 3],
        }
        df = pd.DataFrame(data)
        maj = df.groupby("subject_id")["pred"].apply(lambda x: x.mode()[0])
        self.assertEqual(maj["s1"], 0)
        self.assertEqual(maj["s2"], 3)


if __name__ == "__main__":
    unittest.main()
