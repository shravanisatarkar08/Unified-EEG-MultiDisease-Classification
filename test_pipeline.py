"""
test_pipeline.py - Smoke test suite for the Unified EEG Multi-Disease Classification pipeline.

Tests covered:
  1. Package imports (preprocessing, models)
  2. Config values
  3. Segmentation (continuous + labeled)
  4. Normalization
  5. Metadata dataclass
  6. Channel normalization
  7. CNN (EEGFeatureExtractor) forward pass + tensor shapes
  8. Transformer (EEGTransformerEncoder) forward pass + tensor shapes
  9. Full EEGClassifier end-to-end forward pass
  10. Dataset module imports (alzheimers, parkinsons, modma, tuab)
  11. CHB-MIT parse_summary (without needing actual EDF files)
  12. Multi-class EEGClassifier (5 classes)

Run with:
    python test_pipeline.py
    or:
    python -m unittest test_pipeline
"""

import sys
import os
import unittest
import numpy as np

# Ensure we run from project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch


class TestConfig(unittest.TestCase):
    def test_import_and_values(self):
        from preprocessing.config import (TARGET_SRATE, WINDOW_SEC, OVERLAP,
                                          N_CHANNELS, WINDOW_SAMPLES, COMMON_CHANNELS,
                                          L_FREQ, H_FREQ)
        self.assertEqual(TARGET_SRATE, 256)
        self.assertEqual(WINDOW_SEC, 5)
        self.assertAlmostEqual(OVERLAP, 0.5)
        self.assertEqual(N_CHANNELS, 19)
        self.assertEqual(WINDOW_SAMPLES, 1280)
        self.assertEqual(len(COMMON_CHANNELS), 19)
        self.assertEqual(L_FREQ, 0.5)
        self.assertEqual(H_FREQ, 45.0)


class TestSegmentation(unittest.TestCase):
    def setUp(self):
        from preprocessing.config import TARGET_SRATE, N_CHANNELS
        self.srate = TARGET_SRATE
        self.n_ch = N_CHANNELS
        # 60 seconds of random EEG
        self.data = np.random.randn(self.n_ch, 60 * self.srate).astype(np.float32)

    def test_segment_continuous_shape(self):
        from preprocessing.segment import segment_continuous
        from preprocessing.config import WINDOW_SAMPLES
        windows, ranges = segment_continuous(self.data, srate=self.srate)
        self.assertEqual(windows.ndim, 3)
        self.assertEqual(windows.shape[1], self.n_ch)
        self.assertEqual(windows.shape[2], WINDOW_SAMPLES)
        self.assertEqual(len(ranges), windows.shape[0])

    def test_segment_labeled_seizure(self):
        from preprocessing.segment import segment_labeled
        intervals = [(10.0, 25.0)]  # 15s seizure
        windows, ranges = segment_labeled(self.data, intervals, 'seizure',
                                          srate=self.srate)
        self.assertGreater(len(windows), 0)
        self.assertEqual(len(windows), len(ranges))

    def test_segment_too_short(self):
        from preprocessing.segment import segment_continuous
        short_data = np.random.randn(self.n_ch, 100)
        windows, ranges = segment_continuous(short_data, srate=self.srate)
        self.assertEqual(len(windows), 0)


class TestNormalization(unittest.TestCase):
    def test_normalize_windows(self):
        from preprocessing.common import normalize_windows
        windows = np.random.randn(10, 19, 1280).astype(np.float32)
        normed = normalize_windows(windows)
        self.assertEqual(normed.shape, windows.shape)
        # Each channel in each window should be approx zero-mean, unit-std
        means = normed.mean(axis=2)
        stds  = normed.std(axis=2)
        np.testing.assert_allclose(means, 0.0, atol=1e-5)
        np.testing.assert_allclose(stds, 1.0, atol=1e-4)


class TestMetadata(unittest.TestCase):
    def test_window_metadata_fields(self):
        from preprocessing.metadata import WindowMetadata
        m = WindowMetadata(
            dataset='chbmit', subject_id='chb01', recording_id='chb01_01',
            label='seizure', window_idx=0, window_start=10.0, window_end=15.0,
            n_channels=19, srate=256.0, n_channels_found=19, file_path='test.npy',
        )
        self.assertEqual(m.dataset, 'chbmit')
        self.assertEqual(m.label, 'seizure')
        self.assertEqual(m.n_channels, 19)
        self.assertAlmostEqual(m.window_end - m.window_start, 5.0)


class TestChannelNormalization(unittest.TestCase):
    def test_normalize_known_channels(self):
        from preprocessing.common import normalize_channel_name
        self.assertEqual(normalize_channel_name('EEG Fp1-REF'), 'Fp1')
        self.assertEqual(normalize_channel_name('T7'), 'T3')
        self.assertEqual(normalize_channel_name('T8'), 'T4')
        self.assertEqual(normalize_channel_name('P7'), 'T5')
        self.assertEqual(normalize_channel_name('P8'), 'T6')
        self.assertIsNone(normalize_channel_name('EMG1'))

    def test_normalize_case_variants(self):
        from preprocessing.common import normalize_channel_name
        self.assertEqual(normalize_channel_name('FP1'), 'Fp1')
        self.assertEqual(normalize_channel_name('CZ'), 'Cz')


class TestCNNForwardPass(unittest.TestCase):
    def setUp(self):
        from models import EEGFeatureExtractor
        self.B, self.C, self.T = 4, 19, 1280
        self.embed_dim = 64
        self.n_classes = 2
        self.model = EEGFeatureExtractor(
            n_channels=self.C, n_samples=self.T,
            n_classes=self.n_classes, embed_dim=self.embed_dim,
        )
        self.model.eval()
        self.x = torch.randn(self.B, self.C, self.T)

    def test_extract_features_shape(self):
        with torch.no_grad():
            feat = self.model.extract_features(self.x)
        self.assertEqual(feat.shape, (self.B, self.T // 32, self.embed_dim))

    def test_forward_logits_shape(self):
        with torch.no_grad():
            logits = self.model(self.x)
        self.assertEqual(logits.shape, (self.B, self.n_classes))

    def test_no_nan_in_output(self):
        with torch.no_grad():
            logits = self.model(self.x)
        self.assertFalse(torch.isnan(logits).any().item())


class TestTransformerForwardPass(unittest.TestCase):
    def setUp(self):
        from models import EEGTransformerEncoder
        self.B, self.N, self.D = 4, 40, 64
        self.n_classes = 2
        self.model = EEGTransformerEncoder(
            embed_dim=self.D, num_heads=4, num_layers=2,
            ff_dim=128, dropout=0.0, num_classes=self.n_classes,
        )
        self.model.eval()
        self.tokens = torch.randn(self.B, self.N, self.D)

    def test_output_shape(self):
        with torch.no_grad():
            logits = self.model(self.tokens)
        self.assertEqual(logits.shape, (self.B, self.n_classes))

    def test_no_nan(self):
        with torch.no_grad():
            logits = self.model(self.tokens)
        self.assertFalse(torch.isnan(logits).any().item())


class TestEEGClassifierEndToEnd(unittest.TestCase):
    def setUp(self):
        from models import EEGClassifier
        self.B, self.C, self.T = 4, 19, 1280

    def test_binary_classification(self):
        from models import EEGClassifier
        model = EEGClassifier(n_channels=self.C, n_samples=self.T, num_classes=2)
        model.eval()
        x = torch.randn(self.B, self.C, self.T)
        with torch.no_grad():
            logits = model(x)
        self.assertEqual(logits.shape, (self.B, 2))

    def test_multiclass_classification(self):
        from models import EEGClassifier
        model = EEGClassifier(n_channels=self.C, n_samples=self.T, num_classes=5)
        model.eval()
        x = torch.randn(self.B, self.C, self.T)
        with torch.no_grad():
            logits = model(x)
        self.assertEqual(logits.shape, (self.B, 5))

    def test_get_features_shape(self):
        from models import EEGClassifier
        model = EEGClassifier(n_channels=self.C, n_samples=self.T,
                               num_classes=2, embed_dim=64)
        model.eval()
        x = torch.randn(self.B, self.C, self.T)
        with torch.no_grad():
            feat = model.get_features(x)
        self.assertEqual(feat.shape, (self.B, self.T // 32, 64))

    def test_no_nan_in_logits(self):
        from models import EEGClassifier
        model = EEGClassifier(n_channels=self.C, n_samples=self.T, num_classes=2)
        model.eval()
        x = torch.randn(self.B, self.C, self.T)
        with torch.no_grad():
            logits = model(x)
        self.assertFalse(torch.isnan(logits).any().item())


class TestDatasetModuleImports(unittest.TestCase):
    def test_chbmit_import(self):
        from preprocessing.datasets import chbmit
        self.assertTrue(callable(chbmit.parse_summary))
        self.assertTrue(callable(chbmit.process_dataset))

    def test_alzheimers_import(self):
        from preprocessing.datasets import alzheimers
        self.assertTrue(callable(alzheimers.process_dataset))

    def test_parkinsons_import(self):
        from preprocessing.datasets import parkinsons
        self.assertTrue(callable(parkinsons.process_dataset))

    def test_modma_import(self):
        from preprocessing.datasets import modma
        self.assertTrue(callable(modma.process_dataset))
        self.assertTrue(callable(modma.inspect_raw_dir))

    def test_tuab_import(self):
        from preprocessing.datasets import tuab
        self.assertTrue(callable(tuab.process_dataset))
        self.assertTrue(callable(tuab.is_available))
        self.assertEqual(tuab.STATUS, 'UNAVAILABLE')


class TestChbmitParseSummary(unittest.TestCase):
    def test_parse_summary_basic(self):
        """Test parse_summary with an in-memory temp file."""
        import tempfile
        from preprocessing.datasets.chbmit import parse_summary

        summary_content = """File Name: chb01_03.edf
File Start Time: 13:43:04
File End Time: 14:43:04
Number of Seizures in File: 1
Seizure Start Time: 2996 seconds
Seizure End Time: 3036 seconds

File Name: chb01_04.edf
File Start Time: 14:43:04
File End Time: 15:43:04
Number of Seizures in File: 0
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                         delete=False) as f:
            f.write(summary_content)
            tmp_path = f.name
        try:
            seizures = parse_summary(tmp_path)
            self.assertIn('chb01_03.edf', seizures)
            self.assertEqual(len(seizures['chb01_03.edf']), 1)
            self.assertEqual(seizures['chb01_03.edf'][0], (2996, 3036))
            # chb01_04 has no seizures listed
            self.assertEqual(seizures.get('chb01_04.edf', []), [])
        finally:
            os.unlink(tmp_path)


if __name__ == '__main__':
    # Run with verbose output
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
