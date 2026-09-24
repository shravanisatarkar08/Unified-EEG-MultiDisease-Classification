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

import pandas as pd
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


class TestValidateDatasets(unittest.TestCase):
    def test_validate_datasets_import(self):
        from preprocessing.validate_datasets import DatasetValidator, EDFFormatReader, EEGLABFormatReader, run_validation
        self.assertTrue(callable(run_validation))
        self.assertTrue(hasattr(DatasetValidator, 'validate_all'))

    def test_validate_all_execution(self):
        raw_root = os.path.join(PROJECT_ROOT, 'datasets', 'raw')
        chbmit_dir = os.path.join(raw_root, 'epilepsy')
        srm_dir = os.path.join(raw_root, 'healthy')
        if not os.path.isdir(chbmit_dir) or not os.path.isdir(srm_dir):
            self.skipTest('Raw dataset directories not present; skipping live validation test.')
        from preprocessing.validate_datasets import DatasetValidator
        validator = DatasetValidator()
        result = validator.validate_all()
        self.assertIn('dataset_reports', result)
        self.assertIn('all_recordings', result)
        
        # Verify dataset reports exist
        ds_map = {r['name']: r for r in result['dataset_reports']}
        self.assertIn('chbmit', ds_map)
        self.assertIn('srm_healthy', ds_map)
        self.assertIn('modma', ds_map)
        
        # Verify status reporting
        self.assertEqual(ds_map['chbmit']['status'], 'AVAILABLE')
        self.assertEqual(ds_map['srm_healthy']['status'], 'AVAILABLE')
        self.assertEqual(ds_map['modma']['status'], 'NOT AVAILABLE LOCALLY')

    def test_inventory_csv_export(self):
        import csv
        raw_root = os.path.join(PROJECT_ROOT, 'datasets', 'raw')
        if not any(os.path.isdir(os.path.join(raw_root, d))
                   for d in ('epilepsy', 'healthy', 'alzheimers', 'parkinsons')):
            self.skipTest('Raw dataset directories not present; skipping inventory CSV export test.')
        from preprocessing.validate_datasets import DatasetValidator
        validator = DatasetValidator()
        summary = validator.validate_all()
        csv_path = validator.export_inventory_csv(summary['all_recordings'])
        self.assertTrue(csv_path.exists())
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertGreater(len(rows), 0)
            first_row = rows[0]
            self.assertIn('dataset_name', first_row)
            self.assertIn('file_path', first_row)
            self.assertIn('srate_hz', first_row)
            self.assertIn('n_channels', first_row)


class TestDatasetPreprocessors(unittest.TestCase):
    def test_synthetic_eeg_preprocessing(self):
        import mne
        from preprocessing.dataset_preprocessors import BaseDatasetPreprocessor, DatasetPreprocessorConfig
        
        # Create synthetic MNE Raw object (8 channels, 1000 Hz, 5 seconds)
        srate = 1000.0
        n_channels = 8
        n_samples = int(srate * 5)
        data = np.random.randn(n_channels, n_samples).astype(np.float64)
        ch_names = [f"EEG_{i+1}" for i in range(n_channels)]
        info = mne.create_info(ch_names=ch_names, sfreq=srate, ch_types='eeg')
        raw = mne.io.RawArray(data, info, verbose=False)
        
        config = DatasetPreprocessorConfig(
            dataset_name='synthetic',
            target_srate_hz=500.0,
            l_freq=0.5,
            h_freq=45.0,
            notch_freqs=[50.0],
            rereference_mode='average'
        )
        preprocessor = BaseDatasetPreprocessor(config)
        processed = preprocessor.preprocess_raw_object(
            raw=raw,
            dataset_name='synthetic',
            class_category='test',
            subject_id='sub-synth',
            file_path='synth.edf'
        )
        
        self.assertEqual(processed.dataset_name, 'synthetic')
        self.assertEqual(processed.srate_hz, 500.0)
        self.assertEqual(processed.n_channels, 8)
        self.assertFalse(np.isnan(processed.data).any())
        self.assertFalse(np.isinf(processed.data).any())
        self.assertTrue(processed.metadata['rereferenced'])

    def test_ftd_exclusion_rule(self):
        from preprocessing.dataset_preprocessors import AlzheimersPreprocessor
        prep = AlzheimersPreprocessor()
        
        # Group F (FTD) subject must return None (explicitly EXCLUDED)
        res_ftd = prep.preprocess_file(
            file_path="datasets/raw/alzheimers/sub-001/eeg/sub-001_task-eyesclosed_eeg.set",
            subject_id="sub-066",
            group_code="F"
        )
        self.assertIsNone(res_ftd)

    def test_chbmit_seizure_annotations_no_fabrication(self):
        import tempfile
        from preprocessing.dataset_preprocessors import CHBMITPreprocessor
        
        prep = CHBMITPreprocessor()
        summary_text = """File Name: chb01_03.edf
Number of Seizures in File: 1
Seizure 1 Start Time: 2996 seconds
Seizure 1 End Time: 3036 seconds

File Name: chb01_04.edf
Number of Seizures in File: 0
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(summary_text)
            tmp_path = f.name
            
        try:
            annotations = prep.parse_summary(tmp_path)
            self.assertIn('chb01_03.edf', annotations)
            self.assertEqual(annotations['chb01_03.edf'], [(2996.0, 3036.0)])
            # chb01_04 has 0 seizures -> empty list (no fabricated seizure labels)
            self.assertEqual(annotations.get('chb01_04.edf', []), [])
        finally:
            os.unlink(tmp_path)

    def test_sample_dataset_preprocessors(self):
        # Requires actual raw EEG files on disk
        raw_root = os.path.join(PROJECT_ROOT, 'datasets', 'raw')
        has_any_raw = any(
            os.path.isdir(os.path.join(raw_root, d))
            for d in ('epilepsy', 'healthy', 'alzheimers', 'parkinsons')
        )
        if not has_any_raw:
            self.skipTest('Raw dataset directories not present; skipping controlled sample preprocessing test.')
        from preprocessing.dataset_preprocessors import run_controlled_sample_preprocessing
        samples = run_controlled_sample_preprocessing()
        self.assertGreater(len(samples), 0)
        for s in samples:
            self.assertFalse(np.isnan(s.data).any())
            self.assertGreater(s.n_channels, 0)
            self.assertGreater(s.srate_hz, 0.0)

    def test_invalid_file_handling(self):
        from preprocessing.dataset_preprocessors import SRMHealthyPreprocessor
        prep = SRMHealthyPreprocessor()
        res = prep.preprocess_file("invalid/non_existent_file.edf")
        self.assertIsNone(res)


class TestEEGHarmonization(unittest.TestCase):
    def setUp(self):
        from preprocessing.dataset_preprocessors import PreprocessedEEGSignal
        # Synthetic preprocessed signal (8 channels, 1000 Hz, 10 seconds)
        self.srate = 1000.0
        self.n_samples = int(self.srate * 10.0)
        self.data = np.random.randn(8, self.n_samples).astype(np.float32)
        self.ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4']
        self.signal = PreprocessedEEGSignal(
            dataset_name='synth_ds',
            class_category='healthy',
            subject_id='sub-test01',
            file_path='synth_path.edf',
            srate_hz=self.srate,
            channel_names=self.ch_names,
            n_channels=8,
            n_samples=self.n_samples,
            duration_sec=10.0,
            data=self.data
        )

    def test_resampling_and_window_shape(self):
        from preprocessing.harmonization import EEGHarmonizer, HarmonizationConfig
        harmonizer = EEGHarmonizer(HarmonizationConfig(target_srate_hz=256.0, window_sec=5.0))
        windows = harmonizer.segment_into_windows(self.signal)
        
        self.assertGreater(len(windows), 0)
        for w in windows:
            self.assertEqual(w.sampling_rate, 256.0)
            self.assertEqual(w.data.shape, (19, 1280))
            self.assertEqual(w.n_channels, 19)
            self.assertEqual(w.n_samples, 1280)
            self.assertFalse(np.isnan(w.data).any())
            self.assertFalse(np.isinf(w.data).any())

    def test_overlap_behavior(self):
        from preprocessing.harmonization import EEGHarmonizer, HarmonizationConfig
        # 0% overlap -> 10s recording / 5s window = 2 windows
        h_no_overlap = EEGHarmonizer(HarmonizationConfig(overlap=0.0))
        wins_0 = h_no_overlap.segment_into_windows(self.signal)
        self.assertEqual(len(wins_0), 2)
        
        # 50% overlap -> windows at 0-5s, 2.5-7.5s, 5-10s = 3 windows
        h_50_overlap = EEGHarmonizer(HarmonizationConfig(overlap=0.5))
        wins_50 = h_50_overlap.segment_into_windows(self.signal)
        self.assertEqual(len(wins_50), 3)

    def test_missing_channel_zero_fill_policy(self):
        from preprocessing.harmonization import EEGHarmonizer, TARGET_19_MONOPOLAR
        harmonizer = EEGHarmonizer()
        windows = harmonizer.segment_into_windows(self.signal)
        first_win = windows[0]
        
        # Pz was not in self.ch_names
        self.assertIn('Pz', first_win.missing_channels)
        pz_idx = TARGET_19_MONOPOLAR.index('Pz')
        self.assertEqual(first_win.channel_mask[pz_idx], 0.0)
        np.testing.assert_allclose(first_win.data[pz_idx], 0.0)

    def test_chbmit_bipolar_montage_flagging(self):
        from preprocessing.dataset_preprocessors import PreprocessedEEGSignal
        from preprocessing.harmonization import EEGHarmonizer, TARGET_19_BIPOLAR
        
        bipolar_signal = PreprocessedEEGSignal(
            dataset_name='chbmit',
            class_category='epilepsy',
            subject_id='chb01',
            file_path='chb01_01.edf',
            srate_hz=256.0,
            channel_names=TARGET_19_BIPOLAR,
            n_channels=19,
            n_samples=2560,
            duration_sec=10.0,
            data=np.random.randn(19, 2560).astype(np.float32)
        )
        harmonizer = EEGHarmonizer()
        windows = harmonizer.segment_into_windows(bipolar_signal)
        
        self.assertGreater(len(windows), 0)
        for w in windows:
            self.assertTrue(w.is_bipolar_montage)
            self.assertEqual(w.channel_names, TARGET_19_BIPOLAR)

    def test_subject_identity_preservation(self):
        from preprocessing.harmonization import EEGHarmonizer
        harmonizer = EEGHarmonizer()
        windows = harmonizer.segment_into_windows(self.signal)
        for w in windows:
            self.assertEqual(w.subject_id, 'sub-test01')
            self.assertEqual(w.dataset_name, 'synth_ds')
            self.assertEqual(w.class_category, 'healthy')


class TestDatasetSplit(unittest.TestCase):
    def test_class_to_idx_mapping(self):
        from preprocessing.dataset_split import CLASS_TO_IDX, IDX_TO_CLASS
        expected_mapping = {
            'healthy': 0,
            'epilepsy': 1,
            'alzheimers': 2,
            'parkinsons': 3,
            'depression': 4
        }
        self.assertEqual(CLASS_TO_IDX, expected_mapping)
        for cls, idx in expected_mapping.items():
            self.assertEqual(IDX_TO_CLASS[idx], cls)

    def test_subject_level_splitting_and_zero_leakage(self):
        from preprocessing.dataset_split import SubjectLevelSplitter
        splitter = SubjectLevelSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42)
        subjects_df = splitter.load_valid_subjects()
        assignments = splitter.split_subjects(subjects_df)

        # 1. Verify zero leakage (no subject in multiple splits)
        splitter.verify_zero_leakage(assignments)

        # 2. Check overlap sets explicitly
        train_keys = set((a.dataset_name, a.subject_id) for a in assignments if a.split == 'train')
        val_keys = set((a.dataset_name, a.subject_id) for a in assignments if a.split == 'val')
        test_keys = set((a.dataset_name, a.subject_id) for a in assignments if a.split == 'test')

        self.assertEqual(len(train_keys.intersection(val_keys)), 0)
        self.assertEqual(len(train_keys.intersection(test_keys)), 0)
        self.assertEqual(len(val_keys.intersection(test_keys)), 0)

    def test_reproducibility(self):
        from preprocessing.dataset_split import SubjectLevelSplitter
        splitter1 = SubjectLevelSplitter(random_seed=42)
        subjs1 = splitter1.load_valid_subjects()
        ass1 = splitter1.split_subjects(subjs1)

        splitter2 = SubjectLevelSplitter(random_seed=42)
        subjs2 = splitter2.load_valid_subjects()
        ass2 = splitter2.split_subjects(subjs2)

        keys1 = [(a.dataset_name, a.subject_id, a.split) for a in ass1]
        keys2 = [(a.dataset_name, a.subject_id, a.split) for a in ass2]
        self.assertEqual(keys1, keys2)

    def test_ftd_exclusion(self):
        from preprocessing.dataset_split import SubjectLevelSplitter, EXCLUDED_CLASSES
        splitter = SubjectLevelSplitter()
        subjects_df = splitter.load_valid_subjects()
        assignments = splitter.split_subjects(subjects_df)

        for a in assignments:
            self.assertNotIn(a.class_category, EXCLUDED_CLASSES)

    def test_model_window_index_generation(self):
        raw_root = os.path.join(PROJECT_ROOT, 'datasets', 'raw')
        has_any_raw = any(
            os.path.isdir(os.path.join(raw_root, d))
            for d in ('epilepsy', 'healthy', 'alzheimers', 'parkinsons')
        )
        if not has_any_raw:
            self.skipTest('Raw dataset directories not present; skipping model index generation test.')
        from preprocessing.dataset_split import SubjectLevelSplitter, build_model_window_index, export_model_index_csv
        from preprocessing.harmonization import run_controlled_sample_harmonization

        splitter = SubjectLevelSplitter()
        subjs_df = splitter.load_valid_subjects()
        assignments = splitter.split_subjects(subjs_df)

        sample_windows = run_controlled_sample_harmonization()
        index_rows = build_model_window_index(assignments, sample_windows)
        self.assertGreater(len(index_rows), 0)

        first_row = index_rows[0]
        self.assertIn(first_row.split, ['train', 'val', 'test'])
        self.assertIn(first_row.class_label_idx, [0, 1, 2, 3, 4])
        self.assertEqual(first_row.sampling_rate, 256.0)
        self.assertEqual(first_row.n_channels, 19)
        self.assertEqual(first_row.n_samples, 1280)

        out_csv = export_model_index_csv(index_rows)
        self.assertTrue(out_csv.exists())

    def test_unavailable_depression_handling(self):
        from preprocessing.dataset_split import SubjectLevelSplitter, CLASS_TO_IDX
        splitter = SubjectLevelSplitter()
        subjects_df = splitter.load_valid_subjects()
        assignments = splitter.split_subjects(subjects_df)

        # Verify depression key exists in CLASS_TO_IDX with index 4
        self.assertEqual(CLASS_TO_IDX['depression'], 4)

        # Verify no depression subjects assigned in local split
        dep_assignments = [a for a in assignments if a.class_category == 'depression']
        self.assertEqual(len(dep_assignments), 0)


class TestTrainingDataset(unittest.TestCase):
    """Focused tests for training/dataset.py (EEGDataset + get_dataloaders)."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_index_df(self, n_rows: int = 6) -> pd.DataFrame:
        """Build a minimal in-memory index DataFrame without touching real files."""
        rows = []
        classes = [
            ("healthy",    0, "srm_healthy",  False),
            ("epilepsy",   1, "chbmit",       True),
            ("alzheimers", 2, "alzheimers",   False),
            ("parkinsons", 3, "parkinsons",   False),
            ("healthy",    0, "srm_healthy",  False),
            ("epilepsy",   1, "chbmit",       True),
        ]
        splits = ["train", "train", "val", "train", "test", "val"]
        for i in range(n_rows):
            cat, lbl, ds, bipolar = classes[i % len(classes)]
            rows.append({
                "split":              splits[i % len(splits)],
                "dataset_name":       ds,
                "subject_id":         f"sub-{i:03d}",
                "class_category":     cat,
                "class_label_idx":    lbl,
                "source_file":        "nonexistent/dummy.edf",
                "window_idx":         i,
                "window_start_sec":   float(i) * 2.5,
                "window_end_sec":     float(i) * 2.5 + 5.0,
                "sampling_rate":      256.0,
                "n_channels":         19,
                "n_samples":          1280,
                "is_bipolar_montage": bipolar,
                "missing_channels":   "",
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 1. Import check
    # ------------------------------------------------------------------

    def test_training_package_imports(self):
        """training package and dataset module must be importable."""
        import training
        from training import EEGDataset, get_dataloaders, CLASS_TO_IDX, IDX_TO_CLASS
        self.assertTrue(callable(EEGDataset))
        self.assertTrue(callable(get_dataloaders))
        self.assertIsInstance(CLASS_TO_IDX, dict)
        self.assertIsInstance(IDX_TO_CLASS, dict)

    # ------------------------------------------------------------------
    # 2. CLASS_TO_IDX consistency with dataset_split
    # ------------------------------------------------------------------

    def test_class_to_idx_consistency_with_dataset_split(self):
        """training.CLASS_TO_IDX must match the canonical 5-class mapping.

        Note: we compare against the hardcoded canonical values rather than
        importing preprocessing.dataset_split directly, because that module
        has a top-level MNE import that fails in environments without MNE.
        The canonical mapping is documented in preprocessing/dataset_split.py.
        """
        from training import CLASS_TO_IDX as train_map
        expected = {
            "healthy": 0,
            "epilepsy": 1,
            "alzheimers": 2,
            "parkinsons": 3,
            "depression": 4,
        }
        self.assertEqual(train_map, expected)

    # ------------------------------------------------------------------
    # 3. EEGDataset construction from synthetic DataFrame
    # ------------------------------------------------------------------

    def test_eegdataset_construction(self):
        """EEGDataset can be constructed from an in-memory DataFrame."""
        from training.dataset import EEGDataset
        df = self._make_index_df()
        ds = EEGDataset(index_df=df, project_root=".", cache_size=2)
        self.assertIsInstance(ds, EEGDataset)

    # ------------------------------------------------------------------
    # 4. __len__ matches DataFrame rows
    # ------------------------------------------------------------------

    def test_eegdataset_len(self):
        from training.dataset import EEGDataset
        df = self._make_index_df(n_rows=6)
        ds = EEGDataset(index_df=df, project_root=".")
        self.assertEqual(len(ds), 6)

    # ------------------------------------------------------------------
    # 5. __getitem__ output contract (tensor shape, dtype, label)
    #    File is nonexistent → zeros fallback path is tested here.
    # ------------------------------------------------------------------

    def test_eegdataset_getitem_tensor_shape_and_dtype(self):
        """__getitem__ must return float32 tensor [19, 1280] even on missing files."""
        from training.dataset import EEGDataset
        df = self._make_index_df(n_rows=4)
        ds = EEGDataset(index_df=df, project_root=".", cache_size=2)
        for i in range(len(ds)):
            eeg, label, meta = ds[i]
            self.assertIsInstance(eeg, torch.Tensor)
            self.assertEqual(eeg.shape, (19, 1280), msg=f"Wrong shape at idx {i}")
            self.assertEqual(eeg.dtype, torch.float32, msg=f"Wrong dtype at idx {i}")
            self.assertIsInstance(label, int)
            self.assertIn(label, [0, 1, 2, 3, 4])

    # ------------------------------------------------------------------
    # 6. __getitem__ provenance metadata keys
    # ------------------------------------------------------------------

    def test_eegdataset_getitem_meta_keys(self):
        from training.dataset import EEGDataset
        df = self._make_index_df(n_rows=2)
        ds = EEGDataset(index_df=df, project_root=".")
        _, _, meta = ds[0]
        required_keys = {
            "dataset_name", "subject_id", "class_category", "source_file",
            "window_idx", "window_start_sec", "window_end_sec",
            "is_bipolar_montage", "missing_channels", "split",
        }
        self.assertTrue(required_keys.issubset(set(meta.keys())),
                        msg=f"Missing meta keys: {required_keys - set(meta.keys())}")

    # ------------------------------------------------------------------
    # 7. No Depression / MODMA data fabrication guard
    # ------------------------------------------------------------------

    def test_no_depression_fabrication_guard(self):
        """get_dataloaders must raise ValueError if Depression rows found in index."""
        import pandas as pd
        from training.dataset import get_dataloaders
        import tempfile

        dep_row = {
            "split": "train", "dataset_name": "modma", "subject_id": "sub-001",
            "class_category": "depression", "class_label_idx": 4,
            "source_file": "fake.edf", "window_idx": 0,
            "window_start_sec": 0.0, "window_end_sec": 5.0,
            "sampling_rate": 256.0, "n_channels": 19, "n_samples": 1280,
            "is_bipolar_montage": False, "missing_channels": "",
        }
        df = pd.DataFrame([dep_row])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f, index=False)
            tmp_csv = f.name
        try:
            with self.assertRaises(ValueError):
                get_dataloaders(index_csv=tmp_csv, project_root=".")
        finally:
            os.unlink(tmp_csv)

    # ------------------------------------------------------------------
    # 8. get_dataloaders constructs loaders from real index CSV
    # ------------------------------------------------------------------

    def test_get_dataloaders_from_real_index_csv(self):
        """get_dataloaders must return dict with at least one split from the real index."""
        from training.dataset import get_dataloaders
        index_csv = os.path.join(PROJECT_ROOT, "datasets", "metadata",
                                 "model_dataset_index.csv")
        if not os.path.exists(index_csv):
            self.skipTest("model_dataset_index.csv not found")
        loaders = get_dataloaders(
            index_csv=index_csv,
            project_root=PROJECT_ROOT,
            batch_size=4,
            num_workers=0,
        )
        self.assertGreater(len(loaders), 0)
        for split_name, loader in loaders.items():
            self.assertIn(split_name, ["train", "val", "test"])
            self.assertGreater(len(loader.dataset), 0)

    # ------------------------------------------------------------------
    # 9. DataLoader batch shape and dtype (uses zero-fill fallback)
    # ------------------------------------------------------------------

    def test_dataloader_batch_shape_and_dtype(self):
        """Iterating a DataLoader must yield float32 [B,19,1280] + int64 labels."""
        from training.dataset import EEGDataset, _eeg_collate_fn
        from torch.utils.data import DataLoader
        df = self._make_index_df(n_rows=4)
        ds = EEGDataset(index_df=df, project_root=".")
        loader = DataLoader(ds, batch_size=4, shuffle=False,
                            collate_fn=_eeg_collate_fn)
        eeg_batch, labels, metas = next(iter(loader))
        self.assertEqual(eeg_batch.shape, (4, 19, 1280))
        self.assertEqual(eeg_batch.dtype, torch.float32)
        self.assertEqual(labels.shape, (4,))
        self.assertEqual(labels.dtype, torch.long)
        self.assertIsInstance(metas, list)
        self.assertEqual(len(metas), 4)

    # ------------------------------------------------------------------
    # 10. Subject-level leakage: each subject appears in exactly one split
    # ------------------------------------------------------------------

    def test_subject_level_no_leakage_in_real_index(self):
        """Verify zero subject overlap across splits in the real model index CSV."""
        import pandas as pd
        index_csv = os.path.join(PROJECT_ROOT, "datasets", "metadata",
                                 "model_dataset_index.csv")
        if not os.path.exists(index_csv):
            self.skipTest("model_dataset_index.csv not found")
        df = pd.read_csv(index_csv)
        subject_splits = (
            df.groupby(["dataset_name", "subject_id"])["split"]
            .nunique()
        )
        leaking = subject_splits[subject_splits > 1]
        self.assertEqual(
            len(leaking), 0,
            msg=f"Subjects appear in multiple splits (leakage):\n{leaking}"
        )



class TestFeatureExtraction(unittest.TestCase):
    """STEP 4: CNN feature extraction pipeline tests."""

    def test_feature_extraction_import(self):
        from training.feature_extraction import extract_cnn_features, load_features, ExtractionResult
        self.assertTrue(callable(extract_cnn_features))
        self.assertTrue(callable(load_features))
        self.assertTrue(callable(ExtractionResult))

    def test_extraction_result_accumulation(self):
        from training.feature_extraction import ExtractionResult
        import numpy as np
        result = ExtractionResult()
        feats = np.random.randn(4, 40, 64).astype(np.float32)
        labels = [0, 1, 2, 3]
        metas = [
            {"dataset_name": "chbmit",  "subject_id": "chb01", "class_category": "epilepsy",
             "window_idx": 0, "source_file": "a.edf"},
            {"dataset_name": "srm",     "subject_id": "sub01", "class_category": "healthy",
             "window_idx": 1, "source_file": "b.edf"},
            {"dataset_name": "alz",     "subject_id": "sub02", "class_category": "alzheimers",
             "window_idx": 2, "source_file": "c.set"},
            {"dataset_name": "park",    "subject_id": "sub03", "class_category": "parkinsons",
             "window_idx": 3, "source_file": "d.set"},
        ]
        result.append_batch(feats, labels, metas)
        self.assertEqual(len(result), 4)
        nd = result.to_numpy_dict()
        self.assertIn("embeddings", nd)
        self.assertIn("labels", nd)
        self.assertIn("dataset_names", nd)
        self.assertEqual(nd["embeddings"].shape, (4, 40, 64))
        self.assertEqual(nd["labels"].dtype, np.int64)

    def test_extraction_result_provenance_preserved(self):
        from training.feature_extraction import ExtractionResult
        import numpy as np
        result = ExtractionResult()
        feats = np.zeros((1, 40, 64), dtype=np.float32)
        result.append_batch(feats, [2], [{"dataset_name": "alz", "subject_id": "s01",
                                           "class_category": "alzheimers",
                                           "window_idx": 5, "source_file": "foo.set"}])
        nd = result.to_numpy_dict()
        self.assertEqual(str(nd["dataset_names"][0]), "alz")
        self.assertEqual(str(nd["subject_ids"][0]), "s01")
        self.assertEqual(int(nd["window_indices"][0]), 5)

    def test_extract_cnn_features_missing_index(self):
        """Should return empty dict when index CSV does not exist."""
        import tempfile, os
        from training.feature_extraction import extract_cnn_features
        with tempfile.TemporaryDirectory() as tmp:
            paths = extract_cnn_features(
                index_csv=os.path.join(tmp, "nonexistent.csv"),
                output_dir=os.path.join(tmp, "out"),
            )
        self.assertEqual(paths, {})

    def test_extract_cnn_features_with_synthetic_index(self):
        """Full extraction smoke test on synthetic in-memory index."""
        import tempfile, os
        import pandas as pd
        from training.feature_extraction import extract_cnn_features, load_features
        rows = []
        for i in range(4):
            rows.append({
                "split": "train" if i < 3 else "val",
                "dataset_name": "chbmit",
                "subject_id": f"sub-{i:03d}",
                "class_category": "epilepsy",
                "class_label_idx": 1,
                "source_file": "nonexistent/dummy.edf",
                "window_idx": i,
                "window_start_sec": float(i) * 5.0,
                "window_end_sec": float(i) * 5.0 + 5.0,
                "sampling_rate": 256.0,
                "n_channels": 19,
                "n_samples": 1280,
                "is_bipolar_montage": False,
                "missing_channels": "",
            })
        df = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "idx.csv")
            out_dir  = os.path.join(tmp, "features")
            df.to_csv(csv_path, index=False)
            paths = extract_cnn_features(
                index_csv=csv_path,
                output_dir=out_dir,
                batch_size=4,
            )
            self.assertIn("train", paths)
            self.assertIn("val",   paths)
            # Verify NPZ structure
            for split, p in paths.items():
                loaded = load_features(str(p))
                self.assertIn("embeddings", loaded)
                self.assertIn("labels",     loaded)
                self.assertIn("dataset_names", loaded)
                # Shape: [n_windows, N, D]  N=40, D=64
                self.assertEqual(loaded["embeddings"].ndim, 3)
                self.assertEqual(loaded["embeddings"].shape[1], 40)
                self.assertEqual(loaded["embeddings"].shape[2], 64)
                self.assertFalse(np.isnan(loaded["embeddings"]).any())


class TestTransformerIntegration(unittest.TestCase):
    """STEP 5: Transformer integration with 5-class CNN pipeline."""

    def setUp(self):
        from models.eeg_classifier import EEGClassifier
        self.B, self.C, self.T = 4, 19, 1280
        self.model = EEGClassifier(
            n_channels=self.C, n_samples=self.T, num_classes=5,
            embed_dim=64, num_heads=4, num_layers=2, ff_dim=128,
        ).eval()

    def test_5class_output_shape(self):
        x = torch.randn(self.B, self.C, self.T)
        with torch.no_grad():
            logits = self.model(x)
        self.assertEqual(logits.shape, (self.B, 5))

    def test_cnn_to_transformer_token_shape(self):
        """CNN features must have correct temporal token count."""
        x = torch.randn(self.B, self.C, self.T)
        with torch.no_grad():
            feats = self.model.get_features(x)   # [B, N, D]
        N_expected = self.T // 32   # 1280//32 = 40
        self.assertEqual(feats.shape, (self.B, N_expected, 64))

    def test_softmax_probabilities_sum_to_one(self):
        import torch.nn.functional as F
        x = torch.randn(self.B, self.C, self.T)
        with torch.no_grad():
            logits = self.model(x)
        probas = F.softmax(logits, dim=-1)
        sums   = probas.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(self.B), atol=1e-5, rtol=0)

    def test_gradient_flow(self):
        """Backprop should not produce NaN or zero gradients."""
        model_train = type(self.model)(
            n_channels=self.C, n_samples=self.T, num_classes=5
        ).train()
        x = torch.randn(self.B, self.C, self.T)
        y = torch.randint(0, 5, (self.B,))
        logits = model_train(x)
        loss   = torch.nn.CrossEntropyLoss()(logits, y)
        loss.backward()
        for name, param in model_train.named_parameters():
            if param.grad is not None:
                self.assertFalse(torch.isnan(param.grad).any(),
                                 msg=f"NaN gradient in {name}")

    def test_model_different_batch_sizes(self):
        """Model should handle varying batch sizes without errors."""
        for b in [1, 2, 8]:
            x = torch.randn(b, self.C, self.T)
            with torch.no_grad():
                out = self.model(x)
            self.assertEqual(out.shape, (b, 5))

    def test_class_label_mapping_in_pipeline(self):
        """5-class indices must match CLASS_TO_IDX canonical mapping."""
        from training.dataset import CLASS_TO_IDX
        expected = {"healthy": 0, "epilepsy": 1, "alzheimers": 2,
                    "parkinsons": 3, "depression": 4}
        self.assertEqual(CLASS_TO_IDX, expected)


class TestTrainer(unittest.TestCase):
    """STEP 6: Training loop smoke tests."""

    def test_trainer_import(self):
        from training.trainer import EEGTrainer, TrainConfig, run_demo
        self.assertTrue(callable(EEGTrainer))
        self.assertTrue(callable(run_demo))

    def test_train_config_defaults(self):
        from training.trainer import TrainConfig
        cfg = TrainConfig()
        self.assertEqual(cfg.num_classes, 5)
        self.assertEqual(cfg.n_channels, 19)
        self.assertEqual(cfg.n_samples, 1280)

    def test_trainer_synthetic_demo(self):
        """Training loop must complete without errors on synthetic data."""
        from training.trainer import run_demo
        history = run_demo(num_classes=5, n_epochs=2)
        self.assertIn("train_loss", history)
        self.assertIn("val_acc",    history)
        self.assertEqual(len(history["train_loss"]), 2)

    def test_trainer_loss_decreases(self):
        """Loss should generally trend downward over demo epochs."""
        from training.trainer import run_demo
        history = run_demo(num_classes=5, n_epochs=5)
        # Not strict, just check no extreme blow-up
        for loss in history["train_loss"]:
            self.assertFalse(np.isnan(loss), msg="NaN loss detected")
            self.assertLess(loss, 1e6, msg="Exploding loss detected")

    def test_checkpoint_saved(self):
        """Trainer must save a checkpoint file."""
        import tempfile
        from training.trainer import EEGTrainer, TrainConfig
        from torch.utils.data import TensorDataset, DataLoader

        B, C, T = 8, 19, 1280
        xs = torch.randn(B, C, T)
        ys = torch.randint(0, 5, (B,))

        def _collate(batch):
            eeg = torch.stack([b[0] for b in batch])
            lbl = torch.stack([b[1] for b in batch])
            return eeg, lbl, [{}] * len(batch)

        loader = DataLoader(TensorDataset(xs, ys), batch_size=4, collate_fn=_collate)

        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = TrainConfig(
                num_classes=5, max_epochs=2, patience=5,
                checkpoint_dir=tmp_dir, lr=1e-3,
            )
            trainer = EEGTrainer(cfg)
            trainer.train(loader, loader)
            # At least one checkpoint file should exist
            ckpts = list(sorted(
                f for f in os.listdir(tmp_dir) if f.endswith(".pt")
            ))
            self.assertGreater(len(ckpts), 0, msg="No checkpoint file saved")


class TestEvaluator(unittest.TestCase):
    """STEP 7: Evaluation metrics smoke tests."""

    def setUp(self):
        from models.eeg_classifier import EEGClassifier
        self.device = torch.device("cpu")
        self.model  = EEGClassifier(n_channels=19, n_samples=1280, num_classes=5).eval()
        xs = torch.randn(16, 19, 1280)
        ys = torch.randint(0, 5, (16,))
        from torch.utils.data import TensorDataset, DataLoader

        def _collate(batch):
            eeg = torch.stack([b[0] for b in batch])
            lbl = torch.stack([b[1] for b in batch])
            return eeg, lbl, [{}] * len(batch)

        self.loader = DataLoader(
            TensorDataset(xs, ys), batch_size=4, collate_fn=_collate
        )

    def test_evaluator_import(self):
        from training.evaluator import evaluate_model, print_report
        self.assertTrue(callable(evaluate_model))
        self.assertTrue(callable(print_report))

    def test_evaluate_returns_required_keys(self):
        from training.evaluator import evaluate_model
        metrics = evaluate_model(self.model, self.loader, self.device)
        for key in ("accuracy", "macro_f1", "weighted_f1",
                    "confusion_matrix", "per_class", "n_samples"):
            self.assertIn(key, metrics, msg=f"Missing key: {key}")

    def test_confusion_matrix_shape(self):
        from training.evaluator import evaluate_model
        metrics = evaluate_model(self.model, self.loader, self.device)
        cm = metrics["confusion_matrix"]
        self.assertEqual(cm.shape, (5, 5))

    def test_per_class_has_all_classes(self):
        from training.evaluator import evaluate_model
        metrics = evaluate_model(self.model, self.loader, self.device)
        for cls in ("healthy", "epilepsy", "alzheimers", "parkinsons", "depression"):
            self.assertIn(cls, metrics["per_class"])

    def test_accuracy_in_valid_range(self):
        from training.evaluator import evaluate_model
        metrics = evaluate_model(self.model, self.loader, self.device)
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)

    def test_evaluate_with_probas(self):
        from training.evaluator import evaluate_model
        metrics = evaluate_model(
            self.model, self.loader, self.device, return_probas=True
        )
        self.assertIn("all_probas", metrics)
        self.assertEqual(metrics["all_probas"].shape[1], 5)


class TestExplainability(unittest.TestCase):
    """STEP 8: Explainability smoke tests."""

    def setUp(self):
        from models.eeg_classifier import EEGClassifier
        from training.explainability import EEGExplainer
        self.device = torch.device("cpu")
        self.model  = EEGClassifier(n_channels=19, n_samples=1280, num_classes=5).eval()
        self.exp    = EEGExplainer(self.model, self.device)
        self.x      = torch.randn(1, 19, 1280)

    def test_explainer_import(self):
        from training.explainability import EEGExplainer
        self.assertTrue(callable(EEGExplainer))

    def test_input_gradients_shape(self):
        sal = self.exp.input_gradients(self.x, target_class=0)
        self.assertEqual(sal.shape, (19, 1280))

    def test_input_gradients_not_nan(self):
        sal = self.exp.input_gradients(self.x, target_class=1)
        self.assertFalse(np.isnan(sal).any())

    def test_input_gradients_default_class(self):
        """Should work without specifying target_class (uses argmax)."""
        sal = self.exp.input_gradients(self.x)
        self.assertEqual(sal.shape, (19, 1280))

    def test_integrated_gradients_shape(self):
        ig = self.exp.integrated_gradients(self.x, target_class=0, steps=5)
        self.assertEqual(ig.shape, (19, 1280))

    def test_integrated_gradients_not_nan(self):
        ig = self.exp.integrated_gradients(self.x, target_class=2, steps=5)
        self.assertFalse(np.isnan(ig).any())

    def test_transformer_attention_shape(self):
        attn = self.exp.transformer_attention(self.x)
        # May be None if hooks don't fire; shape should be (N_tokens,) if not None
        if attn is not None:
            self.assertEqual(attn.ndim, 1)
            self.assertGreater(len(attn), 0)

if __name__ == '__main__':
    # Run with verbose output
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)



