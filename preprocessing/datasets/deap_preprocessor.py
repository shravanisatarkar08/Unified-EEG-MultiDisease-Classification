"""
DEAP Dataset Preprocessor.

Maps to: 'depression' class (as proxy — see scientific note)

Dataset properties:
- 32 healthy subjects
- 512 Hz (raw .bdf) or 128 Hz (preprocessed .dat)
- 32 EEG + 8 peripheral channels
- 40 emotion trials per subject, 63 seconds each
- Labels: Valence, Arousal, Dominance, Liking (1-9 scale)

Scientific Note:
    DEAP is an emotion dataset, NOT a clinical depression dataset.
    We use it as a 'depression-proxy' by selecting:
        - Low valence (valence <= 4) AND low arousal (arousal <= 4) trials
        - These correspond to dysphoric/withdrawn affective states
        - This is a methodological limitation; explicitly documented.

File format supported: .dat (Python pickle, preprocessed 128 Hz version)
    Each .dat file: subject data dict with keys:
        'data': shape [40, 40, 8064]  — [trials, channels, samples]
        'labels': shape [40, 4]       — [valence, arousal, dominance, liking]
"""

import logging
import os
import pickle
import warnings
from pathlib import Path
from typing import Optional, List

import numpy as np

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.dataset_preprocessors import (
    BaseDatasetPreprocessor,
    DatasetPreprocessorConfig,
    PreprocessedEEGSignal,
)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# DEAP preprocessed .dat: 32 EEG channels (10-20 system)
# First 32 channels are EEG, channels 33-40 are peripheral
DEAP_EEG_CHANNELS = 32
DEAP_SRATE = 128  # Hz (preprocessed version)

# Standard 10-20 subset matching project's 19 channels
# DEAP channel order (from documentation):
DEAP_CH_NAMES = [
    'Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7',
    'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz',
    'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz',
    'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2',
]

# 19-channel subset that exists in DEAP and matches project standard
# (T7→T3, T8→T4, P7→T5, P8→T6 aliases applied)
PROJECT_19CH_FROM_DEAP = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T7',  'C3',  'Cz', 'C4', 'T8',
    'P7',  'P3',  'Pz', 'P4', 'P8',
    'O1',  'O2',
]

ALIAS_MAP = {'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6'}

# Low valence + low arousal threshold for depression-proxy selection
DEPRESSION_PROXY_THRESHOLD = 4.5  # <= this = low


class DEAPPreprocessor(BaseDatasetPreprocessor):
    """
    DEAP Dataset Preprocessor — Depression-proxy class.

    Reads preprocessed .dat files (128 Hz, 32 EEG channels).
    Selects trials with low valence AND low arousal as depression proxy.
    Picks 19 standard channels, resamples to 256 Hz.
    """

    def __init__(self):
        config = DatasetPreprocessorConfig(
            dataset_name='deap',
            target_srate_hz=256.0,  # Resample from 128 Hz
            l_freq=0.5,
            h_freq=45.0,
            notch_freqs=[50.0],     # European datasets use 50 Hz
            rereference_mode='average',
        )
        super().__init__(config)

    def _resample_array(self, data: np.ndarray, orig_sr: float, target_sr: float) -> np.ndarray:
        """Simple resample using scipy."""
        try:
            from scipy.signal import resample_poly
            from math import gcd
            orig_sr_i = int(orig_sr)
            target_sr_i = int(target_sr)
            g = gcd(orig_sr_i, target_sr_i)
            up = target_sr_i // g
            down = orig_sr_i // g
            return resample_poly(data, up, down, axis=-1).astype(np.float32)
        except ImportError:
            # Fall back to numpy-based resampling
            n_samples_out = int(data.shape[-1] * target_sr / orig_sr)
            from numpy import interp
            x_old = np.linspace(0, 1, data.shape[-1])
            x_new = np.linspace(0, 1, n_samples_out)
            return np.array([interp(x_new, x_old, row) for row in data], dtype=np.float32)

    def preprocess_file(
        self,
        file_path: str,
        subject_id: Optional[str] = None,
    ) -> Optional[PreprocessedEEGSignal]:
        """
        Load a DEAP .dat file and return concatenated depression-proxy EEG.

        Returns a single PreprocessedEEGSignal with all selected trials
        concatenated along the time axis.
        """
        fp = Path(file_path)
        if not fp.exists():
            logger.error(f"[DEAP] File not found: {file_path}")
            return None

        if subject_id is None:
            import re
            m = re.search(r's(\d+)', fp.stem, re.IGNORECASE)
            subject_id = f"s{int(m.group(1)):02d}" if m else fp.stem

        try:
            with open(fp, 'rb') as f:
                data_dict = pickle.load(f, encoding='latin1')
        except Exception as e:
            logger.error(f"[DEAP] Could not load {file_path}: {e}")
            return None

        eeg_data = data_dict.get('data')    # [40, 40, 8064]
        labels   = data_dict.get('labels')  # [40, 4]

        if eeg_data is None or labels is None:
            logger.error(f"[DEAP] Missing 'data' or 'labels' in {file_path}")
            return None

        # Use only EEG channels (first 32 of 40)
        eeg_data = eeg_data[:, :DEAP_EEG_CHANNELS, :]  # [40, 32, 8064]

        # Select depression-proxy trials: low valence AND low arousal
        valence  = labels[:, 0]
        arousal  = labels[:, 1]
        dep_mask = (valence <= DEPRESSION_PROXY_THRESHOLD) & (arousal <= DEPRESSION_PROXY_THRESHOLD)
        n_dep = dep_mask.sum()

        if n_dep == 0:
            logger.warning(f"[DEAP] Subject {subject_id}: no low-VA trials found.")
            return None

        selected = eeg_data[dep_mask]  # [n_dep, 32, 8064]
        logger.info(f"[DEAP] Subject {subject_id}: {n_dep}/40 depression-proxy trials selected")

        # Pick 19 standard channels from DEAP's 32
        ch_idx = [DEAP_CH_NAMES.index(ch) for ch in PROJECT_19CH_FROM_DEAP if ch in DEAP_CH_NAMES]
        selected = selected[:, ch_idx, :]  # [n_dep, 19, 8064]
        ch_names_19 = [DEAP_CH_NAMES[i] for i in ch_idx]

        # Concatenate all selected trials along time axis
        concatenated = np.concatenate(list(selected), axis=-1).astype(np.float32)  # [19, n_dep*8064]

        # Bandpass filter (manual, since we're not using MNE Raw here)
        try:
            from scipy.signal import butter, sosfiltfilt
            sos = butter(5, [0.5, 45.0], btype='bandpass', fs=DEAP_SRATE, output='sos')
            concatenated = sosfiltfilt(sos, concatenated, axis=-1).astype(np.float32)
        except Exception as e:
            logger.warning(f"[DEAP] Bandpass filter failed: {e}")

        # Average reference
        concatenated -= concatenated.mean(axis=0, keepdims=True)

        # Resample from 128 Hz → 256 Hz
        if abs(DEAP_SRATE - 256.0) > 1:
            concatenated = self._resample_array(concatenated, DEAP_SRATE, 256.0)

        # Rename T7/T8/P7/P8 → T3/T4/T5/T6
        ch_names_final = [ALIAS_MAP.get(ch, ch) for ch in ch_names_19]

        n_channels, n_samples = concatenated.shape

        return PreprocessedEEGSignal(
            dataset_name='deap',
            class_category='depression',
            subject_id=str(subject_id),
            file_path=str(file_path).replace("\\", "/"),
            srate_hz=256.0,
            channel_names=ch_names_final,
            n_channels=n_channels,
            n_samples=n_samples,
            duration_sec=float(n_samples) / 256.0,
            data=concatenated,
            seizure_events=[],
            metadata={
                'depression_proxy': True,
                'n_dep_trials': int(n_dep),
                'deap_valence_threshold': DEPRESSION_PROXY_THRESHOLD,
                'deap_arousal_threshold': DEPRESSION_PROXY_THRESHOLD,
                'original_srate_hz': DEAP_SRATE,
                'original_n_trials': 40,
                'has_nan_or_inf': bool(np.isnan(concatenated).any() or np.isinf(concatenated).any()),
                'flat_channels': [],
                'rereferenced': True,
                'native_srate_hz': DEAP_SRATE,
                'mean_signal_std': float(np.std(concatenated)),
            }
        )
