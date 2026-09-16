"""
Common EEG Harmonization and Windowing Layer.

Converts dataset-specifically preprocessed EEG signals into standardized,
model-ready 5-second EEG windows (19 channels x 1280 samples @ 256 Hz) with
comprehensive window metadata and subject-identity tracking.

MONTAGE & CHANNEL HARMONIZATION POLICY:
----------------------------------------
1. Monopolar Scalp Datasets (SRM, Alzheimer's ds004504, Parkinson's ds004584):
   - Standard 19 10-20 locations: ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
                                    'T3', 'C3', 'Cz', 'C4', 'T4',
                                    'T5', 'P3', 'Pz', 'P4', 'T6',
                                    'O1', 'O2']
   - Standard 10-10 equivalences mapped: T7<->T3, T8<->T4, P7<->T5, P8<->T6.
   - Missing Channel Policy ('zero_fill'):
     If an electrode location is missing in a recording (e.g., Pz in ds004584),
     the channel row is zero-filled (0.0), recorded in `missing_channels` metadata,
     and flagged in a 19-element `channel_mask` (1.0 = real, 0.0 = missing).
     NO signal values are fabricated.

2. Bipolar Dataset (CHB-MIT Epilepsy):
   - Recorded in modified longitudinal bipolar differential pairs V(A)-V(B).
   - SCIENTIFIC LIMITATION & POLICY:
     Differential potentials V(A)-V(B) are mathematically distinct from monopolar
     scalp potentials V(A). Forcing bipolar channels into monopolar electrode signals
     introduces arbitrary spatial reference artifacts.
     Therefore, CHB-MIT is harmonized into its 19 standard bipolar pair channels:
     ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
      'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
      'FZ-CZ', 'CZ-PZ', 'P7-T7']
     and explicitly flagged with `is_bipolar_montage = True`.

3. Resampling & Windowing:
   - Resampled to 256 Hz.
   - Window duration = 5.0 seconds -> exactly 1280 time samples.
   - Overlap = 50% (step = 2.5 seconds = 640 samples).
   - Preserves subject_id for strict subject-level train/test splitting.
   - NO global cross-dataset Z-score normalization leakage.
"""

import os
import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import scipy.signal

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.dataset_preprocessors import PreprocessedEEGSignal, run_controlled_sample_preprocessing

logger = logging.getLogger(__name__)

# Target 19 standard 10-20 monopolar channels
TARGET_19_MONOPOLAR = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T3', 'C3', 'Cz', 'C4', 'T4',
    'T5', 'P3', 'Pz', 'P4', 'T6',
    'O1', 'O2'
]

# Standard 10-10 to 10-20 channel equivalences
CHANNEL_ALIASES_10_10 = {
    'T7': 'T3',
    'T8': 'T4',
    'P7': 'T5',
    'P8': 'T6',
    'FP1': 'Fp1',
    'FP2': 'Fp2',
    'FZ': 'Fz',
    'CZ': 'Cz',
    'PZ': 'Pz',
}

# Target 19 standard CHB-MIT bipolar channels
TARGET_19_BIPOLAR = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ', 'CZ-PZ',
    'P7-T7'
]


# -----------------------------------------------------------------------------
# DATA STRUCTURES
# -----------------------------------------------------------------------------

@dataclass
class HarmonizedWindow:
    """Standardized 5-second model-ready EEG window with full provenance metadata."""
    dataset_name: str
    class_category: str
    subject_id: str
    source_file: str
    window_idx: int
    window_start_sec: float
    window_end_sec: float
    sampling_rate: float            # 256.0 Hz
    channel_names: List[str]         # list of 19 channel names
    n_channels: int                 # 19
    n_samples: int                  # 1280
    data: np.ndarray                # shape: [19, 1280], dtype: float32
    channel_mask: np.ndarray        # shape: [19], dtype: float32 (1.0 = real, 0.0 = missing)
    is_bipolar_montage: bool        # True for CHB-MIT, False for monopolar scalp
    missing_channels: List[str]     # List of missing target channels
    seizure_events: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class HarmonizationConfig:
    """Configurable harmonization parameters."""
    target_srate_hz: float = 256.0
    window_sec: float = 5.0
    overlap: float = 0.5              # 50% overlap (2.5s step)
    missing_channel_policy: str = 'zero_fill'  # 'zero_fill', 'subset_only', 'error'


# -----------------------------------------------------------------------------
# HARMONIZER CLASS
# -----------------------------------------------------------------------------

class EEGHarmonizer:
    """Harmonizes preprocessed dataset signals to 256 Hz, 19 channels, and 5s windows."""

    def __init__(self, config: Optional[HarmonizationConfig] = None):
        self.config = config or HarmonizationConfig()
        self.window_samples = int(self.config.window_sec * self.config.target_srate_hz)  # 1280
        self.step_samples = int(self.window_samples * (1.0 - self.config.overlap))        # 640

    def _resample_data(self, data: np.ndarray, orig_srate: float) -> np.ndarray:
        """Resample 2D array [n_channels, n_samples] to target_srate_hz."""
        target_srate = self.config.target_srate_hz
        if abs(orig_srate - target_srate) < 1e-3:
            return data.astype(np.float32)

        # Polyphase resampling for exact integer ratio precision
        orig_srate_int = int(round(orig_srate))
        target_srate_int = int(round(target_srate))
        gcd = np.gcd(orig_srate_int, target_srate_int)
        up = target_srate_int // gcd
        down = orig_srate_int // gcd

        resampled = scipy.signal.resample_poly(data, up, down, axis=1)
        return resampled.astype(np.float32)

    def harmonize_signal(
        self,
        signal: PreprocessedEEGSignal
    ) -> Tuple[np.ndarray, List[str], np.ndarray, bool, List[str]]:
        """Harmonizes channels and resampling to 256 Hz.

        Returns:
            harmonized_data: [19, n_samples_256hz] float32
            channel_names: List[str] of 19 channel names
            channel_mask: [19] float32 (1.0 = present, 0.0 = missing)
            is_bipolar: bool
            missing_channels: List[str]
        """
        # 1. Resample to 256 Hz
        resampled_data = self._resample_data(signal.data, signal.srate_hz)
        n_samples = resampled_data.shape[1]

        # 2. Check if dataset is bipolar (CHB-MIT)
        is_bipolar = (signal.dataset_name == 'chbmit' or any('-' in ch for ch in signal.channel_names[:5]))

        if is_bipolar:
            target_channels = TARGET_19_BIPOLAR
            # Map bipolar channels
            src_map = {re.sub(r'-(\d+)$', '', ch).upper(): i for i, ch in enumerate(signal.channel_names)}
            harmonized = np.zeros((19, n_samples), dtype=np.float32)
            mask = np.zeros(19, dtype=np.float32)
            missing = []

            for idx, tch in enumerate(target_channels):
                tch_clean = re.sub(r'-(\d+)$', '', tch).upper()
                if tch_clean in src_map:
                    harmonized[idx] = resampled_data[src_map[tch_clean]]
                    mask[idx] = 1.0
                else:
                    missing.append(tch)

            return harmonized, target_channels, mask, True, missing

        else:
            # Monopolar scalp EEG dataset (SRM, Alzheimer's, Parkinson's)
            target_channels = TARGET_19_MONOPOLAR
            # Build case-insensitive lookup map with 10-10 aliases
            src_map = {}
            for i, ch in enumerate(signal.channel_names):
                clean_ch = ch.strip()
                src_map[clean_ch.upper()] = i
                # Check alias mapping (e.g. T7 -> T3)
                if clean_ch in CHANNEL_ALIASES_10_10:
                    alias_std = CHANNEL_ALIASES_10_10[clean_ch]
                    if alias_std.upper() not in src_map:
                        src_map[alias_std.upper()] = i

            harmonized = np.zeros((19, n_samples), dtype=np.float32)
            mask = np.zeros(19, dtype=np.float32)
            missing = []

            for idx, tch in enumerate(target_channels):
                tch_upper = tch.upper()
                if tch_upper in src_map:
                    harmonized[idx] = resampled_data[src_map[tch_upper]]
                    mask[idx] = 1.0
                else:
                    missing.append(tch)

            if missing:
                logger.info(f"[{signal.dataset_name}] Subject '{signal.subject_id}': {len(missing)} missing target channels {missing} -> Zero-filled with channel_mask=0.0")

            return harmonized, target_channels, mask, False, missing

    def segment_into_windows(self, signal: PreprocessedEEGSignal) -> List[HarmonizedWindow]:
        """Harmonizes signal and splits into 5-second overlapping windows [19, 1280]."""
        harmonized_data, channel_names, mask, is_bipolar, missing = self.harmonize_signal(signal)
        n_samples = harmonized_data.shape[1]

        if n_samples < self.window_samples:
            logger.warning(f"[{signal.dataset_name}] Recording '{signal.file_path}' too short ({n_samples} < {self.window_samples} samples). Skipping.")
            return []

        windows: List[HarmonizedWindow] = []
        start_idx = 0
        w_idx = 0

        while start_idx + self.window_samples <= n_samples:
            end_idx = start_idx + self.window_samples
            win_data = harmonized_data[:, start_idx:end_idx].copy()

            w_start_sec = start_idx / self.config.target_srate_hz
            w_end_sec = end_idx / self.config.target_srate_hz

            win_obj = HarmonizedWindow(
                dataset_name=signal.dataset_name,
                class_category=signal.class_category,
                subject_id=signal.subject_id,
                source_file=signal.file_path,
                window_idx=w_idx,
                window_start_sec=w_start_sec,
                window_end_sec=w_end_sec,
                sampling_rate=self.config.target_srate_hz,
                channel_names=channel_names,
                n_channels=19,
                n_samples=self.window_samples,
                data=win_data,
                channel_mask=mask.copy(),
                is_bipolar_montage=is_bipolar,
                missing_channels=missing,
                seizure_events=signal.seizure_events
            )
            windows.append(win_obj)

            start_idx += self.step_samples
            w_idx += 1

        return windows


# -----------------------------------------------------------------------------
# CONTROLLED SAMPLE HARMONIZATION RUNNER
# -----------------------------------------------------------------------------

def run_controlled_sample_harmonization() -> List[HarmonizedWindow]:
    """Runs preprocessing and harmonization on a SMALL controlled sample:

    - 1 CHB-MIT recording (epilepsy)
    - 1 SRM Healthy recording (healthy)
    - 1 Alzheimer's recording (ds004504)
    - 1 Parkinson's recording (ds004584)
    """
    signals = run_controlled_sample_preprocessing()
    harmonizer = EEGHarmonizer()
    all_sample_windows: List[HarmonizedWindow] = []

    for sig in signals:
        wins = harmonizer.segment_into_windows(sig)
        if wins:
            # Keep first 5 windows per sample to avoid memory load
            all_sample_windows.extend(wins[:5])
            print(f"Dataset: {sig.dataset_name:<12} | Subject: {sig.subject_id:<8} | Category: {sig.class_category:<10} | "
                  f"Total Windows: {len(wins):4d} | Window Shape: {wins[0].data.shape} | Bipolar: {wins[0].is_bipolar_montage!s:<5} | "
                  f"Missing Ch: {wins[0].missing_channels}")

    return all_sample_windows


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("\n==================================================================================")
    print("                CONTROLLED SAMPLE EEG HARMONIZATION & WINDOWING REPORT            ")
    print("==================================================================================")
    run_controlled_sample_harmonization()
    print("==================================================================================\n")
