"""
Dataset-Specific EEG Preprocessing Layer.

Provides dataset-specific preprocessing adapters for:
- CHB-MIT Epilepsy Dataset (EDF, 256 Hz, 23-ch bipolar montage, seizure/interictal annotations)
- SRM Healthy Control Dataset (EDF, 1024 Hz, 64-ch scalp EEG, average reference)
- OpenNeuro ds004504 Alzheimer's Dataset (SET, 500 Hz, 19-ch scalp EEG, average reference; FTD EXCLUDED)
- OpenNeuro ds004584 Parkinson's Dataset (SET, 500 Hz, 63/64-ch scalp EEG, average reference)
- MODMA Depression Dataset (Adapter stub; raw EEG files unextracted)
- TUAB Dataset (Adapter stub; dataset restricted / unavailable)

Each adapter converts raw EEG into a clean dataset-specific PreprocessedEEGSignal
without performing premature cross-dataset channel force-renaming or global Z-score leakage.
"""

import os
import re
import glob
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import mne
import numpy as np
import pandas as pd

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Suppress verbose MNE runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")
warnings.filterwarnings("ignore", category=UserWarning, module="mne")

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# DATA STRUCTURES
# -----------------------------------------------------------------------------

@dataclass
class PreprocessedEEGSignal:
    """Clean, dataset-specifically preprocessed EEG signal container."""
    dataset_name: str
    class_category: str
    subject_id: str
    file_path: str
    srate_hz: float
    channel_names: List[str]
    n_channels: int
    n_samples: int
    duration_sec: float
    data: np.ndarray  # shape: [n_channels, n_samples], dtype: float32
    seizure_events: List[Tuple[float, float]] = field(default_factory=list)  # [(start_sec, end_sec)]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetPreprocessorConfig:
    """Configuration parameters for dataset-specific preprocessing."""
    dataset_name: str
    target_srate_hz: Optional[float] = None  # None = retain native sampling rate
    l_freq: float = 0.5                      # Highpass filter cutoff (Hz)
    h_freq: float = 45.0                     # Lowpass filter cutoff (Hz)
    notch_freqs: List[float] = field(default_factory=lambda: [50.0, 60.0])  # Line noise frequencies
    rereference_mode: str = 'average'        # 'average', 'bipolar', or 'none'
    check_signal_quality: bool = True
    flat_channel_std_threshold: float = 1e-9 # Threshold to flag flat channels


# -----------------------------------------------------------------------------
# BASE PREPROCESSOR ADAPTER
# -----------------------------------------------------------------------------

class BaseDatasetPreprocessor:
    """Base class for dataset-specific EEG preprocessor adapters."""

    def __init__(self, config: DatasetPreprocessorConfig):
        self.config = config

    def preprocess_raw_object(
        self,
        raw: mne.io.Raw,
        dataset_name: str,
        class_category: str,
        subject_id: str,
        file_path: str,
        seizure_events: Optional[List[Tuple[float, float]]] = None
    ) -> PreprocessedEEGSignal:
        """Apply dataset-specific notch filtering, bandpass filtering, re-referencing,

        resampling (if configured), and signal quality checks to an MNE Raw instance.
        """
        # Load raw data into memory if not loaded
        raw = raw.copy()
        if not raw.preload:
            raw.load_data()

        srate = float(raw.info['sfreq'])

        # 1. Notch filtering for powerline noise (only for frequencies below Nyquist)
        valid_notch = [f for f in self.config.notch_freqs if f < srate / 2.0]
        if valid_notch:
            raw.notch_filter(valid_notch, verbose=False)

        # 2. Bandpass filtering (0.5 - 45.0 Hz FIR filter)
        raw.filter(self.config.l_freq, self.config.h_freq, fir_design='firwin', verbose=False)

        # 3. Re-referencing (average reference for scalp EEG montages with >= 4 channels)
        rereferenced = False
        if self.config.rereference_mode == 'average' and len(raw.ch_names) >= 4:
            try:
                raw.set_eeg_reference('average', projection=False, verbose=False)
                rereferenced = True
            except Exception as e:
                logger.warning(f"[{dataset_name}] Average re-referencing warning: {e}")

        # 4. Optional resampling (if target_srate_hz specified and differs from native srate)
        if self.config.target_srate_hz and abs(srate - self.config.target_srate_hz) > 1e-3:
            raw.resample(self.config.target_srate_hz, verbose=False)
            srate = float(raw.info['sfreq'])

        # 5. Extract signal array
        data = raw.get_data().astype(np.float32)  # [n_channels, n_samples]
        n_channels, n_samples = data.shape
        duration_sec = float(raw.times[-1]) if len(raw.times) > 0 else 0.0

        # 6. Basic signal quality metrics & non-destructive warnings
        has_nan_or_inf = bool(np.isnan(data).any() or np.isinf(data).any())
        channel_stds = np.std(data, axis=1).tolist()
        flat_channels = [
            raw.ch_names[i] for i, std in enumerate(channel_stds)
            if std < self.config.flat_channel_std_threshold
        ]

        quality_metrics = {
            'has_nan_or_inf': has_nan_or_inf,
            'flat_channels': flat_channels,
            'rereferenced': rereferenced,
            'native_srate_hz': float(raw.info['sfreq']),
            'mean_signal_std': float(np.mean(channel_stds)) if channel_stds else 0.0
        }

        if has_nan_or_inf:
            logger.warning(f"[{dataset_name}] Signal in '{file_path}' contains NaN or Inf values!")

        return PreprocessedEEGSignal(
            dataset_name=dataset_name,
            class_category=class_category,
            subject_id=subject_id,
            file_path=str(file_path).replace("\\", "/"),
            srate_hz=srate,
            channel_names=list(raw.ch_names),
            n_channels=n_channels,
            n_samples=n_samples,
            duration_sec=duration_sec,
            data=data,
            seizure_events=seizure_events or [],
            metadata=quality_metrics
        )


# -----------------------------------------------------------------------------
# DATASET-SPECIFIC PREPROCESSOR ADAPTERS
# -----------------------------------------------------------------------------

class CHBMITPreprocessor(BaseDatasetPreprocessor):
    """CHB-MIT Epilepsy EEG Preprocessor Adapter.

    Native Properties:
    - 256 Hz native sampling rate
    - 23 channels (bipolar differential montage e.g., FP1-F7, F7-T7)
    - Re-referencing: 'none' (already differential bipolar montage)
    - Preserves seizure event start/end annotations from summary text files.
    - Class category: 'epilepsy' (interictal or ictal). Does NOT fabricate seizure labels.
    """

    def __init__(self):
        config = DatasetPreprocessorConfig(
            dataset_name='chbmit',
            target_srate_hz=None,  # Retain native 256 Hz
            l_freq=0.5,
            h_freq=45.0,
            notch_freqs=[60.0],    # US 60 Hz powerline line noise
            rereference_mode='none'
        )
        super().__init__(config)

    def parse_summary(self, summary_path: str) -> Dict[str, List[Tuple[float, float]]]:
        """Parse CHB-MIT summary text file for exact seizure start/end timestamps."""
        seizures = {}
        current_file = None
        if not os.path.exists(summary_path):
            return seizures

        with open(summary_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                m_file = re.match(r'File Name:\s*(\S+)', line)
                if m_file:
                    current_file = m_file.group(1)
                    if current_file not in seizures:
                        seizures[current_file] = []
                    continue

                m_start = re.match(r'Seizure\s*\d*\s*Start Time:\s*(\d+)\s*seconds?', line, re.IGNORECASE)
                if m_start and current_file:
                    start_sec = float(m_start.group(1))
                    continue

                m_end = re.match(r'Seizure\s*\d*\s*End Time:\s*(\d+)\s*seconds?', line, re.IGNORECASE)
                if m_end and current_file:
                    end_sec = float(m_end.group(1))
                    seizures[current_file].append((start_sec, end_sec))

        return seizures

    def preprocess_file(self, file_path: str, subject_id: Optional[str] = None) -> Optional[PreprocessedEEGSignal]:
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            logger.error(f"[CHB-MIT] File not found: {file_path}")
            return None

        if subject_id is None:
            # Extract subject ID (e.g., chb01 from chb01_01.edf)
            match = re.search(r'(chb\d+)', file_path_obj.name, re.IGNORECASE)
            subject_id = match.group(1) if match else "chb01"

        # Check for summary file annotations
        parent_dir = file_path_obj.parent
        summary_file = parent_dir / f"{subject_id}-summary.txt"
        seizures_dict = self.parse_summary(str(summary_file))
        seizure_events = seizures_dict.get(file_path_obj.name, [])

        try:
            raw = mne.io.read_raw_edf(str(file_path_obj), preload=False, verbose=False)
            return self.preprocess_raw_object(
                raw=raw,
                dataset_name='chbmit',
                class_category='epilepsy',
                subject_id=subject_id,
                file_path=str(file_path_obj),
                seizure_events=seizure_events
            )
        except Exception as e:
            logger.error(f"[CHB-MIT] Preprocessing failed for '{file_path}': {e}")
            return None


class SRMHealthyPreprocessor(BaseDatasetPreprocessor):
    """SRM Healthy Control EEG Preprocessor Adapter (ds003775).

    Native Properties:
    - 1024 Hz native sampling rate
    - 64 channels (scalp 10-10/10-20 EEG)
    - Re-referencing: 'average' reference across scalp channels
    - Class category: 'healthy'
    """

    def __init__(self):
        config = DatasetPreprocessorConfig(
            dataset_name='srm_healthy',
            target_srate_hz=None,  # Retain native 1024 Hz
            l_freq=0.5,
            h_freq=45.0,
            notch_freqs=[50.0, 60.0],
            rereference_mode='average'
        )
        super().__init__(config)

    def preprocess_file(self, file_path: str, subject_id: Optional[str] = "sub-010") -> Optional[PreprocessedEEGSignal]:
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            logger.error(f"[SRM Healthy] File not found: {file_path}")
            return None

        try:
            raw = mne.io.read_raw_edf(str(file_path_obj), preload=False, verbose=False)
            return self.preprocess_raw_object(
                raw=raw,
                dataset_name='srm_healthy',
                class_category='healthy',
                subject_id=subject_id or "sub-010",
                file_path=str(file_path_obj)
            )
        except Exception as e:
            logger.error(f"[SRM Healthy] Preprocessing failed for '{file_path}': {e}")
            return None


class AlzheimersPreprocessor(BaseDatasetPreprocessor):
    """OpenNeuro ds004504 Alzheimer's EEG Preprocessor Adapter.

    Native Properties:
    - 500 Hz native sampling rate
    - 19 standard 10-20 channels
    - Re-referencing: 'average' reference
    - FTD EXCLUSION RULE: Group 'F' (Frontotemporal Dementia) subjects are explicitly EXCLUDED.
      Group 'A' -> 'alzheimers', Group 'C' -> 'healthy'.
    """

    def __init__(self):
        config = DatasetPreprocessorConfig(
            dataset_name='alzheimers',
            target_srate_hz=None,  # Retain native 500 Hz
            l_freq=0.5,
            h_freq=45.0,
            notch_freqs=[50.0],
            rereference_mode='average'
        )
        super().__init__(config)

    def preprocess_file(
        self,
        file_path: str,
        subject_id: Optional[str] = None,
        group_code: Optional[str] = None
    ) -> Optional[PreprocessedEEGSignal]:
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            logger.error(f"[Alzheimer's] File not found: {file_path}")
            return None

        if subject_id is None:
            match = re.search(r'(sub-\d+)', str(file_path_obj))
            subject_id = match.group(1) if match else "unknown"

        # Explicit FTD Exclusion Check
        if group_code == 'F' or group_code == 'ftd':
            logger.info(f"[Alzheimer's] Subject '{subject_id}' is FTD (Group 'F') -> EXCLUDED per project spec.")
            return None

        class_category = 'alzheimers' if group_code == 'A' else ('healthy' if group_code == 'C' else 'alzheimers')

        try:
            raw = mne.io.read_raw_eeglab(str(file_path_obj), preload=False, verbose=False)
            return self.preprocess_raw_object(
                raw=raw,
                dataset_name='alzheimers',
                class_category=class_category,
                subject_id=subject_id,
                file_path=str(file_path_obj)
            )
        except Exception as e:
            logger.error(f"[Alzheimer's] Preprocessing failed for '{file_path}': {e}")
            return None


class ParkinsonsPreprocessor(BaseDatasetPreprocessor):
    """OpenNeuro ds004584 Parkinson's EEG Preprocessor Adapter.

    Native Properties:
    - 500 Hz native sampling rate
    - 63/64 scalp EEG channels
    - Re-referencing: 'average' reference
    - Group 'PD' -> 'parkinsons', Group 'Control' / 'HC' -> 'healthy'.
    """

    def __init__(self):
        config = DatasetPreprocessorConfig(
            dataset_name='parkinsons',
            target_srate_hz=None,  # Retain native 500 Hz
            l_freq=0.5,
            h_freq=45.0,
            notch_freqs=[50.0],
            rereference_mode='average'
        )
        super().__init__(config)

    def preprocess_file(
        self,
        file_path: str,
        subject_id: Optional[str] = None,
        group_code: Optional[str] = None
    ) -> Optional[PreprocessedEEGSignal]:
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            logger.error(f"[Parkinson's] File not found: {file_path}")
            return None

        if subject_id is None:
            match = re.search(r'(sub-\d+)', str(file_path_obj))
            subject_id = match.group(1) if match else "unknown"

        class_category = 'healthy' if group_code in ['Control', 'HC'] else 'parkinsons'

        try:
            raw = mne.io.read_raw_eeglab(str(file_path_obj), preload=False, verbose=False)
            return self.preprocess_raw_object(
                raw=raw,
                dataset_name='parkinsons',
                class_category=class_category,
                subject_id=subject_id,
                file_path=str(file_path_obj)
            )
        except Exception as e:
            logger.error(f"[Parkinson's] Preprocessing failed for '{file_path}': {e}")
            return None


class MODMAPreprocessor(BaseDatasetPreprocessor):
    """MODMA Depression EEG Preprocessor Stub.

    Status: Raw EEG files not locally extracted (split archives present).
    Reports NOT AVAILABLE LOCALLY.
    """

    def __init__(self):
        config = DatasetPreprocessorConfig(dataset_name='modma')
        super().__init__(config)

    def preprocess_file(self, file_path: str, subject_id: Optional[str] = None) -> Optional[PreprocessedEEGSignal]:
        logger.warning("[MODMA] Raw EEG files not extracted locally. Preprocessing unavailable.")
        return None


class TUABPreprocessor(BaseDatasetPreprocessor):
    """TUAB Dataset Preprocessor Stub.

    Status: Dataset restricted / unavailable locally.
    Reports NOT AVAILABLE LOCALLY.
    """

    def __init__(self):
        config = DatasetPreprocessorConfig(dataset_name='tuab')
        super().__init__(config)

    def preprocess_file(self, file_path: str, subject_id: Optional[str] = None) -> Optional[PreprocessedEEGSignal]:
        logger.warning("[TUAB] Dataset restricted / not available locally.")
        return None


# -----------------------------------------------------------------------------
# CONTROLLED SAMPLE RUNNER FOR DEMONSTRATION
# -----------------------------------------------------------------------------

def run_controlled_sample_preprocessing() -> List[PreprocessedEEGSignal]:
    """Runs dataset-specific preprocessing on a SMALL controlled sample:

    - 1 CHB-MIT recording (epilepsy)
    - 1 SRM Healthy recording (healthy)
    - 1 Alzheimer's recording (ds004504)
    - 1 Parkinson's recording (ds004584)

    Returns lightweight PreprocessedEEGSignal list without creating a data dump.
    """
    results = []

    # 1. CHB-MIT sample
    chb_file = "datasets/raw/epilepsy/chb01_01.edf"
    if os.path.exists(chb_file):
        prep = CHBMITPreprocessor()
        sig = prep.preprocess_file(chb_file, subject_id="chb01")
        if sig:
            results.append(sig)

    # 2. SRM Healthy sample
    srm_file = "datasets/raw/healthy/ds003775/sub-010/ses-t1/eeg/sub-010_ses-t1_task-resteyesc_eeg.edf"
    if os.path.exists(srm_file):
        prep = SRMHealthyPreprocessor()
        sig = prep.preprocess_file(srm_file, subject_id="sub-010")
        if sig:
            results.append(sig)

    # 3. Alzheimer's sample
    alz_file = "datasets/raw/alzheimers/sub-001/eeg/sub-001_task-eyesclosed_eeg.set"
    if os.path.exists(alz_file):
        prep = AlzheimersPreprocessor()
        sig = prep.preprocess_file(alz_file, subject_id="sub-001", group_code="A")
        if sig:
            results.append(sig)

    # 4. Parkinson's sample
    park_file = "datasets/raw/parkinsons/sub-089/eeg/sub-089_task-Rest_eeg.set"
    if os.path.exists(park_file):
        prep = ParkinsonsPreprocessor()
        sig = prep.preprocess_file(park_file, subject_id="sub-089", group_code="PD")
        if sig:
            results.append(sig)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("\n--- Running Controlled Sample Preprocessing ---")
    samples = run_controlled_sample_preprocessing()
    for s in samples:
        print(f"Dataset: {s.dataset_name:<12} | Category: {s.class_category:<10} | Subject: {s.subject_id:<8} | "
              f"Srate: {s.srate_hz:6.1f} Hz | Ch: {s.n_channels:2d} | Shape: {s.data.shape} | Duration: {s.duration_sec:6.1f}s")
