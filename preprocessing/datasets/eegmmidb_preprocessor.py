"""
EEGMMIDB (PhysioNet EEG Motor Movement/Imagery Dataset) Preprocessor.

Maps to: 'healthy' class
- 109 healthy subjects
- 160 Hz, 64 EEG channels (10-10 system)
- Runs 1 & 2 are rest baseline (eyes-open T0 / eyes-closed T0)
- Data is stored in MNE's cache: ~/mne_data/MNE-eegbci-data/

Scientific note:
    All 109 subjects are HEALTHY VOLUNTEERS. This dataset is used for the
    'healthy' class only. Motor imagery runs (3-14) are NOT used, to avoid
    task contamination of the resting-state healthy baseline.
"""

import logging
import warnings
from pathlib import Path
from typing import Optional, List

import mne
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

warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")
warnings.filterwarnings("ignore", category=UserWarning, module="mne")
logger = logging.getLogger(__name__)

# EEGMMIDB rest runs: Run 1 = eyes open (T0), Run 2 = eyes closed (T0)
REST_RUNS = [1, 2]

# 19 canonical channels present in EEGMMIDB (subset of its 64 channels)
# These match the project's standard 10-20 set
EEGMMIDB_19CH = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T7',  'C3',  'Cz', 'C4', 'T8',
    'P7',  'P3',  'Pz', 'P4', 'P8',
    'O1',  'O2',
]

# Channel name aliases: EEGMMIDB uses T7/T8/P7/P8; project uses T3/T4/T5/T6
ALIAS_MAP = {'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6'}


class EEGMMIDBPreprocessor(BaseDatasetPreprocessor):
    """
    PhysioNet EEGMMIDB Healthy-Subject EEG Preprocessor.

    Downloads/loads from MNE's EEGBCI helper.
    Resamples from 160 Hz → 256 Hz.
    Picks 19 standard 10-20 channels.
    Applies average reference.
    Class: 'healthy'
    """

    def __init__(self):
        config = DatasetPreprocessorConfig(
            dataset_name='eegmmidb',
            target_srate_hz=256.0,   # Resample from 160 Hz to project standard
            l_freq=0.5,
            h_freq=45.0,
            notch_freqs=[60.0],      # US power-line noise
            rereference_mode='average',
        )
        super().__init__(config)

    def preprocess_subject(
        self,
        subject_id: int,
        runs: Optional[List[int]] = None,
        mne_data_path: Optional[str] = None,
    ) -> Optional[PreprocessedEEGSignal]:
        """
        Load and preprocess a single EEGMMIDB subject.

        Parameters
        ----------
        subject_id : int
            Subject number (1–109).
        runs : list of int, optional
            Runs to concatenate. Default: [1, 2] (rest baseline).
        mne_data_path : str, optional
            Path to MNE data directory. Defaults to ~/mne_data.
        """
        if runs is None:
            runs = REST_RUNS

        try:
            files = mne.datasets.eegbci.load_data(
                subject_id, runs,
                path=mne_data_path,
                verbose=False,
            )
        except Exception as e:
            logger.error(f"[EEGMMIDB] Failed to load subject {subject_id}: {e}")
            return None

        if not files:
            logger.warning(f"[EEGMMIDB] No files found for subject {subject_id}")
            return None

        raws = []
        for f in files:
            try:
                r = mne.io.read_raw_edf(str(f), preload=True, verbose=False)
                raws.append(r)
            except Exception as e:
                logger.warning(f"[EEGMMIDB] Could not load {f}: {e}")

        if not raws:
            return None

        # Concatenate runs
        if len(raws) == 1:
            raw = raws[0]
        else:
            raw = mne.concatenate_raws(raws, verbose=False)

        # Pick only EEG channels
        raw.pick_types(eeg=True, verbose=False)

        # Rename T7/T8/P7/P8 → T3/T4/T5/T6 to match project standard
        rename_map = {
            ch: ALIAS_MAP[ch]
            for ch in raw.ch_names
            if ch in ALIAS_MAP
        }
        if rename_map:
            raw.rename_channels(rename_map, verbose=False)

        # Pick the 19 standard channels that are available
        available_19 = [
            ch for ch in EEGMMIDB_19CH
            if ch in raw.ch_names or ALIAS_MAP.get(ch, ch) in raw.ch_names
        ]
        # After renaming, pick by canonical names
        canonical_present = [ch for ch in EEGMMIDB_19CH if ch in raw.ch_names]
        if len(canonical_present) < 5:
            logger.warning(
                f"[EEGMMIDB] Subject {subject_id}: only {len(canonical_present)}"
                f" of 19 standard channels found. Skipping."
            )
            return None

        raw.pick_channels(canonical_present, verbose=False)

        subject_str = f"S{subject_id:03d}"
        result = self.preprocess_raw_object(
            raw=raw,
            dataset_name='eegmmidb',
            class_category='healthy',
            subject_id=subject_str,
            file_path=str(files[0]),
        )
        return result

    def preprocess_file(
        self, file_path: str, subject_id: Optional[str] = None
    ) -> Optional[PreprocessedEEGSignal]:
        """Interface-compatible wrapper — parse subject ID from path."""
        import re
        if subject_id is None:
            m = re.search(r'S(\d+)', str(file_path))
            subject_id = int(m.group(1)) if m else 1
        else:
            try:
                subject_id = int(str(subject_id).lstrip('S').lstrip('0') or '1')
            except ValueError:
                subject_id = 1
        return self.preprocess_subject(subject_id)
