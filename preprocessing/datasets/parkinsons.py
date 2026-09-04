"""
Parkinson's disease EEG dataset preprocessor.
Dataset: OpenNeuro ds004584

Expected structure:
  datasets/raw/parkinsons/
    participants.tsv  (columns: participant_id, group / diagnosis)
    sub-XXX/eeg/sub-XXX_*.edf  (or .set, .fif, .bdf)

Parkinson's vs Healthy labels are derived from participants.tsv.
Common group codes in ds004584:
  PD  -> 'parkinson'
  HC / Control / Healthy -> 'healthy'
We inspect actual column values and normalise accordingly.
"""

import os
import glob
import logging
import csv
import numpy as np

try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False

from ..config import (
    TARGET_SRATE, WINDOW_SEC, OVERLAP, WINDOW_SAMPLES,
    N_CHANNELS, DATASET_PATHS, DATASET_OUTPUT, L_FREQ, H_FREQ,
)
from ..segment import segment_continuous
from ..metadata import WindowMetadata, save_metadata
from ..common import normalize_windows, select_and_reorder_channels

logger = logging.getLogger(__name__)

MAX_WINDOWS_PER_FILE = 30

# Possible raw group codes that map to our labels.
# We normalise by upper-casing the raw value.
_PARKINSON_CODES = {'PD', 'PARKINSON', "PARKINSON'S", 'P'}
_HEALTHY_CODES   = {'HC', 'HEALTHY', 'CONTROL', 'NORMAL', 'CN', 'H', 'C'}


def _classify_group(raw_group: str):
    """Return 'parkinson', 'healthy', or None (unknown/exclude)."""
    code = raw_group.strip().upper()
    if code in _PARKINSON_CODES:
        return 'parkinson'
    if code in _HEALTHY_CODES:
        return 'healthy'
    return None


def _read_participants(raw_dir):
    """Parse participants.tsv to {participant_id: label} (only known groups)."""
    tsv_path = os.path.join(raw_dir, 'participants.tsv')
    if not os.path.exists(tsv_path):
        logger.warning(f"participants.tsv not found: {tsv_path}")
        return {}

    participants = {}
    unknown_groups = set()
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        # Accept either 'group' or 'diagnosis' column
        for row in reader:
            pid = row.get('participant_id', '').strip()
            grp_raw = (row.get('group') or row.get('diagnosis') or '').strip()
            if not pid:
                continue
            label = _classify_group(grp_raw)
            if label:
                participants[pid] = label
            else:
                unknown_groups.add(grp_raw)

    n_pd = sum(1 for l in participants.values() if l == 'parkinson')
    n_hc = sum(1 for l in participants.values() if l == 'healthy')
    logger.info(f"ds004584 participants: PD={n_pd}, HC={n_hc}")
    if unknown_groups:
        logger.warning(f"Unrecognised group codes (excluded): {unknown_groups}")
    return participants


def _find_eeg_file(subject_dir):
    """Locate the first EEG data file for a subject."""
    eeg_subdir = os.path.join(subject_dir, 'eeg')
    for d in [eeg_subdir, subject_dir]:
        if not os.path.isdir(d):
            continue
        for ext in ['*.edf', '*.set', '*.fif', '*.bdf', '*.vhdr']:
            matches = glob.glob(os.path.join(d, ext))
            if matches:
                return matches[0]
    return None


def _load_eeg(file_path):
    if not HAS_MNE:
        logger.error("MNE not installed.")
        return None
    try:
        ext = os.path.splitext(file_path)[1].lower()
        loaders = {
            '.edf': mne.io.read_raw_edf,
            '.set': mne.io.read_raw_eeglab,
            '.fif': mne.io.read_raw_fif,
            '.bdf': mne.io.read_raw_bdf,
            '.vhdr': mne.io.read_raw_brainvision,
        }
        loader = loaders.get(ext)
        if loader is None:
            logger.warning(f"Unsupported format: {ext}")
            return None
        return loader(file_path, preload=True, verbose=False)
    except Exception as e:
        logger.error(f"Load failed {file_path}: {e}")
        return None


def _apply_filters(raw):
    srate = raw.info['sfreq']
    notch = [f for f in [50.0, 60.0] if f < srate / 2]
    if notch:
        raw.notch_filter(notch, verbose=False)
    raw.filter(L_FREQ, H_FREQ, fir_design='firwin', verbose=False)
    if srate != TARGET_SRATE:
        raw.resample(TARGET_SRATE, verbose=False)
    return raw


def process_subject(subject_id, label, raw_dir, output_dir, max_windows=None):
    """Process one subject; return list of WindowMetadata."""
    subject_dir = os.path.join(raw_dir, subject_id)
    if not os.path.isdir(subject_dir):
        logger.warning(f"Subject dir not found: {subject_dir}")
        return []
    eeg_path = _find_eeg_file(subject_dir)
    if eeg_path is None:
        logger.warning(f"No EEG file for {subject_id}")
        return []
    raw = _load_eeg(eeg_path)
    if raw is None:
        return []
    logger.info(f"[{subject_id}] sfreq={raw.info['sfreq']} Hz, "
                f"ch={len(raw.ch_names)}, dur={raw.times[-1]:.1f}s, label={label}")
    try:
        raw = _apply_filters(raw)
    except Exception as e:
        logger.error(f"Filter failed {subject_id}: {e}")
        del raw
        return []
    data, channels_found = select_and_reorder_channels(raw)
    del raw
    if len(channels_found) < N_CHANNELS // 2:
        logger.warning(f"[{subject_id}] Only {len(channels_found)} channels — skipping.")
        return []
    windows, time_ranges = segment_continuous(data, srate=TARGET_SRATE)
    if len(windows) == 0:
        return []
    limit = max_windows or MAX_WINDOWS_PER_FILE
    windows = windows[:limit]
    time_ranges = time_ranges[:limit]
    windows = normalize_windows(windows)
    os.makedirs(output_dir, exist_ok=True)
    recording_id = f"{subject_id}_{label}"
    all_meta = []
    for i, (ws, we) in enumerate(time_ranges):
        fname = f"{recording_id}_{i:06d}.npy"
        np.save(os.path.join(output_dir, fname), windows[i])
        all_meta.append(WindowMetadata(
            dataset='parkinson', subject_id=subject_id,
            recording_id=recording_id, label=label,
            window_idx=i, window_start=ws, window_end=we,
            n_channels=N_CHANNELS, srate=TARGET_SRATE,
            n_channels_found=len(channels_found), file_path=fname,
        ))
    logger.info(f"[{subject_id}] Created {len(all_meta)} '{label}' windows.")
    return all_meta


def process_dataset(raw_dir=None, output_dir=None, max_subjects=None,
                    max_windows_per_subject=None):
    """Process full ds004584 Parkinson's dataset."""
    if raw_dir is None:
        raw_dir = DATASET_PATHS['parkinson']
    if output_dir is None:
        output_dir = DATASET_OUTPUT['parkinson']
    if not os.path.isdir(raw_dir):
        logger.error(f"Parkinson's raw dir not found: {raw_dir}")
        return []
    participants = _read_participants(raw_dir)
    if not participants:
        return []
    subjects = sorted(participants.keys())
    if max_subjects:
        subjects = subjects[:max_subjects]
    logger.info(f"Processing {len(subjects)} Parkinson's subjects")
    os.makedirs(output_dir, exist_ok=True)
    all_metadata = []
    for subj in subjects:
        try:
            meta = process_subject(subj, participants[subj], raw_dir, output_dir,
                                   max_windows=max_windows_per_subject)
            all_metadata.extend(meta)
        except Exception as e:
            logger.error(f"Failed {subj}: {e}")
    if all_metadata:
        save_metadata(all_metadata, os.path.join(output_dir, 'metadata.csv'))
        n_pd = sum(1 for m in all_metadata if m.label == 'parkinson')
        n_hc = sum(1 for m in all_metadata if m.label == 'healthy')
        logger.info(f"Done: PD={n_pd} windows, HC={n_hc} windows")
    return all_metadata


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(name)s - %(levelname)s - %(message)s')
    meta = process_dataset(max_subjects=2, max_windows_per_subject=10)
    print(f"Created {len(meta)} windows")
