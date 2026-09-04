"""
Alzheimer's disease EEG dataset preprocessor.
Dataset: OpenNeuro ds004504 (Alzheimer's EEG)

Subject groups in the dataset:
  - A  : Alzheimer's Disease (AD)  -> label 'alzheimer'
  - C  : Healthy Control (CN)       -> label 'healthy'
  - F  : Frontotemporal Dementia (FTD) -> EXCLUDED per project spec

Expected structure:
  datasets/raw/alzheimers/
    participants.tsv  (columns: participant_id, group)
    sub-001/eeg/sub-001_task-eyesclosed_eeg.edf (or .set, .fif, .bdf)
    ...
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

GROUP_ALZHEIMER = 'A'
GROUP_HEALTHY   = 'C'
GROUP_FTD       = 'F'   # explicitly excluded

LABEL_MAP = {
    GROUP_ALZHEIMER: 'alzheimer',
    GROUP_HEALTHY:   'healthy',
}

MAX_WINDOWS_PER_FILE = 30


def _read_participants(raw_dir):
    """Parse participants.tsv to get {participant_id: group} mapping."""
    tsv_path = os.path.join(raw_dir, 'participants.tsv')
    if not os.path.exists(tsv_path):
        logger.warning(f"participants.tsv not found at {tsv_path}")
        return {}
    participants = {}
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            pid = row.get('participant_id', '').strip()
            grp = row.get('group', '').strip().upper()
            if pid:
                participants[pid] = grp
    n_ad  = sum(1 for g in participants.values() if g == GROUP_ALZHEIMER)
    n_hc  = sum(1 for g in participants.values() if g == GROUP_HEALTHY)
    n_ftd = sum(1 for g in participants.values() if g == GROUP_FTD)
    logger.info(f"ds004504 participants: AD={n_ad}, HC={n_hc}, FTD={n_ftd} (FTD excluded)")
    return participants


def _find_eeg_file(subject_dir):
    """Locate EEG data file for a subject."""
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
    """Load EEG file via MNE, return Raw or None."""
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
        logger.error(f"Failed to load {file_path}: {e}")
        return None


def _apply_filters(raw):
    """Apply notch + bandpass filter and resample to TARGET_SRATE."""
    srate = raw.info['sfreq']
    notch = [f for f in [50.0, 60.0] if f < srate / 2]
    if notch:
        raw.notch_filter(notch, verbose=False)
    raw.filter(L_FREQ, H_FREQ, fir_design='firwin', verbose=False)
    if srate != TARGET_SRATE:
        raw.resample(TARGET_SRATE, verbose=False)
    return raw


def process_subject(subject_id, group, raw_dir, output_dir, max_windows=None):
    """Process one subject and write .npy windows + return metadata list."""
    label = LABEL_MAP.get(group)
    if label is None:
        logger.warning(f"Skipping {subject_id}: group '{group}' not AD or HC")
        return []
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
        logger.error(f"Filter failed for {subject_id}: {e}")
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
            dataset='alzheimer', subject_id=subject_id,
            recording_id=recording_id, label=label,
            window_idx=i, window_start=ws, window_end=we,
            n_channels=N_CHANNELS, srate=TARGET_SRATE,
            n_channels_found=len(channels_found), file_path=fname,
        ))
    logger.info(f"[{subject_id}] Created {len(all_meta)} '{label}' windows.")
    return all_meta


def process_dataset(raw_dir=None, output_dir=None, max_subjects=None,
                    max_windows_per_subject=None):
    """Process full ds004504 dataset (AD + HC only, FTD excluded)."""
    if raw_dir is None:
        raw_dir = DATASET_PATHS['alzheimer']
    if output_dir is None:
        output_dir = DATASET_OUTPUT['alzheimer']
    if not os.path.isdir(raw_dir):
        logger.error(f"Alzheimer's raw dir not found: {raw_dir}")
        return []
    participants = _read_participants(raw_dir)
    if not participants:
        return []
    eligible = {pid: grp for pid, grp in participants.items()
                if grp in (GROUP_ALZHEIMER, GROUP_HEALTHY)}
    ftd_excluded = [pid for pid, g in participants.items() if g == GROUP_FTD]
    if ftd_excluded:
        logger.info(f"FTD subjects excluded: {len(ftd_excluded)}")
    subjects = sorted(eligible.keys())
    if max_subjects:
        subjects = subjects[:max_subjects]
    logger.info(f"Processing {len(subjects)} subjects (AD+HC, FTD excluded)")
    os.makedirs(output_dir, exist_ok=True)
    all_metadata = []
    for subj in subjects:
        try:
            meta = process_subject(subj, eligible[subj], raw_dir, output_dir,
                                   max_windows=max_windows_per_subject)
            all_metadata.extend(meta)
        except Exception as e:
            logger.error(f"Failed {subj}: {e}")
    if all_metadata:
        save_metadata(all_metadata, os.path.join(output_dir, 'metadata.csv'))
        n_ad = sum(1 for m in all_metadata if m.label == 'alzheimer')
        n_hc = sum(1 for m in all_metadata if m.label == 'healthy')
        logger.info(f"Done: AD={n_ad} windows, HC={n_hc} windows")
    return all_metadata


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(name)s - %(levelname)s - %(message)s')
    meta = process_dataset(max_subjects=2, max_windows_per_subject=10)
    print(f"Created {len(meta)} windows")
