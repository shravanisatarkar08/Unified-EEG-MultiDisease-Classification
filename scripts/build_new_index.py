"""
Build the model_dataset_index.csv for the new multi-dataset experiment.

Datasets:
  1. EEGMMIDB (PhysioNet)     → healthy     (runs 1-2, subjects 1-40)
  2. CHB-MIT (Kaggle/chb01)   → epilepsy    (17 recordings, with 7 seizure sessions)
  3. DEAP (.dat files)        → depression  (if .dat files are provided)

Splits:
  - Healthy (EEGMMIDB): 70% train, 15% val, 15% test by subject ID
  - Epilepsy (CHB-MIT): partitioned across 17 multi-hour recordings:
      • Train (11 files): 5 seizure files + 6 non-seizure files
      • Val (3 files): 1 seizure file (chb01_21) + 2 non-seizure files
      • Test (3 files): 1 seizure file (chb01_26) + 2 non-seizure files
    This ensures epilepsy events exist in train, validation, AND test splits,
    enabling honest, multi-class confusion matrix, ROC-AUC, and Grad-CAM evaluation.
  - Subsampling: CHB-MIT windows are sampled (~40-45 windows per 1-hour recording,
    always including all seizure windows) to achieve balanced representation with healthy windows (~750-900 windows each).
"""

import csv
import logging
import os
import random
import sys
from pathlib import Path

import mne
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_SRATE   = 256
WINDOW_SEC     = 5
WINDOW_SAMPLES = TARGET_SRATE * WINDOW_SEC   # 1280
N_CHANNELS     = 19
RANDOM_SEED    = 42

METADATA_DIR = PROJECT_ROOT / "datasets" / "metadata"
INDEX_CSV    = METADATA_DIR / "model_dataset_index.csv"
EEGBCI_PATH  = Path.home() / "mne_data"

CLASS_TO_IDX = {
    'healthy':    0,
    'epilepsy':   1,
    'alzheimers': 2,
    'parkinsons': 3,
    'depression': 4,
}

# Annotated seizure intervals (in seconds) for CHB-MIT chb01 recordings
CHB01_SEIZURES = {
    "chb01_03.edf": [(2996, 3036)],
    "chb01_04.edf": [(1467, 1494)],
    "chb01_15.edf": [(1732, 1772)],
    "chb01_16.edf": [(1015, 1066)],
    "chb01_18.edf": [(1720, 1810)],
    "chb01_21.edf": [(327, 420)],
    "chb01_26.edf": [(1862, 1963)],
}

CHB01_PARTITIONS = {
    # Train: 5 seizure files, 6 non-seizure files
    "chb01_01.edf": "train",
    "chb01_02.edf": "train",
    "chb01_03.edf": "train",
    "chb01_04.edf": "train",
    "chb01_05.edf": "train",
    "chb01_06.edf": "train",
    "chb01_07.edf": "train",
    "chb01_08.edf": "train",
    "chb01_15.edf": "train",
    "chb01_16.edf": "train",
    "chb01_18.edf": "train",
    # Val: 1 seizure file, 2 non-seizure files
    "chb01_09.edf": "val",
    "chb01_10.edf": "val",
    "chb01_21.edf": "val",
    # Test: 1 seizure file, 2 non-seizure files
    "chb01_11.edf": "test",
    "chb01_12.edf": "test",
    "chb01_26.edf": "test",
}


def subject_level_split(subjects, seed=RANDOM_SEED):
    """Split subjects into train/val/test at 70/15/15."""
    rng = random.Random(seed)
    subj = list(subjects)
    rng.shuffle(subj)
    n = len(subj)
    n_test = max(1, round(n * 0.15))
    n_val  = max(1, round(n * 0.15))
    n_train = n - n_val - n_test
    if n == 1:
        return subj, [], []
    if n == 2:
        return subj[:1], subj[1:], []
    return subj[:n_train], subj[n_train:n_train+n_val], subj[n_train+n_val:]


# ── EEGMMIDB (Healthy) ────────────────────────────────────────────────────────

def collect_eegmmidb_subjects():
    """Return list of subject IDs (int) that have both run 1 and run 2 cached."""
    base = EEGBCI_PATH / "MNE-eegbci-data" / "files" / "eegmmidb" / "1.0.0"
    if not base.exists():
        logger.warning("[EEGMMIDB] Cache path not found: %s", base)
        return []

    available = []
    for s in range(1, 110):
        r1 = base / f"S{s:03d}" / f"S{s:03d}R01.edf"
        r2 = base / f"S{s:03d}" / f"S{s:03d}R02.edf"
        if r1.exists() and r2.exists() and r1.stat().st_size > 50_000 and r2.stat().st_size > 50_000:
            available.append(s)
    logger.info("[EEGMMIDB] %d subjects with both rest runs cached", len(available))
    return available


def index_eegmmidb(writer, split_map):
    """Scan EEGMMIDB and write index rows for each file."""
    base = EEGBCI_PATH / "MNE-eegbci-data" / "files" / "eegmmidb" / "1.0.0"
    rows_written = 0

    for subj_int, split in split_map.items():
        subject_str = f"S{subj_int:03d}"
        r1 = base / subject_str / f"{subject_str}R01.edf"
        r2 = base / subject_str / f"{subject_str}R02.edf"

        for edf_path in [r1, r2]:
            if not edf_path.exists():
                continue
            try:
                raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)
                native_sr = float(raw.info['sfreq'])
                duration_sec = raw.n_times / native_sr
            except Exception as e:
                logger.warning("[EEGMMIDB] %s: load failed: %s", edf_path.name, e)
                continue

            # Generate 5-second non-overlapping windows
            starts = np.arange(0, duration_sec - WINDOW_SEC + 1e-4, WINDOW_SEC)
            for win_idx, ws in enumerate(starts):
                we = ws + WINDOW_SEC
                writer.writerow({
                    'split':            split,
                    'dataset_name':     'eegmmidb',
                    'subject_id':       subject_str,
                    'class_category':   'healthy',
                    'class_label_idx':  CLASS_TO_IDX['healthy'],
                    'source_file':      str(edf_path).replace('\\', '/'),
                    'window_idx':       win_idx,
                    'window_start_sec': round(float(ws), 3),
                    'window_end_sec':   round(float(we), 3),
                    'sampling_rate':    TARGET_SRATE,
                    'n_channels':       N_CHANNELS,
                    'n_samples':        WINDOW_SAMPLES,
                    'is_bipolar_montage': False,
                    'missing_channels': '',
                })
                rows_written += 1

    logger.info("[EEGMMIDB] Wrote %d window rows", rows_written)
    return rows_written


# ── CHB-MIT (Epilepsy) ────────────────────────────────────────────────────────

def index_chbmit(writer):
    """Scan CHB-MIT and write index rows partitioned across recordings."""
    epilepsy_dir = PROJECT_ROOT / "datasets" / "raw" / "epilepsy" / "chb01"
    if not epilepsy_dir.exists():
        logger.warning("[CHB-MIT] Epilepsy directory not found: %s", epilepsy_dir)
        return 0

    rows_written = 0
    rng = random.Random(RANDOM_SEED)

    for edf_name, split in CHB01_PARTITIONS.items():
        edf_path = epilepsy_dir / edf_name
        if not edf_path.exists():
            continue

        try:
            raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)
            native_sr = float(raw.info['sfreq'])
            duration_sec = raw.n_times / native_sr
        except Exception as e:
            logger.warning("[CHB-MIT] %s: load failed: %s", edf_name, e)
            continue

        seizures = CHB01_SEIZURES.get(edf_name, [])
        window_starts = []

        if seizures:
            # 1. Capture ALL seizure windows
            for sz_start, sz_end in seizures:
                # Pre-ictal windows (up to 30s before)
                pre_starts = np.arange(max(0, sz_start - 30), sz_start, WINDOW_SEC)
                window_starts.extend(pre_starts)
                # Ictal windows (step 2.5s for rich seizure representation)
                ictal_starts = np.arange(sz_start, sz_end - WINDOW_SEC + 1e-4, 2.5)
                window_starts.extend(ictal_starts)
                # Post-ictal windows (up to 30s after)
                post_starts = np.arange(sz_end, min(duration_sec - WINDOW_SEC, sz_end + 30), WINDOW_SEC)
                window_starts.extend(post_starts)

            # 2. Add interictal windows from rest of recording
            interictal_candidates = np.arange(0, duration_sec - WINDOW_SEC, 100.0)
            # Filter candidates that don't overlap with seizures
            interictal_valid = []
            for c in interictal_candidates:
                in_sz = any(sz_s - 60 <= c <= sz_e + 60 for sz_s, sz_e in seizures)
                if not in_sz:
                    interictal_valid.append(c)
            # Sample 25 interictal windows
            if len(interictal_valid) > 25:
                window_starts.extend(rng.sample(interictal_valid, 25))
            else:
                window_starts.extend(interictal_valid)
        else:
            # Non-seizure recording: sample 40 evenly distributed windows across the hour
            # (every 85 seconds)
            all_starts = np.arange(10.0, duration_sec - WINDOW_SEC - 10.0, 85.0)
            window_starts.extend(all_starts[:42])

        # Remove duplicates and sort
        window_starts = sorted(list(set(round(float(ws), 3) for ws in window_starts if ws + WINDOW_SEC <= duration_sec)))

        for win_idx, ws in enumerate(window_starts):
            we = ws + WINDOW_SEC
            writer.writerow({
                'split':            split,
                'dataset_name':     'chbmit',
                'subject_id':       'chb01',
                'class_category':   'epilepsy',
                'class_label_idx':  CLASS_TO_IDX['epilepsy'],
                'source_file':      str(edf_path).replace('\\', '/'),
                'window_idx':       win_idx,
                'window_start_sec': ws,
                'window_end_sec':   round(we, 3),
                'sampling_rate':    TARGET_SRATE,
                'n_channels':       N_CHANNELS,
                'n_samples':        WINDOW_SAMPLES,
                'is_bipolar_montage': True,
                'missing_channels': '',
            })
            rows_written += 1

    logger.info("[CHB-MIT] Wrote %d window rows", rows_written)
    return rows_written


# ── DEAP (Depression proxy) ───────────────────────────────────────────────────

def collect_deap_subjects():
    """Return list of DEAP .dat file paths."""
    deap_dir = PROJECT_ROOT / "datasets" / "raw" / "depression"
    if not deap_dir.exists():
        return []
    dat_files = list(deap_dir.glob("s*.dat"))
    return sorted(dat_files)


def index_deap(writer, split_map):
    """Scan DEAP and write index rows."""
    rows_written = 0
    deap_dir = PROJECT_ROOT / "datasets" / "raw" / "depression"

    for dat_file, split in split_map.items():
        import re
        m = re.search(r's(\d+)', dat_file.stem, re.IGNORECASE)
        subject_id = f"s{int(m.group(1)):02d}" if m else dat_file.stem

        N_TRIALS_EST  = 10
        n_samples_est = int(N_TRIALS_EST * 63 * TARGET_SRATE)
        starts = np.arange(0, (n_samples_est // TARGET_SRATE) - WINDOW_SEC, WINDOW_SEC)

        for win_idx, ws in enumerate(starts):
            we = ws + WINDOW_SEC
            writer.writerow({
                'split':            split,
                'dataset_name':     'deap',
                'subject_id':       subject_id,
                'class_category':   'depression',
                'class_label_idx':  CLASS_TO_IDX['depression'],
                'source_file':      str(dat_file).replace('\\', '/'),
                'window_idx':       win_idx,
                'window_start_sec': round(float(ws), 3),
                'window_end_sec':   round(float(we), 3),
                'sampling_rate':    TARGET_SRATE,
                'n_channels':       N_CHANNELS,
                'n_samples':        WINDOW_SAMPLES,
                'is_bipolar_montage': False,
                'missing_channels': '',
            })
            rows_written += 1

    logger.info("[DEAP] Wrote %d window rows", rows_written)
    return rows_written


# ── Main ──────────────────────────────────────────────────────────────────────

def build_index():
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Building model_dataset_index.csv")
    logger.info("=" * 60)

    # Collect available subjects
    eegmmidb_subjects = collect_eegmmidb_subjects()
    deap_files        = collect_deap_subjects()

    # Subject-level splits for healthy (EEGMMIDB)
    eegmmidb_train, eegmmidb_val, eegmmidb_test = subject_level_split(eegmmidb_subjects)
    deap_train, deap_val, deap_test = subject_level_split(deap_files, seed=RANDOM_SEED)

    logger.info("[EEGMMIDB] train=%d val=%d test=%d subjects",
                len(eegmmidb_train), len(eegmmidb_val), len(eegmmidb_test))

    # Build split maps
    eeg_split_map = {}
    for s in eegmmidb_train: eeg_split_map[s] = 'train'
    for s in eegmmidb_val:   eeg_split_map[s] = 'val'
    for s in eegmmidb_test:  eeg_split_map[s] = 'test'

    deap_split_map = {}
    for f in deap_train: deap_split_map[f] = 'train'
    for f in deap_val:   deap_split_map[f] = 'val'
    for f in deap_test:  deap_split_map[f] = 'test'

    fieldnames = [
        'split', 'dataset_name', 'subject_id', 'class_category',
        'class_label_idx', 'source_file', 'window_idx',
        'window_start_sec', 'window_end_sec', 'sampling_rate',
        'n_channels', 'n_samples', 'is_bipolar_montage', 'missing_channels',
    ]

    total_rows = 0
    with open(INDEX_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        if eegmmidb_subjects:
            total_rows += index_eegmmidb(writer, eeg_split_map)
        else:
            logger.warning("No EEGMMIDB subjects available — healthy class EMPTY")

        total_rows += index_chbmit(writer)

        if deap_files:
            total_rows += index_deap(writer, deap_split_map)
        else:
            logger.info("[DEAP] Not available — depression class absent")

    logger.info("=" * 60)
    logger.info("Index written: %s", INDEX_CSV)
    logger.info("Total window rows: %d", total_rows)
    logger.info("=" * 60)
    return total_rows


if __name__ == "__main__":
    n = build_index()
    print(f"\nIndex built successfully: {n} windows -> {INDEX_CSV}")
