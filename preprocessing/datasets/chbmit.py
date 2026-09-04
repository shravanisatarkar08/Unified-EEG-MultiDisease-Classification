"""
CHB-MIT Epilepsy dataset preprocessing.

Dataset specifics:
- Flat directory structure with chb01_XX.edf files and chb01-summary.txt
- Bipolar montage (e.g., FP1-F7, F7-T7) - NOT individual electrodes
- 256 Hz sampling rate (native)
- Seizure annotations in summary text file

Since CHB-MIT uses a bipolar montage (23 channels), we CANNOT directly map
to the standard 19-channel 10-20 montage. Instead, we use a fixed set of
bipolar channels as our feature channels for this dataset. The CNN architecture
handles variable channel counts through configuration.

For the prototype, we select 19 of the most common bipolar channels.
"""

import os
import re
import glob
import logging
import numpy as np
import mne

from ..config import (TARGET_SRATE, WINDOW_SEC, OVERLAP, WINDOW_SAMPLES,
                     N_CHANNELS, DATASET_PATHS, DATASET_OUTPUT, L_FREQ, H_FREQ)
from ..segment import segment_continuous, segment_labeled
from ..metadata import WindowMetadata, save_metadata, append_metadata
from ..common import normalize_windows

logger = logging.getLogger(__name__)

# CHB-MIT uses bipolar montage - these are the 19 most common channels
# across most subjects (channels 1-19 in most recordings)
CHBMIT_CHANNELS = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ', 'CZ-PZ',
    'P7-T7'  # 19th channel to match N_CHANNELS
]

# Alternate naming that may appear
CHBMIT_ALIASES = {
    'FP1-F7': ['FP1-F7'],
    'T8-P8-0': 'T8-P8',  # MNE deduplication
    'T8-P8-1': None,      # skip duplicate
}

# Limit non-seizure windows per file to manage class imbalance
MAX_NONSEIZURE_WINDOWS_PER_FILE = 20


def parse_summary(summary_path):
    """Parse CHB-MIT summary file to extract seizure annotations.

    Args:
        summary_path: path to chbXX-summary.txt

    Returns:
        dict mapping filename -> list of (start_sec, end_sec)
    """
    seizures = {}
    current_file = None

    with open(summary_path, 'r') as f:
        for line in f:
            line = line.strip()

            # Match file name
            m = re.match(r'File Name:\s*(\S+)', line)
            if m:
                current_file = m.group(1)
                if current_file not in seizures:
                    seizures[current_file] = []
                continue

            # Match seizure start time (handles "Seizure Start Time" and
            # "Seizure 1 Start Time", "Seizure 2 Start Time", etc.)
            m = re.match(r'Seizure\s*\d*\s*Start Time:\s*(\d+)\s*seconds?', line)
            if m and current_file:
                start = int(m.group(1))
                continue

            # Match seizure end time
            m = re.match(r'Seizure\s*\d*\s*End Time:\s*(\d+)\s*seconds?', line)
            if m and current_file:
                end = int(m.group(1))
                seizures[current_file].append((start, end))
                continue

    # Filter to only files with seizures noted
    return seizures


def load_recording(edf_path):
    """Load a CHB-MIT EDF file.

    Args:
        edf_path: path to .edf file

    Returns:
        MNE Raw object or None if loading fails
    """
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        return raw
    except Exception as e:
        logger.error(f"Failed to load {edf_path}: {e}")
        return None


def select_chbmit_channels(raw):
    """Select and reorder CHB-MIT bipolar channels.

    Returns:
        numpy array [N_CHANNELS, N_samples], list of channels found
    """
    data = raw.get_data()
    ch_names = raw.ch_names
    n_samples = data.shape[1]

    # Build mapping: normalize channel names
    ch_map = {}
    for i, ch in enumerate(ch_names):
        # Remove -0, -1 suffixes from MNE dedup
        clean = re.sub(r'-(\d+)$', '', ch)
        if clean not in ch_map:  # keep first occurrence
            ch_map[clean] = i

    # Select our target channels
    output = np.zeros((N_CHANNELS, n_samples), dtype=np.float64)
    found = []
    for j, target_ch in enumerate(CHBMIT_CHANNELS[:N_CHANNELS]):
        if target_ch in ch_map:
            output[j] = data[ch_map[target_ch]]
            found.append(target_ch)
        else:
            logger.debug(f"Channel {target_ch} not found")

    return output, found


def apply_filters(raw):
    """Apply filters to CHB-MIT data (already at 256 Hz, no resampling needed)."""
    raw.load_data()

    # Notch filter
    srate = raw.info['sfreq']
    notch_freqs = [f for f in [50.0, 60.0] if f < srate / 2]
    if notch_freqs:
        raw.notch_filter(notch_freqs, verbose=False)

    # Bandpass filter
    raw.filter(L_FREQ, H_FREQ, fir_design='firwin', verbose=False)

    return raw


def process_subject(subject_prefix, raw_dir, output_dir, max_files=None):
    """Process all recordings for one CHB-MIT subject.

    Args:
        subject_prefix: e.g., 'chb01'
        raw_dir: directory containing EDF files and summary
        output_dir: directory to save processed windows
        max_files: limit number of files to process (for testing)

    Returns:
        list of WindowMetadata
    """
    summary_path = os.path.join(raw_dir, f'{subject_prefix}-summary.txt')
    if not os.path.exists(summary_path):
        logger.warning(f"No summary file for {subject_prefix}")
        return []

    seizures = parse_summary(summary_path)
    logger.info(f"[{subject_prefix}] Found seizure annotations for "
                f"{sum(1 for v in seizures.values() if v)} files")

    # Find available EDF files for this subject
    edf_pattern = os.path.join(raw_dir, f'{subject_prefix}_*.edf')
    edf_files = sorted(glob.glob(edf_pattern))
    if max_files:
        edf_files = edf_files[:max_files]

    logger.info(f"[{subject_prefix}] Processing {len(edf_files)} EDF files")

    os.makedirs(output_dir, exist_ok=True)
    all_metadata = []
    window_counter = 0

    for edf_path in edf_files:
        edf_name = os.path.basename(edf_path)
        recording_id = os.path.splitext(edf_name)[0]

        raw = load_recording(edf_path)
        if raw is None:
            continue

        # Apply filters
        try:
            apply_filters(raw)
        except Exception as e:
            logger.error(f"Filter failed for {edf_name}: {e}")
            continue

        # Select channels
        data, channels_found = select_chbmit_channels(raw)
        if len(channels_found) < N_CHANNELS // 2:
            logger.warning(f"Too few channels in {edf_name}: {len(channels_found)}")
            continue

        srate = raw.info['sfreq']
        file_seizures = seizures.get(edf_name, [])

        if file_seizures:
            # Create seizure windows
            seizure_windows, seizure_ranges = segment_labeled(
                data, file_seizures, 'seizure', srate=srate
            )
            if len(seizure_windows) > 0:
                seizure_windows = normalize_windows(seizure_windows)
                for i, (ws, we) in enumerate(seizure_ranges):
                    fname = f"{recording_id}_seizure_{window_counter:06d}.npy"
                    fpath = os.path.join(output_dir, fname)
                    np.save(fpath, seizure_windows[i])
                    all_metadata.append(WindowMetadata(
                        dataset='chbmit', subject_id=subject_prefix,
                        recording_id=recording_id, label='seizure',
                        window_idx=window_counter, window_start=ws,
                        window_end=we, n_channels=N_CHANNELS,
                        srate=srate, n_channels_found=len(channels_found),
                        file_path=fname
                    ))
                    window_counter += 1

            # Create non-seizure windows from non-seizure intervals
            total_duration = data.shape[1] / srate
            non_seizure_intervals = []
            prev_end = 0
            for ss, se in sorted(file_seizures):
                if ss > prev_end:
                    non_seizure_intervals.append((prev_end, ss))
                prev_end = se
            if prev_end < total_duration:
                non_seizure_intervals.append((prev_end, total_duration))

            ns_windows, ns_ranges = segment_labeled(
                data, non_seizure_intervals, 'non_seizure', srate=srate
            )
            if len(ns_windows) > 0:
                # Limit non-seizure windows
                limit = MAX_NONSEIZURE_WINDOWS_PER_FILE
                ns_windows = ns_windows[:limit]
                ns_ranges = ns_ranges[:limit]
                ns_windows = normalize_windows(ns_windows)
                for i, (ws, we) in enumerate(ns_ranges):
                    fname = f"{recording_id}_nonseizure_{window_counter:06d}.npy"
                    fpath = os.path.join(output_dir, fname)
                    np.save(fpath, ns_windows[i])
                    all_metadata.append(WindowMetadata(
                        dataset='chbmit', subject_id=subject_prefix,
                        recording_id=recording_id, label='non_seizure',
                        window_idx=window_counter, window_start=ws,
                        window_end=we, n_channels=N_CHANNELS,
                        srate=srate, n_channels_found=len(channels_found),
                        file_path=fname
                    ))
                    window_counter += 1
        else:
            # Non-seizure file: create limited windows
            ns_windows, ns_ranges = segment_continuous(data, srate=srate)
            if len(ns_windows) > 0:
                limit = MAX_NONSEIZURE_WINDOWS_PER_FILE
                ns_windows = ns_windows[:limit]
                ns_ranges = ns_ranges[:limit]
                ns_windows = normalize_windows(ns_windows)
                for i, (ws, we) in enumerate(ns_ranges):
                    fname = f"{recording_id}_nonseizure_{window_counter:06d}.npy"
                    fpath = os.path.join(output_dir, fname)
                    np.save(fpath, ns_windows[i])
                    all_metadata.append(WindowMetadata(
                        dataset='chbmit', subject_id=subject_prefix,
                        recording_id=recording_id, label='non_seizure',
                        window_idx=window_counter, window_start=ws,
                        window_end=we, n_channels=N_CHANNELS,
                        srate=srate, n_channels_found=len(channels_found),
                        file_path=fname
                    ))
                    window_counter += 1

        del raw  # free memory

    logger.info(f"[{subject_prefix}] Created {window_counter} windows")
    return all_metadata


def process_dataset(raw_dir=None, output_dir=None, max_subjects=None, max_files_per_subject=None):
    """Process the entire CHB-MIT dataset.

    Args:
        raw_dir: path to raw epilepsy data
        output_dir: path to save processed data
        max_subjects: limit number of subjects
        max_files_per_subject: limit files per subject
    """
    if raw_dir is None:
        raw_dir = DATASET_PATHS['chbmit']
    if output_dir is None:
        output_dir = DATASET_OUTPUT['chbmit']

    os.makedirs(output_dir, exist_ok=True)

    # Find subject prefixes from summary files
    summary_files = glob.glob(os.path.join(raw_dir, '*-summary.txt'))
    subjects = []
    for sf in sorted(summary_files):
        name = os.path.basename(sf)
        prefix = name.replace('-summary.txt', '')
        subjects.append(prefix)

    if max_subjects:
        subjects = subjects[:max_subjects]

    logger.info(f"Processing {len(subjects)} CHB-MIT subjects: {subjects}")

    all_metadata = []
    for subj in subjects:
        subj_output = os.path.join(output_dir, subj)
        try:
            meta = process_subject(subj, raw_dir, subj_output,
                                   max_files=max_files_per_subject)
            all_metadata.extend(meta)
        except Exception as e:
            logger.error(f"Failed to process {subj}: {e}")
            continue

    # Save combined metadata
    if all_metadata:
        meta_path = os.path.join(output_dir, 'metadata.csv')
        save_metadata(all_metadata, meta_path)
        logger.info(f"CHB-MIT preprocessing complete: {len(all_metadata)} windows")
    else:
        logger.warning("No windows created for CHB-MIT")

    return all_metadata


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
    # Quick test with first subject, first 2 files
    meta = process_dataset(max_subjects=1, max_files_per_subject=2)
    print(f"Created {len(meta)} windows")
    if meta:
        print(f"Labels: seizure={sum(1 for m in meta if m.label=='seizure')}, "
              f"non_seizure={sum(1 for m in meta if m.label=='non_seizure')}")
