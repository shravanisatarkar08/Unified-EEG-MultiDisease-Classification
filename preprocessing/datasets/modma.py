"""
NEMAR nm000114 depression EEG dataset handler.

The repository keeps the historical module name "modma" for compatibility
with the existing preprocessing configuration, but the current raw data
under datasets/raw/depression/ is NEMAR nm000114.

Labels:
    sub-HS*   -> healthy
    sub-MDDS* -> depression

Valid recordings:
    eyesClosed
    eyesOpen

Excluded:
    P300 task recordings
"""

import os
import glob
import logging

import numpy as np

try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False

from ..config import (
    TARGET_SRATE,
    WINDOW_SEC,
    OVERLAP,
    WINDOW_SAMPLES,
    N_CHANNELS,
    DATASET_PATHS,
    DATASET_OUTPUT,
    L_FREQ,
    H_FREQ,
)
from ..segment import segment_continuous
from ..metadata import WindowMetadata, save_metadata
from ..common import normalize_windows, select_and_reorder_channels

logger = logging.getLogger(__name__)

MAX_WINDOWS_PER_FILE = 30

LABEL_HEALTHY = "healthy"
LABEL_DEPRESSION = "depression"


def inspect_raw_dir(raw_dir=None):
    """Summarize the local NEMAR nm000114 package without fabricating participant counts.

    The on-disk BIDS package currently contains 30 ``sub-HS*`` folders and
    34 ``sub-MDDS*`` folders. Source documentation describes 34 participants;
    this function reports folder counts as observed and does not claim 64
    unique clinical participants.
    """
    if raw_dir is None:
        raw_dir = DATASET_PATHS["modma"]

    hs_folders = sorted(
        p for p in glob.glob(os.path.join(raw_dir, "sub-HS*"))
        if os.path.isdir(p)
    )
    mdds_folders = sorted(
        p for p in glob.glob(os.path.join(raw_dir, "sub-MDDS*"))
        if os.path.isdir(p)
    )
    files = _find_eeg_files(raw_dir) if os.path.isdir(raw_dir) else []
    p300 = glob.glob(os.path.join(raw_dir, "sub-*", "eeg", "*task-p300*.edf"))

    summary = {
        "raw_dir": raw_dir,
        "hs_folders": len(hs_folders),
        "mdds_folders": len(mdds_folders),
        "eligible_resting_edf": len(files),
        "p300_edf": len(p300),
        "documentation_note": (
            "Observed BIDS folders: {hs} HS + {mdds} MDDS. "
            "Source documentation describes 34 participants. "
            "Do not equate folder count with unique clinical participants."
        ).format(hs=len(hs_folders), mdds=len(mdds_folders)),
    }
    logger.info("NEMAR inspect_raw_dir: %s", summary)
    return summary


def _find_eeg_files(raw_dir):
    """Find eligible NEMAR EDF recordings.

    Only eyesClosed and eyesOpen recordings are included.
    P300 recordings are explicitly excluded.
    """
    pattern = os.path.join(raw_dir, "sub-*", "eeg", "*.edf")
    all_files = sorted(glob.glob(pattern))

    eligible = []
    excluded_p300 = 0

    for path in all_files:
        name = os.path.basename(path).lower()

        if "task-p300" in name:
            excluded_p300 += 1
            continue

        if "task-eyesclosed" in name or "task-eyesopen" in name:
            eligible.append(path)

    logger.info(
        "NEMAR depression discovery: total EDF=%d, "
        "eligible resting-state=%d, P300 excluded=%d",
        len(all_files),
        len(eligible),
        excluded_p300,
    )

    return eligible


def _get_subject_and_label(file_path):
    """Extract subject ID and diagnostic label from NEMAR folder name."""
    subject_dir = os.path.basename(
        os.path.dirname(os.path.dirname(file_path))
    )

    if subject_dir.startswith("sub-"):
        subject_id = subject_dir[4:]
    else:
        subject_id = subject_dir

    if subject_id.startswith("HS"):
        label = LABEL_HEALTHY
    elif subject_id.startswith("MDDS"):
        label = LABEL_DEPRESSION
    else:
        logger.warning(
            "Skipping file with unknown NEMAR subject prefix: %s",
            file_path,
        )
        return None, None

    return subject_id, label


def _get_recording_id(file_path, subject_id, label):
    """Create a stable recording identifier."""
    filename = os.path.splitext(os.path.basename(file_path))[0]

    # Remove the BIDS subject prefix from the filename when possible.
    prefix = f"sub-{subject_id}_"
    if filename.startswith(prefix):
        filename = filename[len(prefix):]

    return f"{subject_id}_{label}_{filename}"


def _load_eeg(file_path):
    """Load an EDF recording with MNE."""
    if not HAS_MNE:
        logger.error("MNE is not installed.")
        return None

    try:
        return mne.io.read_raw_edf(
            file_path,
            preload=True,
            verbose=False,
        )
    except Exception as e:
        logger.error("Failed to load %s: %s", file_path, e)
        return None


def _apply_filters(raw):
    """Apply project-standard filtering and resampling."""
    srate = raw.info["sfreq"]

    notch = [f for f in (50.0, 60.0) if f < srate / 2]
    if notch:
        raw.notch_filter(notch, verbose=False)

    raw.filter(
        L_FREQ,
        H_FREQ,
        fir_design="firwin",
        verbose=False,
    )

    if srate != TARGET_SRATE:
        raw.resample(TARGET_SRATE, verbose=False)

    return raw


def process_recording(
    file_path,
    raw_dir,
    output_dir,
    max_windows=None,
):
    """Process one NEMAR resting-state EDF recording."""
    subject_id, label = _get_subject_and_label(file_path)

    if subject_id is None:
        return []

    raw = _load_eeg(file_path)
    if raw is None:
        return []

    logger.info(
        "[%s] sfreq=%.1f Hz, ch=%d, dur=%.1fs, label=%s, file=%s",
        subject_id,
        raw.info["sfreq"],
        len(raw.ch_names),
        raw.times[-1] if len(raw.times) else 0.0,
        label,
        os.path.basename(file_path),
    )

    try:
        raw = _apply_filters(raw)
    except Exception as e:
        logger.error(
            "Filtering/resampling failed for %s: %s",
            file_path,
            e,
        )
        del raw
        return []

    try:
        data, channels_found = select_and_reorder_channels(raw)
    except Exception as e:
        logger.error(
            "Channel harmonization failed for %s: %s",
            file_path,
            e,
        )
        del raw
        return []

    del raw

    if len(channels_found) < N_CHANNELS // 2:
        logger.warning(
            "[%s] Only %d harmonized channels found; skipping.",
            subject_id,
            len(channels_found),
        )
        return []

    try:
        windows, time_ranges = segment_continuous(
            data,
            srate=TARGET_SRATE,
        )
    except Exception as e:
        logger.error(
            "Segmentation failed for %s: %s",
            file_path,
            e,
        )
        return []

    if len(windows) == 0:
        logger.warning("No windows created for %s", file_path)
        return []

    limit = max_windows or MAX_WINDOWS_PER_FILE
    windows = windows[:limit]
    time_ranges = time_ranges[:limit]

    windows = normalize_windows(windows)

    os.makedirs(output_dir, exist_ok=True)

    recording_id = _get_recording_id(
        file_path,
        subject_id,
        label,
    )

    all_meta = []

    for i, (ws, we) in enumerate(time_ranges):
        fname = f"{recording_id}_{i:06d}.npy"

        np.save(
            os.path.join(output_dir, fname),
            windows[i],
        )

        all_meta.append(
            WindowMetadata(
                dataset="modma",
                subject_id=subject_id,
                recording_id=recording_id,
                label=label,
                window_idx=i,
                window_start=ws,
                window_end=we,
                n_channels=N_CHANNELS,
                srate=TARGET_SRATE,
                n_channels_found=len(channels_found),
                file_path=fname,
            )
        )

    logger.info(
        "[%s] Created %d '%s' windows.",
        recording_id,
        len(all_meta),
        label,
    )

    return all_meta


def process_dataset(
    raw_dir=None,
    output_dir=None,
    max_subjects=None,
    max_windows_per_recording=None,
):
    """Process NEMAR nm000114 healthy + depression EEG recordings."""

    if raw_dir is None:
        raw_dir = DATASET_PATHS["modma"]

    if output_dir is None:
        output_dir = DATASET_OUTPUT["modma"]

    if not os.path.isdir(raw_dir):
        logger.error(
            "NEMAR depression raw directory not found: %s",
            raw_dir,
        )
        return []

    files = _find_eeg_files(raw_dir)

    if not files:
        logger.error(
            "No eligible NEMAR eyesClosed/eyesOpen EDF recordings found."
        )
        return []

    if max_subjects is not None:
        subject_ids = sorted(
            {
                _get_subject_and_label(path)[0]
                for path in files
            }
        )
        subject_ids = [
            s for s in subject_ids
            if s is not None
        ][:max_subjects]

        files = [
            path
            for path in files
            if _get_subject_and_label(path)[0] in subject_ids
        ]

    n_healthy = sum(
        1
        for path in files
        if _get_subject_and_label(path)[1] == LABEL_HEALTHY
    )

    n_depression = sum(
        1
        for path in files
        if _get_subject_and_label(path)[1] == LABEL_DEPRESSION
    )

    logger.info(
        "Processing NEMAR recordings: healthy=%d, depression=%d",
        n_healthy,
        n_depression,
    )

    os.makedirs(output_dir, exist_ok=True)

    all_metadata = []

    for file_path in files:
        try:
            metadata = process_recording(
                file_path,
                raw_dir,
                output_dir,
                max_windows=max_windows_per_recording,
            )
            all_metadata.extend(metadata)
        except Exception as e:
            logger.error(
                "Failed recording %s: %s",
                file_path,
                e,
            )

    if all_metadata:
        metadata_path = os.path.join(
            output_dir,
            "metadata.csv",
        )
        save_metadata(all_metadata, metadata_path)

        n_hc_windows = sum(
            1 for m in all_metadata
            if m.label == LABEL_HEALTHY
        )

        n_dep_windows = sum(
            1 for m in all_metadata
            if m.label == LABEL_DEPRESSION
        )

        logger.info(
            "NEMAR depression preprocessing complete: "
            "healthy=%d windows, depression=%d windows, total=%d",
            n_hc_windows,
            n_dep_windows,
            len(all_metadata),
        )

    return all_metadata


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s - %(levelname)s - %(message)s",
    )

    meta = process_dataset(max_subjects=2, max_windows_per_recording=10)
    print(f"Created {len(meta)} windows")