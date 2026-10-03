"""
SRM Resting-State EEG healthy dataset loader.

Dataset:
    OpenNeuro ds003775 (local path: datasets/raw/healthy/ds003775)

Purpose:
    Preprocess healthy resting-state EEG into the common 19-channel,
    256 Hz, 5-second, 50%-overlap window representation used by the project.
"""

import os
import csv
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
TASK_TOKEN = "task-resteyesc"


def _find_eeg_files(raw_dir):
    """Find BIDS ds003775 eyes-closed resting-state EDF recordings only."""
    search_roots = [raw_dir]
    nested = os.path.join(raw_dir, "ds003775")
    if os.path.isdir(nested):
        search_roots.append(nested)

    files = []
    for root in search_roots:
        pattern = os.path.join(root, "**", f"*{TASK_TOKEN}_eeg.edf")
        files.extend(glob.glob(pattern, recursive=True))

    unique = []
    seen = set()
    for path in sorted(files):
        if TASK_TOKEN not in os.path.basename(path).lower():
            continue
        norm = os.path.normpath(path)
        if norm not in seen:
            seen.add(norm)
            unique.append(norm)

    logger.info("Found %d SRM %s EEG recordings", len(unique), TASK_TOKEN)
    return unique


def _get_subject_id(file_path):
    """Extract BIDS subject ID from the recording path."""
    parts = os.path.normpath(file_path).split(os.sep)
    for part in parts:
        if part.startswith("sub-"):
            return part
    return "unknown"


def _load_eeg(file_path):
    """Load an EDF recording with MNE."""
    if not HAS_MNE:
        logger.error("MNE is not installed.")
        return None
    try:
        return mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    except Exception as e:
        logger.error("Failed to load %s: %s", file_path, e)
        return None


def _apply_filters(raw):
    """Apply project-standard filtering and resampling."""
    srate = raw.info["sfreq"]
    notch = [f for f in (50.0, 60.0) if f < srate / 2]
    if notch:
        raw.notch_filter(notch, verbose=False)
    raw.filter(L_FREQ, H_FREQ, fir_design="firwin", verbose=False)
    if srate != TARGET_SRATE:
        raw.resample(TARGET_SRATE, verbose=False)
    return raw


def _write_failures(failures, output_dir):
    """Persist failed-recording records so skips are not silent."""
    if not failures:
        return
    fail_path = os.path.join(output_dir, "failed_recordings.csv")
    with open(fail_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_path", "subject_id", "reason"])
        writer.writeheader()
        writer.writerows(failures)
    logger.warning("Wrote %d failed SRM recordings to %s", len(failures), fail_path)


def process_dataset(
    raw_dir=None,
    output_dir=None,
    max_subjects=None,
    max_windows_per_recording=None,
):
    """
    Process SRM healthy resting-state EEG.

    Args:
        raw_dir: Dataset root directory (datasets/raw/healthy).
        output_dir: Processed output directory.
        max_subjects: Optional limit for testing.
        max_windows_per_recording: Optional window limit for testing.

    Returns:
        List of WindowMetadata objects.
    """
    raw_dir = raw_dir or DATASET_PATHS["healthy"]
    output_dir = output_dir or DATASET_OUTPUT["healthy"]

    if not os.path.isdir(raw_dir):
        logger.error("Healthy raw directory not found: %s", raw_dir)
        return []

    eeg_files = _find_eeg_files(raw_dir)
    if not eeg_files:
        logger.error("No ds003775 %s EDF recordings found under %s", TASK_TOKEN, raw_dir)
        return []

    if max_subjects is not None:
        subjects = sorted(set(_get_subject_id(path) for path in eeg_files))[:max_subjects]
        eeg_files = [
            path for path in eeg_files
            if _get_subject_id(path) in subjects
        ]

    logger.info(
        "Processing %d SRM recordings from %d subjects",
        len(eeg_files),
        len(set(_get_subject_id(path) for path in eeg_files)),
    )

    os.makedirs(output_dir, exist_ok=True)
    all_metadata = []
    failures = []

    for file_path in eeg_files:
        subject_id = _get_subject_id(file_path)
        recording_id = os.path.splitext(os.path.basename(file_path))[0]
        try:
            raw = _load_eeg(file_path)
            if raw is None:
                failures.append({
                    "file_path": file_path,
                    "subject_id": subject_id,
                    "reason": "mne_load_failed",
                })
                continue

            logger.info(
                "[%s] sfreq=%.1f Hz, ch=%d, dur=%.1fs, file=%s",
                subject_id,
                raw.info["sfreq"],
                len(raw.ch_names),
                raw.times[-1] if len(raw.times) else 0.0,
                os.path.basename(file_path),
            )

            try:
                raw = _apply_filters(raw)
            except Exception as e:
                logger.error("Filtering/resampling failed for %s: %s", file_path, e)
                failures.append({
                    "file_path": file_path,
                    "subject_id": subject_id,
                    "reason": f"filter_failed: {e}",
                })
                del raw
                continue

            data, channels_found = select_and_reorder_channels(raw)
            del raw

            if len(channels_found) < N_CHANNELS // 2:
                logger.warning(
                    "[%s] Only %d harmonized channels found; skipping.",
                    subject_id,
                    len(channels_found),
                )
                failures.append({
                    "file_path": file_path,
                    "subject_id": subject_id,
                    "reason": f"too_few_channels:{len(channels_found)}",
                })
                continue

            windows, time_ranges = segment_continuous(data, srate=TARGET_SRATE)
            if len(windows) == 0:
                logger.warning("No windows created for %s", file_path)
                failures.append({
                    "file_path": file_path,
                    "subject_id": subject_id,
                    "reason": "no_windows",
                })
                continue

            limit = max_windows_per_recording or MAX_WINDOWS_PER_FILE
            windows = windows[:limit]
            time_ranges = time_ranges[:limit]
            windows = normalize_windows(windows)

            for i, (ws, we) in enumerate(time_ranges):
                fname = f"{subject_id}_{recording_id}_{i:06d}.npy"
                np.save(os.path.join(output_dir, fname), windows[i])
                all_metadata.append(
                    WindowMetadata(
                        dataset="srm_healthy",
                        subject_id=subject_id,
                        recording_id=recording_id,
                        label="healthy",
                        window_idx=i,
                        window_start=ws,
                        window_end=we,
                        n_channels=N_CHANNELS,
                        srate=TARGET_SRATE,
                        n_channels_found=len(channels_found),
                        file_path=fname,
                    )
                )

            logger.info("Processed %s: %d windows", recording_id, len(windows))

        except Exception as exc:
            logger.exception("Failed to process %s: %s", file_path, exc)
            failures.append({
                "file_path": file_path,
                "subject_id": subject_id,
                "reason": str(exc),
            })

    _write_failures(failures, output_dir)

    if all_metadata:
        save_metadata(all_metadata, os.path.join(output_dir, "metadata.csv"))

    logger.info(
        "SRM processing complete: %d windows, %d failed recordings",
        len(all_metadata),
        len(failures),
    )
    return all_metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
    meta = process_dataset(max_subjects=1, max_windows_per_recording=4)
    print(f"Created {len(meta)} test windows.")
