"""
EEG segmentation: split continuous EEG into fixed-size windows.
"""

import numpy as np
import logging

from . import config

logger = logging.getLogger(__name__)


def segment_continuous(data, window_sec=None, overlap=None, srate=None):
    """Segment continuous EEG data into overlapping windows.

    Args:
        data: numpy array [N_channels, N_samples]
        window_sec: window duration in seconds
        overlap: fraction overlap (0-1)
        srate: sampling rate in Hz

    Returns:
        windows: numpy array [N_windows, N_channels, N_samples_per_window]
        time_ranges: list of (start_sec, end_sec) tuples
    """
    if window_sec is None:
        window_sec = config.WINDOW_SEC
    if overlap is None:
        overlap = config.OVERLAP
    if srate is None:
        srate = config.TARGET_SRATE

    n_channels, total_samples = data.shape
    window_samples = int(window_sec * srate)
    step_samples = int(window_samples * (1 - overlap))

    if total_samples < window_samples:
        logger.warning(f"Data too short ({total_samples} samples) for "
                      f"window ({window_samples} samples). Skipping.")
        return np.array([]), []

    windows = []
    time_ranges = []

    start = 0
    while start + window_samples <= total_samples:
        window = data[:, start:start + window_samples]
        windows.append(window)

        start_sec = start / srate
        end_sec = (start + window_samples) / srate
        time_ranges.append((start_sec, end_sec))

        start += step_samples

    windows = np.array(windows, dtype=np.float32)
    logger.info(f"Segmented: {windows.shape[0]} windows, "
                f"shape={windows.shape}")
    return windows, time_ranges


def segment_labeled(data, label_intervals, label_name, srate=None,
                    window_sec=None, overlap=None):
    """Segment specific labeled intervals from continuous data.

    Args:
        data: numpy array [N_channels, N_samples]
        label_intervals: list of (start_sec, end_sec) for this label
        label_name: string label
        srate: sampling rate
        window_sec: window duration
        overlap: overlap fraction

    Returns:
        windows: numpy array [N_windows, N_channels, N_samples_per_window]
        time_ranges: list of (start_sec, end_sec)
    """
    if srate is None:
        srate = config.TARGET_SRATE
    if window_sec is None:
        window_sec = config.WINDOW_SEC
    if overlap is None:
        overlap = config.OVERLAP

    window_samples = int(window_sec * srate)
    step_samples = int(window_samples * (1 - overlap))
    total_samples = data.shape[1]

    all_windows = []
    all_ranges = []

    for start_sec, end_sec in label_intervals:
        seg_start = int(start_sec * srate)
        seg_end = min(int(end_sec * srate), total_samples)

        if seg_end - seg_start < window_samples:
            # Interval too short for a full window, skip
            continue

        pos = seg_start
        while pos + window_samples <= seg_end:
            window = data[:, pos:pos + window_samples]
            all_windows.append(window)

            w_start = pos / srate
            w_end = (pos + window_samples) / srate
            all_ranges.append((w_start, w_end))

            pos += step_samples

    if all_windows:
        windows = np.array(all_windows, dtype=np.float32)
        logger.info(f"Segmented '{label_name}': {len(all_windows)} windows")
    else:
        windows = np.array([], dtype=np.float32)
        logger.warning(f"No windows created for label '{label_name}'")

    return windows, all_ranges
