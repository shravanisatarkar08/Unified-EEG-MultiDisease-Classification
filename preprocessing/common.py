"""
Common EEG preprocessing operations using MNE-Python.
"""

import numpy as np
import mne
import logging

from . import config

logger = logging.getLogger(__name__)


def resample(raw, target_srate=None):
    """Resample raw EEG to target sampling rate."""
    if target_srate is None:
        target_srate = config.TARGET_SRATE
    if raw.info['sfreq'] != target_srate:
        logger.info(f"Resampling from {raw.info['sfreq']} Hz to {target_srate} Hz")
        raw.resample(target_srate)
    return raw


def bandpass_filter(raw, l_freq=None, h_freq=None):
    """Apply bandpass filter."""
    if l_freq is None:
        l_freq = config.L_FREQ
    if h_freq is None:
        h_freq = config.H_FREQ
    logger.info(f"Bandpass filtering: {l_freq}-{h_freq} Hz")
    raw.filter(l_freq, h_freq, fir_design='firwin', verbose=False)
    return raw


def notch_filter(raw, freqs=None):
    """Apply notch filter to remove power line noise."""
    if freqs is None:
        freqs = config.NOTCH_FREQS
    # Only notch filter if srate is high enough (>2x notch freq)
    srate = raw.info['sfreq']
    valid_freqs = [f for f in freqs if f < srate / 2]
    if valid_freqs:
        logger.info(f"Notch filtering at: {valid_freqs} Hz")
        raw.notch_filter(valid_freqs, verbose=False)
    return raw


def normalize_channel_name(ch_name):
    """Normalize a channel name to our standard naming.

    Strips common prefixes like 'EEG ', '-REF', '-LE', etc.
    Then maps aliases (T7->T3, FP1->Fp1, etc.)
    """
    name = ch_name.strip()

    # Remove common prefixes
    for prefix in ['EEG ', 'EEG-', 'eeg ', 'eeg-']:
        if name.startswith(prefix):
            name = name[len(prefix):]

    # Remove common suffixes
    for suffix in ['-REF', '-LE', '-Ref', '-ref', '-AVG', '-Avg']:
        if name.endswith(suffix):
            name = name[:-len(suffix)]

    name = name.strip()

    # Check aliases
    if name in config.CHANNEL_ALIASES:
        return config.CHANNEL_ALIASES[name]

    # Check if it matches a standard channel (case-sensitive)
    if name in config.COMMON_CHANNELS:
        return name

    # Try case-insensitive match
    name_lower = name.lower()
    for std_ch in config.COMMON_CHANNELS:
        if std_ch.lower() == name_lower:
            return std_ch

    # Check aliases case-insensitive
    for alias, std in config.CHANNEL_ALIASES.items():
        if alias.lower() == name_lower:
            return std

    return None  # Channel not in our standard set


def select_and_reorder_channels(raw, common_channels=None):
    """Select and reorder channels to match the common channel set.

    Missing channels are zero-filled.

    Args:
        raw: MNE Raw object
        common_channels: list of target channel names

    Returns:
        numpy array [N_channels, N_samples], list of channel names present
    """
    if common_channels is None:
        common_channels = config.COMMON_CHANNELS

    data = raw.get_data()  # [n_ch, n_samples]
    n_samples = data.shape[1]
    ch_names = raw.ch_names

    # Build mapping from raw channels to standard channels
    raw_to_std = {}
    for i, ch in enumerate(ch_names):
        std_name = normalize_channel_name(ch)
        if std_name is not None:
            raw_to_std[std_name] = i

    # Build output array
    output = np.zeros((len(common_channels), n_samples), dtype=np.float64)
    channels_found = []

    for j, std_ch in enumerate(common_channels):
        if std_ch in raw_to_std:
            output[j] = data[raw_to_std[std_ch]]
            channels_found.append(std_ch)

    n_found = len(channels_found)
    n_missing = len(common_channels) - n_found
    if n_missing > 0:
        logger.warning(f"Missing {n_missing}/{len(common_channels)} channels. "
                      f"Found: {channels_found}")

    return output, channels_found


def normalize_windows(windows):
    """Z-score normalize each channel within each window.

    Args:
        windows: numpy array [N_windows, N_channels, N_samples]

    Returns:
        normalized array, same shape
    """
    # Per-channel, per-window normalization
    mean = windows.mean(axis=2, keepdims=True)
    std = windows.std(axis=2, keepdims=True)
    std[std < 1e-8] = 1e-8  # avoid division by zero
    return (windows - mean) / std
