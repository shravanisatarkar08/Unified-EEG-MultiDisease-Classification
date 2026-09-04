"""
EEG harmonization: apply common preprocessing chain to bring
different datasets into a common representation.
"""

import mne
import logging

from . import config
from .common import resample, bandpass_filter, notch_filter

logger = logging.getLogger(__name__)


def harmonize_raw(raw, dataset_name=None):
    """Apply standard preprocessing chain to a raw EEG recording.

    Steps:
        1. Set EEG channel types
        2. Notch filter (50/60 Hz)
        3. Bandpass filter (0.5-45 Hz)
        4. Resample to TARGET_SRATE

    Args:
        raw: MNE Raw object (modified in-place)
        dataset_name: optional string for logging

    Returns:
        MNE Raw object (same object, modified)
    """
    prefix = f"[{dataset_name}] " if dataset_name else ""
    logger.info(f"{prefix}Harmonizing: srate={raw.info['sfreq']}Hz, "
                f"n_channels={len(raw.ch_names)}, "
                f"duration={raw.times[-1]:.1f}s")

    # Load data into memory if not already
    raw.load_data()

    # Apply filters
    notch_filter(raw)
    bandpass_filter(raw)

    # Resample
    resample(raw)

    logger.info(f"{prefix}Harmonized: srate={raw.info['sfreq']}Hz, "
                f"duration={raw.times[-1]:.1f}s")

    return raw
