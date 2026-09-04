"""
Metadata tracking for EEG preprocessing pipeline.
"""

import os
import csv
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class WindowMetadata:
    """Metadata for a single EEG window."""
    dataset: str
    subject_id: str
    recording_id: str
    label: str
    window_idx: int
    window_start: float
    window_end: float
    n_channels: int
    srate: float
    n_channels_found: int = 0
    file_path: str = ''


METADATA_FIELDS = [
    'dataset', 'subject_id', 'recording_id', 'label',
    'window_idx', 'window_start', 'window_end',
    'n_channels', 'srate', 'n_channels_found', 'file_path'
]


def save_metadata(metadata_list, output_path):
    """Save list of WindowMetadata to CSV file.

    Args:
        metadata_list: list of WindowMetadata objects
        output_path: path to output CSV file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        for meta in metadata_list:
            writer.writerow(asdict(meta))

    logger.info(f"Saved {len(metadata_list)} metadata entries to {output_path}")


def append_metadata(metadata_list, output_path):
    """Append metadata entries to existing CSV file.

    Creates file with header if it doesn't exist.
    """
    file_exists = os.path.exists(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        if not file_exists:
            writer.writeheader()
        for meta in metadata_list:
            writer.writerow(asdict(meta))


def load_metadata(path):
    """Load metadata from CSV file.

    Returns:
        list of dicts
    """
    entries = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(row)
    logger.info(f"Loaded {len(entries)} metadata entries from {path}")
    return entries
