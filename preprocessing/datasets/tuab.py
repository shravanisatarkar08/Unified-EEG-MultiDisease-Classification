"""
TUAB / TUH EEG Corpus dataset stub.

STATUS: UNAVAILABLE
-------------------
The Temple University Hospital Abnormal EEG (TUAB) corpus requires
institutional registration and access approval from Temple University.
It is NOT available locally and is NOT downloaded during this prototype.

This module documents the dataset's status and provides the interface
for future integration once access credentials are obtained.

To integrate TUAB when access is granted:
  1. Download the corpus from https://isip.piconepress.com/projects/tuh_eeg/
  2. Place data under datasets/raw/tuab/
  3. Implement _read_labels() to parse TUAB label files (normal/abnormal)
  4. Implement process_dataset() following the same pattern as chbmit.py
"""

import os
import logging

logger = logging.getLogger(__name__)

STATUS = "UNAVAILABLE"
REASON = (
    "TUAB/TUH Corpus requires institutional registration with Temple University. "
    "Dataset is NOT available locally. Skipped in prototype."
)
ACCESS_URL = "https://isip.piconepress.com/projects/tuh_eeg/"


def is_available(raw_dir=None):
    """Return True if TUAB data is accessible locally."""
    from ..config import DATASET_PATHS
    if raw_dir is None:
        raw_dir = DATASET_PATHS.get('tuab', '')
    if not os.path.isdir(raw_dir):
        return False
    # Simple heuristic: check for .edf files
    import glob
    edfs = glob.glob(os.path.join(raw_dir, '**', '*.edf'), recursive=True)
    return len(edfs) > 0


def process_dataset(raw_dir=None, output_dir=None, **kwargs):
    """Stub: document TUAB as unavailable and return empty list.

    Returns
    -------
    list  (always empty — TUAB not available)
    """
    if is_available(raw_dir):
        logger.warning(
            "TUAB data detected locally but full preprocessing not yet "
            "implemented. Please implement process_dataset() in tuab.py."
        )
    else:
        logger.warning(
            f"TUAB: {REASON}\n"
            f"Access URL: {ACCESS_URL}"
        )
    return []


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(name)s - %(levelname)s - %(message)s')
    print(f"TUAB status: {STATUS}")
    print(f"Reason: {REASON}")
    print(f"Access: {ACCESS_URL}")
    print(f"Locally available: {is_available()}")
