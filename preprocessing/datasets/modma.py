"""
MODMA depression EEG dataset handler.
Dataset: MODMA (Multi-modal Open Dataset for Mental-disorder Analysis)

STATUS: DATA ACCESS BLOCKER
-------------------------------
The MODMA dataset is distributed as split archive files (e.g., .zip.001,
.zip.002, ...). Reconstruction and extraction requires:
  1. All archive parts present locally.
  2. Microsoft Visual C++ build tools (for pyEDFlib extraction on Windows).
  3. OR a Linux environment with unzip/7z available.

This module:
  - Inspects the archive directory without modifying or deleting files.
  - Reports whether extraction is feasible.
  - Documents the blocker clearly if it cannot proceed.

To use MODMA when the data IS available and extracted:
  - Place .mat or .edf files under datasets/raw/depression/
  - A participants CSV/JSON mapping subject ID -> group (depression/healthy)
    must also be present.

DO NOT: rename, delete, concatenate, or otherwise modify split archive files.
"""

import os
import glob
import logging

logger = logging.getLogger(__name__)

BLOCKER_MSG = (
    "MODMA DATASET: DATA ACCESS BLOCKER\n"
    "  Reason: MODMA is distributed as split archives (.zip.001, .zip.002, ...).\n"
    "  Extraction requires Microsoft Visual C++ build tools on Windows (for\n"
    "  pyEDFlib) AND all archive parts present.\n"
    "  The split archives will NOT be modified.\n"
    "  Action required: Extract MODMA archives in a Linux environment,\n"
    "  then place extracted .mat/.edf files under datasets/raw/depression/.\n"
    "  Until then, MODMA preprocessing is skipped."
)


def inspect_raw_dir(raw_dir):
    """Inspect MODMA raw directory without modifying any files.

    Returns a status dict describing what was found.
    """
    status = {
        'raw_dir': raw_dir,
        'exists': False,
        'split_archives': [],
        'edf_files': [],
        'mat_files': [],
        'can_proceed': False,
        'blocker': None,
    }

    if not os.path.isdir(raw_dir):
        status['blocker'] = f"Directory not found: {raw_dir}"
        return status

    status['exists'] = True

    # Look for split archives (do NOT open or modify them)
    for pattern in ['*.zip.*', '*.part*', '*.rar', '*.7z']:
        found = glob.glob(os.path.join(raw_dir, '**', pattern), recursive=True)
        status['split_archives'].extend(found)

    # Look for already-extracted EEG files
    status['edf_files'] = glob.glob(os.path.join(raw_dir, '**', '*.edf'),
                                    recursive=True)
    status['mat_files'] = glob.glob(os.path.join(raw_dir, '**', '*.mat'),
                                    recursive=True)

    n_edf = len(status['edf_files'])
    n_mat = len(status['mat_files'])
    n_arc = len(status['split_archives'])

    if n_edf > 0 or n_mat > 0:
        status['can_proceed'] = True
        logger.info(f"MODMA: Found {n_edf} EDF and {n_mat} MAT files — "
                    "preprocessing may be attempted.")
    elif n_arc > 0:
        status['blocker'] = (
            f"MODMA: Found {n_arc} split archive file(s) but no extracted EEG "
            "files. Archives will NOT be modified. "
            "Please extract on Linux and place files in datasets/raw/depression/."
        )
        logger.warning(status['blocker'])
    else:
        status['blocker'] = (
            "MODMA: No EEG files or archives found in "
            f"{raw_dir}. Dataset unavailable."
        )
        logger.warning(status['blocker'])

    return status


def process_dataset(raw_dir=None, output_dir=None, **kwargs):
    """Attempt MODMA preprocessing; document blocker if data unavailable.

    Returns
    -------
    list of WindowMetadata (empty if data unavailable)
    """
    from ..config import DATASET_PATHS, DATASET_OUTPUT
    if raw_dir is None:
        raw_dir = DATASET_PATHS['modma']
    if output_dir is None:
        output_dir = DATASET_OUTPUT['modma']

    status = inspect_raw_dir(raw_dir)

    if status['blocker']:
        logger.error("MODMA preprocessing BLOCKED:\n" + status['blocker'])
        print("\n" + BLOCKER_MSG + "\n")
        return []

    if not status['can_proceed']:
        return []

    # If extracted files are present, attempt basic processing
    logger.info("MODMA extracted files found — attempting preprocessing...")
    logger.warning(
        "MODMA preprocessing is not fully implemented. "
        "A participants label file (participants.tsv or labels.csv) "
        "mapping subject IDs to depression/healthy labels is required."
    )
    return []


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(name)s - %(levelname)s - %(message)s')
    from preprocessing.config import DATASET_PATHS
    status = inspect_raw_dir(DATASET_PATHS.get('modma', 'datasets/raw/depression'))
    print("MODMA status:", status)
