"""
Preprocessing configuration for unified EEG pipeline.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset paths
RAW_DIR = os.path.join(PROJECT_ROOT, 'datasets', 'raw')
INTERIM_DIR = os.path.join(PROJECT_ROOT, 'datasets', 'interim')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'datasets', 'processed')

# Target parameters
TARGET_SRATE = 256       # Hz
WINDOW_SEC = 5           # seconds
OVERLAP = 0.5            # 50% overlap
N_CHANNELS = 19          # standard 10-20 montage
WINDOW_SAMPLES = TARGET_SRATE * WINDOW_SEC  # 1280

# Filter settings
L_FREQ = 0.5             # Hz highpass
H_FREQ = 45.0            # Hz lowpass
NOTCH_FREQS = [50.0, 60.0]  # power line frequencies

# 19 standard 10-20 channels (using older T3/T4/T5/T6 naming)
COMMON_CHANNELS = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T3', 'C3', 'Cz', 'C4', 'T4',
    'T5', 'P3', 'Pz', 'P4', 'T6',
    'O1', 'O2'
]

# Mapping from alternate channel names to our standard names
# Handles both old (T3/T4/T5/T6) and new (T7/T8/P7/P8) naming
CHANNEL_ALIASES = {
    # New 10-20 naming -> old naming
    'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6',
    # Case variations
    'FP1': 'Fp1', 'FP2': 'Fp2', 'FZ': 'Fz', 'CZ': 'Cz', 'PZ': 'Pz',
    'fp1': 'Fp1', 'fp2': 'Fp2', 'fz': 'Fz', 'cz': 'Cz', 'pz': 'Pz',
    'f7': 'F7', 'f3': 'F3', 'f4': 'F4', 'f8': 'F8',
    't3': 'T3', 'c3': 'C3', 'c4': 'C4', 't4': 'T4',
    't5': 'T5', 'p3': 'P3', 'p4': 'P4', 't6': 'T6',
    'o1': 'O1', 'o2': 'O2',
    't7': 'T3', 't8': 'T4', 'p7': 'T5', 'p8': 'T6',
}

# Dataset-specific raw directories
DATASET_PATHS = {
    'chbmit': os.path.join(RAW_DIR, 'epilepsy'),
    'alzheimer': os.path.join(RAW_DIR, 'alzheimers'),
    'parkinson': os.path.join(RAW_DIR, 'parkinsons'),
    'modma': os.path.join(RAW_DIR, 'depression'),
    'tuab': os.path.join(RAW_DIR, 'tuab'),
}

# Dataset-specific output directories
DATASET_OUTPUT = {
    'chbmit': os.path.join(PROCESSED_DIR, 'chbmit'),
    'alzheimer': os.path.join(PROCESSED_DIR, 'alzheimer'),
    'parkinson': os.path.join(PROCESSED_DIR, 'parkinson'),
    'modma': os.path.join(PROCESSED_DIR, 'modma'),
    'tuab': os.path.join(PROCESSED_DIR, 'tuab'),
}
