"""
training/__init__.py - Training package for Unified EEG Multi-Disease Classification.

Exposes:
    EEGDataset       - PyTorch Dataset (lazy EEG loading from index CSV)
    get_dataloaders  - Returns train / val / test DataLoaders
    CLASS_TO_IDX     - Canonical class → integer label mapping (mirrors dataset_split)
    IDX_TO_CLASS     - Inverse label mapping
"""

from training.dataset import EEGDataset, get_dataloaders, CLASS_TO_IDX, IDX_TO_CLASS

__all__ = [
    "EEGDataset",
    "get_dataloaders",
    "CLASS_TO_IDX",
    "IDX_TO_CLASS",
]
