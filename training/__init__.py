"""
training/__init__.py - Training package for Unified EEG Multi-Disease Classification.

Exposes:
    EEGDataset          - PyTorch Dataset (lazy EEG loading from index CSV)
    get_dataloaders     - Returns train / val / test DataLoaders
    CLASS_TO_IDX        - Canonical class -> integer label mapping (mirrors dataset_split)
    IDX_TO_CLASS        - Inverse label mapping
    EEGTrainer          - Training loop with AdamW + cosine LR + early stopping
    TrainConfig         - Dataclass of all training hyperparameters
    evaluate_model      - Per-class metrics, confusion matrix, F1 scores
    EEGExplainer        - Gradient-based attribution maps (saliency, IG, attention)
    extract_cnn_features - CNN embedding extraction pipeline (saves NPZ)
"""

from training.dataset import EEGDataset, get_dataloaders, CLASS_TO_IDX, IDX_TO_CLASS

# Lazy imports to avoid import errors if optional dependencies are missing
try:
    from training.trainer import EEGTrainer, TrainConfig
    from training.evaluator import evaluate_model
    from training.explainability import EEGExplainer
    from training.feature_extraction import extract_cnn_features
except ImportError:
    pass

__all__ = [
    "EEGDataset",
    "get_dataloaders",
    "CLASS_TO_IDX",
    "IDX_TO_CLASS",
    "EEGTrainer",
    "TrainConfig",
    "evaluate_model",
    "EEGExplainer",
    "extract_cnn_features",
]
