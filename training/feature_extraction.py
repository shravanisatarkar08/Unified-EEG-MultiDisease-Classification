"""
training/feature_extraction.py - CNN Feature Extraction Pipeline.

Extracts CNN embeddings from EEGFeatureExtractor for all windows in
the model dataset index. Provenance is preserved alongside embeddings.

Design
------
* Reads model_dataset_index.csv to discover windows.
* Uses EEGDataset (lazy-load) to fetch normalised EEG tensors.
* Runs EEGFeatureExtractor.extract_features() -> [B, N, D] tokens.
* Saves each split as a compressed NPZ with fields:
    - embeddings:       float32 [n_windows, N, D]
    - labels:           int64   [n_windows]
    - dataset_names, subject_ids, window_indices, class_categories, source_files
* If raw EEG is absent (Step 3 blocker), EEGDataset returns zero-fill
  tensors; embeddings are still extracted and saved for structural tests.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Path bootstrap
_MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_DEFAULT = _MODULE_DIR.parent
if str(PROJECT_ROOT_DEFAULT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_DEFAULT))

from models.eeg_cnn import EEGFeatureExtractor
from training.dataset import EEGDataset, _eeg_collate_fn

logger = logging.getLogger(__name__)

DEFAULT_INDEX_CSV = PROJECT_ROOT_DEFAULT / "datasets" / "metadata" / "model_dataset_index.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT_DEFAULT / "datasets" / "features"

# CNN hyperparameters - must match training config
CNN_N_CHANNELS = 19
CNN_N_SAMPLES  = 1280   # 5 s @ 256 Hz
CNN_EMBED_DIM  = 64
CNN_F1         = 16
CNN_D          = 2
CNN_F2         = 32
CNN_DROPOUT    = 0.0    # deterministic extraction


class ExtractionResult:
    """Accumulates embeddings and provenance for one split."""

    def __init__(self):
        self.embeddings:       List[np.ndarray] = []
        self.labels:           List[int]        = []
        self.dataset_names:    List[str]        = []
        self.subject_ids:      List[str]        = []
        self.class_categories: List[str]        = []
        self.window_indices:   List[int]        = []
        self.source_files:     List[str]        = []

    def append_batch(self, feats: np.ndarray, labels: List[int],
                     metas: List[dict]) -> None:
        """Append one mini-batch of results."""
        for i, (feat, label, meta) in enumerate(zip(feats, labels, metas)):
            self.embeddings.append(feat)
            self.labels.append(label)
            self.dataset_names.append(str(meta.get("dataset_name", "")))
            self.subject_ids.append(str(meta.get("subject_id", "")))
            self.class_categories.append(str(meta.get("class_category", "")))
            self.window_indices.append(int(meta.get("window_idx", i)))
            self.source_files.append(str(meta.get("source_file", "")))

    def to_numpy_dict(self) -> dict:
        return {
            "embeddings":       np.stack(self.embeddings, axis=0).astype(np.float32),
            "labels":           np.array(self.labels, dtype=np.int64),
            "dataset_names":    np.array(self.dataset_names),
            "subject_ids":      np.array(self.subject_ids),
            "class_categories": np.array(self.class_categories),
            "window_indices":   np.array(self.window_indices, dtype=np.int64),
            "source_files":     np.array(self.source_files),
        }

    def __len__(self) -> int:
        return len(self.labels)


def extract_cnn_features(
    index_csv: str = str(DEFAULT_INDEX_CSV),
    project_root: str = str(PROJECT_ROOT_DEFAULT),
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    model_weights: Optional[str] = None,
    num_workers: int = 0,
) -> Dict[str, Path]:
    """Extract CNN features for all splits and save as NPZ files.

    Parameters
    ----------
    index_csv     : path to model_dataset_index.csv
    project_root  : project root for resolving relative source_file paths
    output_dir    : directory to save NPZ files per split
    batch_size    : DataLoader mini-batch size
    device        : torch device (defaults to cpu)
    model_weights : optional checkpoint path to load CNN weights from
    num_workers   : DataLoader workers (0 = main process, Windows-safe)

    Returns
    -------
    dict mapping split name -> output NPZ path
    """
    index_csv  = Path(index_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build CNN model
    cnn = EEGFeatureExtractor(
        n_channels=CNN_N_CHANNELS,
        n_samples=CNN_N_SAMPLES,
        n_classes=5,
        F1=CNN_F1, D=CNN_D, F2=CNN_F2,
        dropout_rate=CNN_DROPOUT,
        embed_dim=CNN_EMBED_DIM,
    ).to(device)

    if model_weights is not None:
        ckpt      = torch.load(model_weights, map_location=device)
        state     = ckpt.get("model_state_dict", ckpt)
        cnn_state = {k.replace("cnn.", "", 1): v
                     for k, v in state.items() if k.startswith("cnn.")}
        cnn.load_state_dict(cnn_state if cnn_state else state, strict=False)
        logger.info("Loaded CNN weights from %s", model_weights)

    cnn.eval()

    if not index_csv.exists():
        logger.warning("index_csv not found: %s -- skipping extraction.", index_csv)
        return {}

    df = pd.read_csv(index_csv)
    output_paths: Dict[str, Path] = {}

    for split in df["split"].unique().tolist():
        split_df = df[df["split"] == split].reset_index(drop=True)
        logger.info("Extracting features for split=%s (%d windows)", split, len(split_df))

        dataset = EEGDataset(
            index_df=split_df, project_root=str(project_root), cache_size=0
        )
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=_eeg_collate_fn,
        )
        result = ExtractionResult()

        with torch.no_grad():
            for eeg_batch, label_batch, meta_batch in loader:
                eeg_batch   = eeg_batch.to(device)          # [B, 19, 1280]
                feats       = cnn.extract_features(eeg_batch)  # [B, N, D]
                feats_np    = feats.cpu().numpy()
                labels_list = (label_batch.tolist()
                               if hasattr(label_batch, "tolist")
                               else list(label_batch))
                result.append_batch(feats_np, labels_list, meta_batch)

        out_path = output_dir / f"cnn_features_{split}.npz"
        np.savez_compressed(out_path, **result.to_numpy_dict())
        logger.info("Saved %d embeddings -> %s", len(result), out_path)
        output_paths[split] = out_path

    return output_paths


def load_features(npz_path: str) -> dict:
    """Load a saved NPZ feature file and return as a plain dict of arrays."""
    data = np.load(npz_path, allow_pickle=True)
    return {k: data[k] for k in data.files}


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Extract CNN features from EEG windows.")
    parser.add_argument("--index-csv",  default=str(DEFAULT_INDEX_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--weights",    default=None,
                        help="Optional trained CNN/classifier checkpoint.")
    args = parser.parse_args()

    paths = extract_cnn_features(
        index_csv=args.index_csv,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=torch.device(args.device),
        model_weights=args.weights,
    )
    for split, p in paths.items():
        print(f"[{split}] -> {p}")
