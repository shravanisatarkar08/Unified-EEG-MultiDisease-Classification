"""
scripts/build_new_index.py - Build the unified 5-class model_dataset_index.csv.

Datasets:
  1. SRM Healthy (ds003775) & Healthy Controls from AD, PD, NEMAR → Healthy (Class 0)
  2. CHB-MIT Epilepsy (chb01)                                      → Epilepsy (Class 1)
  3. OpenNeuro ds004504 Alzheimer's                                → Alzheimer's Disease (Class 2)
  4. OpenNeuro ds004584 Parkinson's                                → Parkinson's Disease (Class 3)
  5. NEMAR nm000114 (MODMA) Depression                             → Depression (Class 4)

Subject-level splitting:
  - 70% Train, 15% Validation, 15% Test partitioned deterministically by subject ID.
  - Zero subject leakage across splits:
      train_subjects ∩ val_subjects = ∅
      train_subjects ∩ test_subjects = ∅
      val_subjects ∩ test_subjects = ∅
  - Single-subject datasets (CHB-MIT chb01, SRM sub-010) are allocated to Train
    with limitations explicitly documented.
"""

import csv
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Constants
TARGET_SRATE = 256.0
WINDOW_SEC = 5.0
WINDOW_SAMPLES = int(TARGET_SRATE * WINDOW_SEC)  # 1280
N_CHANNELS = 19
RANDOM_SEED = 42

METADATA_DIR = PROJECT_ROOT / "datasets" / "metadata"
INDEX_CSV = METADATA_DIR / "model_dataset_index.csv"

CLASS_TO_IDX: Dict[str, int] = {
    "healthy": 0,
    "epilepsy": 1,
    "alzheimers": 2,
    "alzheimer": 2,
    "parkinsons": 3,
    "parkinson": 3,
    "depression": 4,
    "seizure": 1,
    "non_seizure": 1,
}

CANONICAL_CLASS: Dict[int, str] = {
    0: "healthy",
    1: "epilepsy",
    2: "alzheimers",
    3: "parkinsons",
    4: "depression",
}


def build_five_class_index(
    processed_root: Path = PROJECT_ROOT / "datasets" / "processed",
    output_csv: Path = INDEX_CSV,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Construct unified 5-class model index from all processed datasets."""
    rng = random.Random(seed)
    dataset_dirs = ["alzheimer", "parkinson", "modma", "healthy", "chbmit"]
    all_rows = []

    for d in dataset_dirs:
        meta_csv = processed_root / d / "metadata.csv"
        if not meta_csv.exists():
            logger.warning(f"Metadata not found for {d}: {meta_csv}")
            continue

        df = pd.read_csv(meta_csv)
        subjs = sorted(df["subject_id"].unique().tolist())
        shuffled = subjs.copy()
        rng.shuffle(shuffled)

        n = len(shuffled)
        if n >= 4:
            n_test = max(1, round(n * 0.15))
            n_val = max(1, round(n * 0.15))
            n_train = n - n_val - n_test
            train_s = set(shuffled[:n_train])
            val_s = set(shuffled[n_train : n_train + n_val])
            test_s = set(shuffled[n_train + n_val :])
        elif n in (2, 3):
            train_s = {shuffled[0]}
            val_s = {shuffled[1]}
            test_s = set(shuffled[2:]) if n == 3 else set()
        else:  # n == 1 (e.g. chb01, srm sub-010)
            train_s = {shuffled[0]}
            val_s = set()
            test_s = set()

        for idx, row in df.iterrows():
            sid = str(row["subject_id"])
            raw_label = str(row["label"]).strip().lower()
            label_idx = CLASS_TO_IDX[raw_label]
            class_cat = CANONICAL_CLASS[label_idx]

            if sid in train_s:
                split = "train"
            elif sid in val_s:
                split = "val"
            elif sid in test_s:
                split = "test"
            else:
                split = "train"

            f_name = str(row["file_path"])
            if d == "chbmit":
                if (processed_root / "chbmit" / f_name).exists():
                    rel_path = f"datasets/processed/chbmit/{f_name}"
                elif (processed_root / "chbmit" / sid / f_name).exists():
                    rel_path = f"datasets/processed/chbmit/{sid}/{f_name}"
                else:
                    rel_path = f"datasets/processed/chbmit/chb01/{f_name}"
            else:
                rel_path = f"datasets/processed/{d}/{f_name}"

            # Verify file exists on disk
            full_file = PROJECT_ROOT / rel_path
            if not full_file.exists():
                logger.error(f"Missing processed file: {full_file}")
                continue

            all_rows.append(
                {
                    "split": split,
                    "dataset_name": str(row.get("dataset", d)),
                    "subject_id": sid,
                    "class_category": class_cat,
                    "class_label_idx": label_idx,
                    "source_file": rel_path,
                    "window_idx": int(row.get("window_idx", idx)),
                    "window_start_sec": float(row.get("window_start", 0.0)),
                    "window_end_sec": float(row.get("window_end", 5.0)),
                    "sampling_rate": TARGET_SRATE,
                    "n_channels": N_CHANNELS,
                    "n_samples": WINDOW_SAMPLES,
                    "is_bipolar_montage": (d == "chbmit"),
                    "missing_channels": (
                        "Pz"
                        if d == "parkinson" and row.get("n_channels_found", 19) < 19
                        else ""
                    ),
                }
            )

    index_df = pd.DataFrame(all_rows)

    # Leakage verification
    for ds_name in index_df["dataset_name"].unique():
        sub_df = index_df[index_df["dataset_name"] == ds_name]
        tr_s = set(sub_df[sub_df["split"] == "train"]["subject_id"])
        va_s = set(sub_df[sub_df["split"] == "val"]["subject_id"])
        te_s = set(sub_df[sub_df["split"] == "test"]["subject_id"])
        assert len(tr_s & va_s) == 0, f"{ds_name} train/val leakage: {tr_s & va_s}"
        assert len(tr_s & te_s) == 0, f"{ds_name} train/test leakage: {tr_s & te_s}"
        assert len(va_s & te_s) == 0, f"{ds_name} val/test leakage: {va_s & te_s}"

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    index_df.to_csv(output_csv, index=False)
    logger.info(f"Exported unified 5-class index ({len(index_df)} rows) to: {output_csv}")

    # Summary
    print("\n=======================================================")
    print("      UNIFIED 5-CLASS MODEL DATASET INDEX SUMMARY      ")
    print("=======================================================")
    print(f"Total Windows: {len(index_df)}")
    print("\nClass Distribution:")
    for cat, count in index_df["class_category"].value_counts().items():
        print(f"  {cat:<15}: {count:5d} ({count/len(index_df)*100:.1f}%)")

    print("\nSplit Distribution:")
    for s, count in index_df["split"].value_counts().items():
        print(f"  {s:<15}: {count:5d} ({count/len(index_df)*100:.1f}%)")

    print("\nCross-Tabulation (Class vs Split):")
    print(pd.crosstab(index_df["class_category"], index_df["split"], margins=True))
    print("=======================================================\n")

    return index_df


if __name__ == "__main__":
    build_five_class_index()
