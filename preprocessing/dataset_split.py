"""
Subject-Level Dataset Splitting and Model-Dataset Preparation Module.

Performs leakage-safe Train / Validation / Test dataset splitting at the SUBJECT level.
All 5-second EEG windows from a subject remain strictly inside exactly one split (Train, Val, or Test).

FIVE TARGET CLASSES & REPRODUCIBLE LABEL MAPPING:
------------------------------------------------
0: healthy
1: epilepsy
2: alzheimers
3: parkinsons
4: depression

KEY SCIENTIFIC RULES ENFORCED:
-------------------------------
1. Zero Subject Leakage:
   `train_subjects ∩ val_subjects = ∅`
   `train_subjects ∩ test_subjects = ∅`
   `val_subjects ∩ test_subjects = ∅`
2. Explicit FTD Exclusion:
   Frontotemporal Dementia (Group 'F' in ds004504) subjects are explicitly EXCLUDED.
3. No Fake Data Fabrication:
   Depression (MODMA) is reported as UNAVAILABLE LOCALLY. No fake samples are generated.
4. Dataset Provenance Tracking:
   `dataset_name` is preserved in all window indices (distinguishing `srm_healthy` vs
   healthy controls from `alzheimers` / `parkinsons`).
5. Subject-Level Stratification & Proportions:
   Target split ratios: 70% Train, 15% Validation, 15% Test.
   Single-subject classes (e.g. CHB-MIT sample) are allocated to Train for development testing
   with documented limitations.
6. Lightweight Index Output:
   Exports model index to `datasets/metadata/model_dataset_index.csv` without duplicating
   large raw EEG arrays.
"""

import os
import re
import csv
import logging
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple, Set, Any

import numpy as np
import pandas as pd

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.dataset_preprocessors import run_controlled_sample_preprocessing
from preprocessing.harmonization import EEGHarmonizer, HarmonizationConfig, HarmonizedWindow

logger = logging.getLogger(__name__)

METADATA_DIR = PROJECT_ROOT / "datasets" / "metadata"
INVENTORY_CSV = METADATA_DIR / "dataset_inventory.csv"
MODEL_INDEX_CSV = METADATA_DIR / "model_dataset_index.csv"

# -----------------------------------------------------------------------------
# CLASS-TO-INTEGER LABEL MAPPING (REPRODUCIBLE & EXPLICIT)
# -----------------------------------------------------------------------------

CLASS_TO_IDX: Dict[str, int] = {
    'healthy': 0,
    'epilepsy': 1,
    'alzheimers': 2,
    'parkinsons': 3,
    'depression': 4
}

IDX_TO_CLASS: Dict[int, str] = {v: k for k, v in CLASS_TO_IDX.items()}

EXCLUDED_CLASSES: Set[str] = {'ftd', 'F', 'Group F'}


# -----------------------------------------------------------------------------
# DATA STRUCTURES
# -----------------------------------------------------------------------------

@dataclass
class SubjectAssignment:
    """Subject-level split assignment record."""
    subject_id: str
    dataset_name: str
    class_category: str
    class_label_idx: int
    split: str  # 'train', 'val', or 'test'


@dataclass
class ModelWindowIndexRow:
    """Lightweight metadata index row pointing to an EEG window."""
    split: str
    dataset_name: str
    subject_id: str
    class_category: str
    class_label_idx: int
    source_file: str
    window_idx: int
    window_start_sec: float
    window_end_sec: float
    sampling_rate: float            # 256.0 Hz
    n_channels: int                 # 19
    n_samples: int                  # 1280
    is_bipolar_montage: bool
    missing_channels: str           # pipe-separated string e.g. "Pz"


# -----------------------------------------------------------------------------
# SUBJECT-LEVEL SPLITTER
# -----------------------------------------------------------------------------

class SubjectLevelSplitter:
    """Splits subjects into Train (70%), Validation (15%), and Test (15%) splits.

    Ensures zero subject overlap across splits and preserves class label mappings.
    """

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42
    ):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed

        total = round(train_ratio + val_ratio + test_ratio, 4)
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    def load_valid_subjects(self, inventory_csv: Path = INVENTORY_CSV) -> pd.DataFrame:
        """Loads inventory CSV, filters out FTD subjects, and maps target class label indices."""
        if not inventory_csv.exists():
            raise FileNotFoundError(f"Dataset inventory file not found: {inventory_csv}")

        df = pd.read_csv(inventory_csv)

        # 1. Explicitly filter out FTD subjects
        df_valid = df[~df['class_category'].isin(EXCLUDED_CLASSES)].copy()

        # 2. Add class_label_idx column
        df_valid['class_label_idx'] = df_valid['class_category'].map(CLASS_TO_IDX)

        # 3. Filter unique subjects per (dataset_name, subject_id, class_category, class_label_idx)
        subj_df = df_valid.groupby(['dataset_name', 'subject_id', 'class_category', 'class_label_idx']).size().reset_index(name='n_recordings')
        return subj_df

    def split_subjects(self, subjects_df: pd.DataFrame) -> List[SubjectAssignment]:
        """Performs stratified subject-level splitting into train, val, and test assignments."""
        assignments: List[SubjectAssignment] = []
        rng = np.random.RandomState(self.random_seed)

        # Process each class category separately
        categories = sorted(subjects_df['class_category'].unique())

        for cat in categories:
            cat_df = subjects_df[subjects_df['class_category'] == cat].copy()
            label_idx = CLASS_TO_IDX[cat]

            # Unique subject tuples (dataset_name, subject_id)
            subj_tuples = sorted(list(set(zip(cat_df['dataset_name'], cat_df['subject_id']))))
            n_subjs = len(subj_tuples)

            # Shuffle subjects deterministically
            shuffled_subjs = subj_tuples.copy()
            rng.shuffle(shuffled_subjs)

            if n_subjs >= 3:
                n_val = max(1, int(round(n_subjs * self.val_ratio)))
                n_test = max(1, int(round(n_subjs * self.test_ratio)))
                n_train = n_subjs - n_val - n_test

                if n_train <= 0:
                    n_train = 1
                    n_val = max(1, (n_subjs - 1) // 2)
                    n_test = n_subjs - n_train - n_val

                train_subjs = shuffled_subjs[:n_train]
                val_subjs = shuffled_subjs[n_train:n_train + n_val]
                test_subjs = shuffled_subjs[n_train + n_val:]

            elif n_subjs == 2:
                train_subjs = [shuffled_subjs[0]]
                val_subjs = [shuffled_subjs[1]]
                test_subjs = []
                logger.info(f"[{cat}] Only 2 subjects available -> Assigned 1 to Train, 1 to Val")

            else:  # n_subjs == 1
                train_subjs = [shuffled_subjs[0]]
                val_subjs = []
                test_subjs = []
                logger.info(f"[{cat}] Single-subject development sample '{shuffled_subjs[0]}' assigned to Train split")

            # Create assignment records
            for ds_name, sid in train_subjs:
                assignments.append(SubjectAssignment(sid, ds_name, cat, label_idx, 'train'))

            for ds_name, sid in val_subjs:
                assignments.append(SubjectAssignment(sid, ds_name, cat, label_idx, 'val'))

            for ds_name, sid in test_subjs:
                assignments.append(SubjectAssignment(sid, ds_name, cat, label_idx, 'test'))

        return assignments

    def verify_zero_leakage(self, assignments: List[SubjectAssignment]):
        """Verifies zero subject overlap between splits and complete FTD exclusion."""
        train_keys = set((a.dataset_name, a.subject_id) for a in assignments if a.split == 'train')
        val_keys = set((a.dataset_name, a.subject_id) for a in assignments if a.split == 'val')
        test_keys = set((a.dataset_name, a.subject_id) for a in assignments if a.split == 'test')

        # Check intersections
        tv_overlap = train_keys.intersection(val_keys)
        tt_overlap = train_keys.intersection(test_keys)
        vt_overlap = val_keys.intersection(test_keys)

        if tv_overlap:
            raise ValueError(f"LEAKAGE DETECTED: Subjects in Train & Val: {tv_overlap}")
        if tt_overlap:
            raise ValueError(f"LEAKAGE DETECTED: Subjects in Train & Test: {tt_overlap}")
        if vt_overlap:
            raise ValueError(f"LEAKAGE DETECTED: Subjects in Val & Test: {vt_overlap}")

        # Check FTD exclusion
        ftd_found = [a for a in assignments if a.class_category in EXCLUDED_CLASSES]
        if ftd_found:
            raise ValueError(f"FTD EXCLUSION VIOLATED: Found FTD subjects in splits: {ftd_found}")

        logger.info("✅ ZERO SUBJECT LEAKAGE VERIFIED across Train, Val, and Test splits!")
        logger.info("✅ COMPLETE FTD EXCLUSION VERIFIED!")


# -----------------------------------------------------------------------------
# INDEX GENERATION & EXPORT
# -----------------------------------------------------------------------------

def build_model_window_index(
    assignments: List[SubjectAssignment],
    sample_windows: Optional[List[HarmonizedWindow]] = None
) -> List[ModelWindowIndexRow]:
    """Builds lightweight model window index records pointing to source windows."""
    split_lookup = {a.subject_id: a.split for a in assignments}
    index_rows: List[ModelWindowIndexRow] = []

    if sample_windows:
        for w in sample_windows:
            split = split_lookup.get(w.subject_id, 'train')
            label_idx = CLASS_TO_IDX.get(w.class_category, 0)
            missing_str = "|".join(w.missing_channels) if w.missing_channels else ""

            row = ModelWindowIndexRow(
                split=split,
                dataset_name=w.dataset_name,
                subject_id=w.subject_id,
                class_category=w.class_category,
                class_label_idx=label_idx,
                source_file=w.source_file,
                window_idx=w.window_idx,
                window_start_sec=w.window_start_sec,
                window_end_sec=w.window_end_sec,
                sampling_rate=w.sampling_rate,
                n_channels=w.n_channels,
                n_samples=w.n_samples,
                is_bipolar_montage=w.is_bipolar_montage,
                missing_channels=missing_str
            )
            index_rows.append(row)

    return index_rows


def export_model_index_csv(
    index_rows: List[ModelWindowIndexRow],
    output_path: Path = MODEL_INDEX_CSV
) -> Path:
    """Exports lightweight model window index CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'split', 'dataset_name', 'subject_id', 'class_category', 'class_label_idx',
        'source_file', 'window_idx', 'window_start_sec', 'window_end_sec',
        'sampling_rate', 'n_channels', 'n_samples', 'is_bipolar_montage', 'missing_channels'
    ]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in index_rows:
            writer.writerow(asdict(row))

    logger.info(f"Exported model window index ({len(index_rows)} rows) to: {output_path}")
    return output_path


# -----------------------------------------------------------------------------
# CONSOLE SUMMARY & EXECUTION
# -----------------------------------------------------------------------------

def print_split_summary(assignments: List[SubjectAssignment], index_rows: Optional[List[ModelWindowIndexRow]] = None):
    """Prints structured report of subject-level dataset split."""
    df_ass = pd.DataFrame([asdict(a) for a in assignments])

    print("\n==================================================================================")
    print("                 SUBJECT-LEVEL DATASET SPLIT & PREPARATION REPORT                 ")
    print("==================================================================================")
    print("Status: DEVELOPMENT SPLIT (Label mapping active; Depression currently UNAVAILABLE)")
    print("----------------------------------------------------------------------------------")
    print("Class Label Index Mapping:")
    for cls, idx in CLASS_TO_IDX.items():
        avail_str = "AVAILABLE" if cls != 'depression' else "NOT AVAILABLE LOCALLY"
        print(f"  [{idx}] {cls:<12} -> {avail_str}")
    print("----------------------------------------------------------------------------------")

    print("\nSubject Counts by Split and Class Category:")
    pivot_subj = pd.pivot_table(df_ass, index='class_category', columns='split', values='subject_id', aggfunc='nunique', fill_value=0)
    print(pivot_subj.to_string())

    print("\nSubject Counts by Split and Source Dataset:")
    pivot_ds = pd.pivot_table(df_ass, index='dataset_name', columns='split', values='subject_id', aggfunc='nunique', fill_value=0)
    print(pivot_ds.to_string())

    print("\nTotal Subjects per Split:")
    print(df_ass['split'].value_counts().to_string())

    if index_rows:
        df_idx = pd.DataFrame([asdict(r) for r in index_rows])
        print("\nDevelopment Sample Window Counts per Split:")
        print(df_idx['split'].value_counts().to_string())

    print("==================================================================================\n")


def run_subject_level_splitting() -> Tuple[List[SubjectAssignment], List[ModelWindowIndexRow]]:
    """Runs dataset inventory subject loading, zero-leakage splitting, and index generation."""
    splitter = SubjectLevelSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42)
    subjects_df = splitter.load_valid_subjects()
    assignments = splitter.split_subjects(subjects_df)
    splitter.verify_zero_leakage(assignments)

    # Run on controlled sample harmonization windows
    from preprocessing.harmonization import run_controlled_sample_harmonization
    sample_windows = run_controlled_sample_harmonization()
    index_rows = build_model_window_index(assignments, sample_windows)
    export_model_index_csv(index_rows)

    print_split_summary(assignments, index_rows)
    return assignments, index_rows


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_subject_level_splitting()
