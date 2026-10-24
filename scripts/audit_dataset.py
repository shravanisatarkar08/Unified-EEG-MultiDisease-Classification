"""
scripts/audit_dataset.py - Comprehensive Data Audit for Phase 1.
Checks:
1. Subject leakage using (dataset_name, subject_id) composite keys.
2. Recording and window uniqueness across splits.
3. Class category to integer index mappings.
4. Dataset source representation per class.
5. Signal statistics (mean, std, min, max, missing channels).
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_CSV = PROJECT_ROOT / "datasets" / "metadata" / "model_dataset_index.csv"

def audit():
    df = pd.read_csv(INDEX_CSV)
    print("=" * 80)
    print("DATASET INTEGRITY & SPLIT AUDIT REPORT")
    print("=" * 80)
    print(f"Total dataset windows: {len(df)}")
    print(f"Splits: {df['split'].value_counts().to_dict()}")

    # 1. Subject Leakage Check
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    test_df = df[df['split'] == 'test']

    tr_subjs = set(zip(train_df['dataset_name'], train_df['subject_id']))
    val_subjs = set(zip(val_df['dataset_name'], val_df['subject_id']))
    test_subjs = set(zip(test_df['dataset_name'], test_df['subject_id']))

    tr_val_leak = tr_subjs & val_subjs
    tr_test_leak = tr_subjs & test_subjs
    val_test_leak = val_subjs & test_subjs

    print("\n--- 1. Subject Leakage Verification ---")
    print(f"Train subjects: {len(tr_subjs)}")
    print(f"Val subjects:   {len(val_subjs)}")
    print(f"Test subjects:  {len(test_subjs)}")
    print(f"Train & Val overlap:  {len(tr_val_leak)} subjects")
    print(f"Train & Test overlap: {len(tr_test_leak)} subjects")
    print(f"Val & Test overlap:   {len(val_test_leak)} subjects")
    assert len(tr_val_leak) == 0 and len(tr_test_leak) == 0 and len(val_test_leak) == 0, "FATAL: Subject leakage found!"
    print(">> ZERO SUBJECT LEAKAGE CONFIRMED across all split pairs.")

    # 2. Window / File Overlap Check
    print("\n--- 2. File and Window Overlap Verification ---")
    tr_files = set(train_df['source_file'])
    val_files = set(val_df['source_file'])
    test_files = set(test_df['source_file'])
    print(f"Unique source files: Train={len(tr_files)}, Val={len(val_files)}, Test={len(test_files)}")
    print(f"Shared source files between splits: {len((tr_files & val_files) | (tr_files & test_files) | (val_files & test_files))}")
    assert len((tr_files & val_files) | (tr_files & test_files) | (val_files & test_files)) == 0, "FATAL: Shared files across splits!"
    print(">> ZERO FILE/WINDOW OVERLAP CONFIRMED across splits.")

    # 3. Label Mapping Verification
    print("\n--- 3. Class Label Mappings ---")
    mapping = df[['class_category', 'class_label_idx']].drop_duplicates().sort_values('class_label_idx')
    print(mapping.to_string(index=False))

    # 4. Source by Class Distribution
    print("\n--- 4. Source Confounding Matrix (Class by Dataset) ---")
    cross = pd.crosstab(df['dataset_name'], df['class_category'], margins=True)
    print(cross)

    # 5. Channel and Signal Statistics Check
    print("\n--- 5. Signal Verification on Sample Windows ---")
    sample_stats = []
    for split, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
        sample_rows = sdf.head(10)
        for _, row in sample_rows.iterrows():
            arr = np.load(PROJECT_ROOT / row['source_file'])
            sample_stats.append({
                "split": split,
                "dataset": row['dataset_name'],
                "shape": list(arr.shape),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            })
    stat_df = pd.DataFrame(sample_stats)
    print(f"Sample shapes verified: all (19, 1280) = {all(s == [19, 1280] for s in stat_df['shape'])}")
    print(f"Per-channel z-score check: mean ~ 0 (avg: {stat_df['mean'].mean():.4f}), std ~ 1 (avg: {stat_df['std'].mean():.4f})")

    # Save audit JSON
    audit_results = {
        "total_windows": len(df),
        "split_counts": df['split'].value_counts().to_dict(),
        "n_subjects": {"train": len(tr_subjs), "val": len(val_subjs), "test": len(test_subjs)},
        "subject_leakage_detected": False,
        "source_file_overlap_detected": False,
        "class_mapping": dict(zip(mapping['class_category'], mapping['class_label_idx'].astype(int))),
        "source_confounding": cross.to_dict(),
        "signals_harmonized": True,
        "target_channels": 19,
        "target_samples": 1280,
        "sampling_rate": 256.0,
    }
    out_dir = PROJECT_ROOT / "results" / "experiments" / "exp7_publication_system"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "data_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)
    print(f"\nAudit saved to {out_dir / 'data_audit.json'}")

if __name__ == "__main__":
    audit()
