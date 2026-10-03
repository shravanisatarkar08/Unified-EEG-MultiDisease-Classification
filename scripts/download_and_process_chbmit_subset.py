"""
Fast download of CHB-MIT subjects (chb02 through chb10) from PhysioNet using curl.exe,
preprocess into harmonized 19-channel windows, and rebuild the model dataset index.
"""

import os
import sys
import time
import re
import subprocess
import logging
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from preprocessing.datasets.chbmit import parse_summary, process_subject
from scripts.build_new_index import build_five_class_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_EPILEPSY_DIR = ROOT / "datasets" / "raw" / "epilepsy"
PROCESSED_CHBMIT_DIR = ROOT / "datasets" / "processed" / "chbmit"
BASE_URL = "https://physionet.org/files/chbmit/1.0.0"

# Target subjects: chb02 to chb10
SUBJECTS_TO_DOWNLOAD = ["chb02", "chb03", "chb04", "chb05", "chb06", "chb07", "chb08", "chb09", "chb10"]

def download_file_curl(url: str, dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        logger.info(f"File exists: {dest_path.name} ({dest_path.stat().st_size / 1e6:.1f} MB)")
        return True

    logger.info(f"Downloading {url} -> {dest_path.name} ...")
    t0 = time.time()
    res = subprocess.run(["curl.exe", "-s", "-L", "-o", str(dest_path), url], capture_output=True)
    elapsed = time.time() - t0
    
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        logger.info(f"Downloaded {dest_path.name} ({dest_path.stat().st_size / 1e6:.1f} MB in {elapsed:.1f}s)")
        return True
    else:
        logger.warning(f"Download failed or empty file for {url} (code={res.returncode})")
        if dest_path.exists():
            dest_path.unlink()
        return False

def download_and_process_subject(subj: str):
    subj_dir = RAW_EPILEPSY_DIR / subj
    subj_dir.mkdir(parents=True, exist_ok=True)

    summary_url = f"{BASE_URL}/{subj}/{subj}-summary.txt"
    summary_path = subj_dir / f"{subj}-summary.txt"
    if not download_file_curl(summary_url, summary_path):
        return 0

    seizures_dict = parse_summary(str(summary_path))
    seizure_files = [f for f, sz in seizures_dict.items() if len(sz) > 0]

    with open(summary_path, "r", encoding="utf-8", errors="ignore") as f:
        all_edf_matches = re.findall(r"File Name:\s*(\S+\.edf)", f.read(), re.IGNORECASE)

    # Clean file names (remove non-standard characters)
    clean_files = []
    for sf in seizure_files[:2]:
        if not re.search(r"[+*?]", sf):
            clean_files.append(sf)

    for ef in all_edf_matches:
        if ef not in clean_files and ef not in seizure_files and not re.search(r"[+*?]", ef):
            clean_files.append(ef)
            break

    logger.info(f"[{subj}] Target EDF files: {clean_files}")
    for edf_file in clean_files:
        edf_url = f"{BASE_URL}/{subj}/{edf_file}"
        edf_path = subj_dir / edf_file
        download_file_curl(edf_url, edf_path)

    # Preprocess subject
    out_subj_dir = PROCESSED_CHBMIT_DIR / subj
    meta_list = process_subject(subj, str(subj_dir), str(out_subj_dir), max_files=4)
    logger.info(f"[{subj}] Preprocessed {len(meta_list)} windows.")
    return len(meta_list)

def main():
    logger.info("=" * 80)
    logger.info("DOWNLOADING & PREPROCESSING CHB-MIT SUBJECTS CHB02 - CHB10")
    logger.info("=" * 80)

    total_new_wins = 0
    for subj in SUBJECTS_TO_DOWNLOAD:
        n_wins = download_and_process_subject(subj)
        total_new_wins += n_wins

    # Reconstruct consolidated datasets/processed/chbmit/metadata.csv
    logger.info("Reconstructing consolidated datasets/processed/chbmit/metadata.csv...")
    all_chb_meta = []
    for subj_folder in sorted(PROCESSED_CHBMIT_DIR.glob("chb*")):
        if subj_folder.is_dir():
            s_name = subj_folder.name
            npy_files = list(subj_folder.glob("*.npy"))
            for npy_f in npy_files:
                lbl = "seizure" if "seizure" in npy_f.name and "nonseizure" not in npy_f.name else "non_seizure"
                all_chb_meta.append({
                    "dataset": "chbmit",
                    "subject_id": s_name,
                    "recording_id": npy_f.stem,
                    "label": lbl,
                    "window_idx": 0,
                    "window_start": 0.0,
                    "window_end": 5.0,
                    "n_channels": 19,
                    "srate": 256.0,
                    "n_channels_found": 19,
                    "file_path": f"{s_name}/{npy_f.name}"
                })

    chb_meta_df = pd.DataFrame(all_chb_meta)
    chb_meta_df.to_csv(PROCESSED_CHBMIT_DIR / "metadata.csv", index=False)
    logger.info(f"Saved CHB-MIT metadata.csv with {len(chb_meta_df)} total windows across {chb_meta_df['subject_id'].nunique()} subjects: {sorted(chb_meta_df['subject_id'].unique())}")

    # Rebuild unified 5-class model index
    logger.info("Rebuilding unified model dataset index...")
    new_index_df = build_five_class_index()
    logger.info(f"New total dataset size: {len(new_index_df)} rows")

    # Print summary breakdown
    print("\n" + "=" * 80)
    print("  NEW DATASET SPLIT DISTRIBUTION WITH INDEPENDENT EPILEPSY SUBJECTS")
    print("=" * 80)
    print(pd.crosstab(new_index_df['split'], new_index_df['class_category'], margins=True))
    print("\nEpilepsy Subjects Per Split:")
    for sp in ["train", "val", "test"]:
        ep_subjs = sorted(new_index_df[(new_index_df['split'] == sp) & (new_index_df['class_category'] == 'epilepsy')]['subject_id'].unique())
        ep_wins = len(new_index_df[(new_index_df['split'] == sp) & (new_index_df['class_category'] == 'epilepsy')])
        print(f"  {sp.upper():<6}: {len(ep_subjs)} subjects {ep_subjs} | {ep_wins} windows")
    print("=" * 80)

if __name__ == "__main__":
    main()
