"""
Download and preprocess CHB-MIT subjects chb07 (chb07_13.edf) and chb08 (chb08_02.edf)
to achieve 7 total independent epilepsy subjects.
"""

import concurrent.futures
import logging
import os
import re
import sys
import time
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from preprocessing.datasets.chbmit import parse_summary, process_subject
from scripts.build_new_index import build_five_class_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = ROOT / "datasets" / "raw" / "epilepsy"
PROCESSED_DIR = ROOT / "datasets" / "processed" / "chbmit"
BASE_URL = "https://physionet.org/files/chbmit/1.0.0"

SUBJECT_TARGETS = {
    "chb07": "chb07_13.edf",
    "chb08": "chb08_02.edf",
}

def download_summary(subj: str):
    subj_dir = RAW_DIR / subj
    subj_dir.mkdir(parents=True, exist_ok=True)
    summary_path = subj_dir / f"{subj}-summary.txt"
    if summary_path.exists() and summary_path.stat().st_size > 100:
        return summary_path
    url = f"{BASE_URL}/{subj}/{subj}-summary.txt"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp, open(summary_path, "wb") as f:
            f.write(resp.read())
        return summary_path
    except Exception as e:
        logger.warning(f"Summary download failed for {subj}: {e}")
        return None

def download_chunked(url: str, dest_path: Path, n_threads: int = 16):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 500_000:
        logger.info(f"File exists: {dest_path.name} ({dest_path.stat().st_size / 1e6:.1f} MB)")
        return True

    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        total_size = int(resp.headers.get("content-length", 0))

    logger.info(f"Downloading {dest_path.name} ({total_size / 1e6:.1f} MB) in {n_threads} parallel threads...")
    chunk_size = total_size // n_threads
    chunks = []
    for i in range(n_threads):
        s = i * chunk_size
        e = (i + 1) * chunk_size - 1 if i < n_threads - 1 else total_size - 1
        chunks.append((i, s, e))

    def fetch(item):
        idx, s, e = item
        r = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Range": f"bytes={s}-{e}"})
        for _ in range(3):
            try:
                with urllib.request.urlopen(r, timeout=30) as response:
                    return idx, response.read()
            except Exception:
                time.sleep(1)
        return idx, b""

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
        res = list(executor.map(fetch, chunks))

    res.sort(key=lambda x: x[0])
    with open(dest_path, "wb") as f:
        for _, d in res:
            f.write(d)

    elapsed = time.time() - t0
    final_sz = dest_path.stat().st_size
    logger.info(f"Downloaded {dest_path.name} ({final_sz / 1e6:.1f} MB in {elapsed:.1f}s)")
    return final_sz == total_size

def process_single_subj(subj: str, edf_name: str):
    download_summary(subj)
    url = f"{BASE_URL}/{subj}/{edf_name}"
    dest = RAW_DIR / subj / edf_name
    download_chunked(url, dest, n_threads=16)
    out_dir = PROCESSED_DIR / subj
    meta = process_subject(subj, str(RAW_DIR / subj), str(out_dir), max_files=1)
    logger.info(f"[{subj}] Preprocessed {len(meta)} windows.")
    return len(meta)

def main():
    logger.info("=" * 80)
    logger.info("DOWNLOADING & PREPROCESSING CHB07 & CHB08")
    logger.info("=" * 80)

    for subj, edf in SUBJECT_TARGETS.items():
        try:
            process_single_subj(subj, edf)
        except Exception as e:
            logger.error(f"Error on {subj}: {e}")

    # Consolidate metadata
    logger.info("Consolidating datasets/processed/chbmit/metadata.csv...")
    all_chb_meta = []
    for subj_folder in sorted(PROCESSED_DIR.glob("chb*")):
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

    chb_df = pd.DataFrame(all_chb_meta)
    chb_df.to_csv(PROCESSED_DIR / "metadata.csv", index=False)
    logger.info(f"Saved CHB metadata: {len(chb_df)} windows across {chb_df['subject_id'].nunique()} subjects: {sorted(chb_df['subject_id'].unique())}")

    # Rebuild unified 5-class model index
    logger.info("Rebuilding unified model dataset index...")
    index_df = build_five_class_index()
    logger.info(f"Exported unified index ({len(index_df)} rows)")

    print("\n" + "=" * 80)
    print("  NEW 5-CLASS DATASET SPLIT DISTRIBUTION (WITH INDEPENDENT EPILEPSY SUBJECTS)")
    print("=" * 80)
    print(pd.crosstab(index_df['split'], index_df['class_category'], margins=True))
    print("\nEpilepsy Subjects Per Split:")
    for sp in ["train", "val", "test"]:
        subjs = sorted(index_df[(index_df['split'] == sp) & (index_df['class_category'] == 'epilepsy')]['subject_id'].unique())
        wins = len(index_df[(index_df['split'] == sp) & (index_df['class_category'] == 'epilepsy')])
        print(f"  {sp.upper():<6}: {len(subjs)} subjects {subjs} | {wins} windows")
    print("=" * 80)

if __name__ == "__main__":
    main()
