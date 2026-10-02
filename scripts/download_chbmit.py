"""
Download CHB-MIT chb01 EDF files one by one using kaggle CLI.
chb01 has 42 recordings (chb01_01.edf through chb01_42.edf, non-contiguous).
Per summary file, seizures are present in specific files.
"""
import subprocess
import sys
import os
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EPILEPSY_DIR = PROJECT_ROOT / "datasets" / "raw" / "epilepsy"
DATASET = "beyzanurdin/chb-mit-scalp-eeg-database"

# chb01 files known from the Kaggle listing (summary + all EDF files)
# We'll download a practical subset: files with seizures + some non-seizure files
# From chb01-summary.txt:
# Seizures in: chb01_03, chb01_04, chb01_15, chb01_16, chb01_18, chb01_21, chb01_26
SEIZURE_FILES = [
    "chb01_03.edf", "chb01_04.edf", "chb01_15.edf",
    "chb01_16.edf", "chb01_18.edf", "chb01_21.edf", "chb01_26.edf"
]
# Non-seizure files for balance
NON_SEIZURE_FILES = [
    "chb01_01.edf", "chb01_02.edf", "chb01_05.edf",
    "chb01_06.edf", "chb01_07.edf", "chb01_08.edf",
    "chb01_09.edf", "chb01_10.edf", "chb01_11.edf",
    "chb01_12.edf"
]

ALL_FILES = SEIZURE_FILES + NON_SEIZURE_FILES


def download_file(remote_path, dest_dir):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["kaggle", "datasets", "download",
         "-d", DATASET,
         "--file", remote_path,
         "-p", str(dest_dir),
         "--unzip"],
        capture_output=True, text=True
    )
    return result.returncode


def main():
    subject_dir = EPILEPSY_DIR / "chb01"
    subject_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading CHB-MIT chb01 EDF files to: {subject_dir}")
    print(f"Files to download: {len(ALL_FILES)}")

    for fname in ALL_FILES:
        local = subject_dir / fname
        if local.exists() and local.stat().st_size > 40_000_000:
            print(f"  {fname}: already exists ({local.stat().st_size//1_000_000} MB)")
            continue

        remote = f"1.0.0/chb01/{fname}"
        print(f"  Downloading {fname}...", end=" ", flush=True)
        t = time.time()
        rc = download_file(remote, subject_dir)
        elapsed = time.time() - t

        if local.exists() and local.stat().st_size > 40_000_000:
            print(f"OK ({local.stat().st_size//1_000_000} MB, {elapsed:.1f}s)")
        else:
            print(f"FAILED or MISSING after {elapsed:.1f}s (rc={rc})")

    # Summary
    edfs = list(subject_dir.glob("*.edf"))
    total_mb = sum(f.stat().st_size for f in edfs) / 1e6
    print(f"\nDone: {len(edfs)} EDF files, {total_mb:.0f} MB total")
    print(f"Path: {subject_dir}")


if __name__ == "__main__":
    main()
