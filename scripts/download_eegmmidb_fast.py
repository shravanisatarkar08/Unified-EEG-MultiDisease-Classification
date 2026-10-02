"""
Fast concurrent downloader for EEGMMIDB rest baseline runs (R01, R02) from PhysioNet.
Saves directly into MNE's standard eegbci data directory:
  ~/mne_data/MNE-eegbci-data/files/eegmmidb/1.0.0/Sxxx/SxxxR0y.edf
"""
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_URL = "https://physionet.org/files/eegmmidb/1.0.0"
MNE_DATA = Path.home() / "mne_data" / "MNE-eegbci-data" / "files" / "eegmmidb" / "1.0.0"

def download_file(sub_id: int, run_id: int):
    sub_str = f"S{sub_id:03d}"
    filename = f"{sub_str}R{run_id:02d}.edf"
    dest_dir = MNE_DATA / sub_str
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename

    if dest_path.exists() and dest_path.stat().st_size > 100_000:
        return f"{filename} already exists"

    url = f"{BASE_URL}/{sub_str}/{filename}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp, open(dest_path, "wb") as out_file:
            out_file.write(resp.read())
        return f"{filename} downloaded ({dest_path.stat().st_size // 1024} KB)"
    except Exception as e:
        if dest_path.exists():
            dest_path.unlink()
        return f"FAILED {filename}: {e}"

def main():
    # Download 40 subjects (S001 to S040, runs 1 and 2 = 80 files)
    # 40 healthy subjects provides a statistically powerful control cohort
    tasks = []
    print(f"Target directory: {MNE_DATA}")
    with ThreadPoolExecutor(max_workers=8) as pool:
        for s in range(1, 41):
            for r in [1, 2]:
                tasks.append(pool.submit(download_file, s, r))
        
        for future in as_completed(tasks):
            res = future.result()
            print(res)

    print("Concurrent download complete!")

if __name__ == "__main__":
    main()
