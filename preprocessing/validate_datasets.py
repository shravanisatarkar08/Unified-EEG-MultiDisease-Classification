"""
EEG Dataset Validation and Inventory Module.

Scans datasets under `datasets/raw/`, validates raw EEG recording files,
extracts recording metadata (channels, sampling frequency, duration, sample count),
handles participant metadata mapping, and exports an inventory CSV to `datasets/metadata/dataset_inventory.csv`.

Extensible architecture allows registering additional format readers (.edf, .set, .bdf, .fif, etc.)
and dataset definitions without modifying core validation logic.
"""

import os
import re
import csv
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Type, Any

import mne
import pandas as pd

# Suppress verbose MNE runtime warnings during header inspection
warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")
warnings.filterwarnings("ignore", category=UserWarning, module="mne")

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Project root path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "datasets" / "raw"
METADATA_DIR = PROJECT_ROOT / "datasets" / "metadata"


@dataclass
class EEGRecordingMetadata:
    """Metadata for a single raw EEG recording file."""
    dataset_name: str
    class_category: str
    subject_id: str
    file_path: str
    file_format: str
    n_channels: int
    srate_hz: float
    duration_sec: float
    n_samples: int
    channel_names: str


# -----------------------------------------------------------------------------
# EXTENSIBLE FORMAT READER ARCHITECTURE
# -----------------------------------------------------------------------------

class BaseFormatReader:
    """Base interface for reading EEG file headers."""
    supported_extensions: List[str] = []

    def inspect_file(self, file_path: Path) -> Dict[str, Any]:
        """Extract header metadata from EEG file without preloading full data.

        Returns:
            dict with keys: n_channels, srate_hz, duration_sec, n_samples, channel_names
        """
        raise NotImplementedError


class EDFFormatReader(BaseFormatReader):
    """Format reader for European Data Format (.edf) files."""
    supported_extensions = [".edf"]

    def inspect_file(self, file_path: Path) -> Dict[str, Any]:
        raw = mne.io.read_raw_edf(str(file_path), preload=False, verbose=False)
        return {
            "n_channels": len(raw.ch_names),
            "srate_hz": float(raw.info["sfreq"]),
            "duration_sec": round(float(raw.times[-1]), 4) if len(raw.times) > 0 else 0.0,
            "n_samples": int(raw.n_times),
            "channel_names": "|".join(raw.ch_names)
        }


class EEGLABFormatReader(BaseFormatReader):
    """Format reader for EEGLAB (.set) files."""
    supported_extensions = [".set"]

    def inspect_file(self, file_path: Path) -> Dict[str, Any]:
        raw = mne.io.read_raw_eeglab(str(file_path), preload=False, verbose=False)
        return {
            "n_channels": len(raw.ch_names),
            "srate_hz": float(raw.info["sfreq"]),
            "duration_sec": round(float(raw.times[-1]), 4) if len(raw.times) > 0 else 0.0,
            "n_samples": int(raw.n_times),
            "channel_names": "|".join(raw.ch_names)
        }


class FormatRegistry:
    """Registry mapping file extensions to format readers."""
    def __init__(self):
        self._readers: Dict[str, BaseFormatReader] = {}

    def register(self, reader: BaseFormatReader):
        for ext in reader.supported_extensions:
            self._readers[ext.lower()] = reader

    def get_reader(self, file_extension: str) -> Optional[BaseFormatReader]:
        return self._readers.get(file_extension.lower())

    def supported_extensions(self) -> List[str]:
        return list(self._readers.keys())


# Default format registry initialization
DEFAULT_REGISTRY = FormatRegistry()
DEFAULT_REGISTRY.register(EDFFormatReader())
DEFAULT_REGISTRY.register(EEGLABFormatReader())


# -----------------------------------------------------------------------------
# PLANNED DATASETS SPECIFICATION
# -----------------------------------------------------------------------------

PLANNED_DATASETS = [
    {
        "name": "chbmit",
        "folder": "epilepsy",
        "default_category": "epilepsy",
        "description": "CHB-MIT Seizure / Epilepsy EEG Dataset"
    },
    {
        "name": "srm_healthy",
        "folder": "healthy",
        "default_category": "healthy",
        "description": "SRM Healthy Control EEG Dataset (ds003775)"
    },
    {
        "name": "alzheimers",
        "folder": "alzheimers",
        "default_category": "alzheimers",
        "description": "OpenNeuro ds004504 Alzheimer's Dataset"
    },
    {
        "name": "parkinsons",
        "folder": "parkinsons",
        "default_category": "parkinsons",
        "description": "OpenNeuro ds004584 Parkinson's Dataset"
    },
    {
        "name": "modma",
        "folder": "depression",
        "default_category": "depression",
        "description": "MODMA Depression EEG Dataset"
    },
    {
        "name": "tuab",
        "folder": "tuab",
        "default_category": "unspecified",
        "description": "TUAB Abnormal/Normal EEG Dataset"
    }
]


# -----------------------------------------------------------------------------
# DATASET VALIDATOR & INVENTORY BUILDER
# -----------------------------------------------------------------------------

class DatasetValidator:
    """Scans raw datasets, validates recording files, and exports inventory."""

    def __init__(self, raw_dir: Path = RAW_DIR, registry: FormatRegistry = DEFAULT_REGISTRY):
        self.raw_dir = Path(raw_dir)
        self.registry = registry

    def _extract_subject_id(self, file_path: Path) -> str:
        """Extract subject ID from directory structure or filename."""
        for part in file_path.parts:
            if part.startswith("sub-") or re.match(r"^chb\d+$", part, re.IGNORECASE):
                return part

        filename = file_path.name
        match = re.search(r"(sub-[a-zA-Z0-9]+|chb\d+)", filename, re.IGNORECASE)
        if match:
            return match.group(1)
        return "unknown"

    def _load_participant_categories(self, dataset_folder_path: Path, default_category: str) -> Dict[str, str]:
        """Load subject -> class category mapping from participants.tsv if available."""
        mapping = {}
        participants_tsv = dataset_folder_path / "participants.tsv"
        if not participants_tsv.exists():
            return mapping

        try:
            df = pd.read_csv(participants_tsv, sep="\t")
            subj_col = None
            for col in ["participant_id", "subject_id", "ID"]:
                if col in df.columns:
                    subj_col = col
                    break

            group_col = None
            for col in ["Group", "GROUP", "group", "diagnosis", "Group/Diagnosis"]:
                if col in df.columns:
                    group_col = col
                    break

            if subj_col and group_col:
                for _, row in df.iterrows():
                    sid = str(row[subj_col]).strip()
                    grp = str(row[group_col]).strip()

                    # Standardize category labels
                    if grp in ["A", "AD", "Alzheimer"]:
                        cat = "alzheimers"
                    elif grp in ["C", "Control", "HC", "Healthy"]:
                        cat = "healthy"
                    elif grp in ["PD", "Parkinson"]:
                        cat = "parkinsons"
                    elif grp in ["F", "FTD"]:
                        cat = "ftd"
                    elif grp in ["MDD", "Depression"]:
                        cat = "depression"
                    else:
                        cat = grp.lower()

                    mapping[sid] = cat
        except Exception as e:
            logger.warning(f"Could not parse participants.tsv in {dataset_folder_path}: {e}")

        return mapping

    def scan_dataset(self, dataset_spec: Dict[str, str]) -> Dict[str, Any]:
        """Scan a single planned dataset folder."""
        ds_name = dataset_spec["name"]
        folder_name = dataset_spec["folder"]
        default_cat = dataset_spec["default_category"]
        ds_path = self.raw_dir / folder_name

        result = {
            "name": ds_name,
            "folder": folder_name,
            "status": "NOT AVAILABLE LOCALLY",
            "files": [],
            "recordings": []
        }

        if not ds_path.exists():
            logger.info(f"Dataset '{ds_name}' ({folder_name}): Directory missing -> NOT AVAILABLE LOCALLY")
            return result

        # Load participant category map if available
        participant_map = self._load_participant_categories(ds_path, default_cat)

        # Scan for matching EEG files
        supported_exts = set(self.registry.supported_extensions())
        candidate_files = []

        for root, dirs, files in os.walk(ds_path):
            # Skip BIDS derivatives to prevent duplicate derivative indexing
            if "derivatives" in Path(root).parts:
                continue

            for file in files:
                ext = Path(file).suffix.lower()
                if ext in supported_exts:
                    candidate_files.append(Path(root) / file)

        if not candidate_files:
            logger.info(f"Dataset '{ds_name}' ({folder_name}): No raw EEG files found -> NOT AVAILABLE LOCALLY")
            return result

        result["status"] = "AVAILABLE"
        result["files"] = candidate_files

        # Process each raw recording
        for file_path in candidate_files:
            ext = file_path.suffix.lower()
            reader = self.registry.get_reader(ext)
            if not reader:
                continue

            subj_id = self._extract_subject_id(file_path)
            category = participant_map.get(subj_id, default_cat)

            try:
                header_info = reader.inspect_file(file_path)
                rel_path = str(file_path.relative_to(PROJECT_ROOT)).replace("\\", "/")

                rec_meta = EEGRecordingMetadata(
                    dataset_name=ds_name,
                    class_category=category,
                    subject_id=subj_id,
                    file_path=rel_path,
                    file_format=ext,
                    n_channels=header_info["n_channels"],
                    srate_hz=header_info["srate_hz"],
                    duration_sec=header_info["duration_sec"],
                    n_samples=header_info["n_samples"],
                    channel_names=header_info["channel_names"]
                )
                result["recordings"].append(rec_meta)
            except Exception as e:
                logger.error(f"Failed to inspect file '{file_path}': {e}")

        logger.info(f"Dataset '{ds_name}' ({folder_name}): AVAILABLE ({len(result['recordings'])} recordings verified)")
        return result

    def validate_all(self) -> Dict[str, Any]:
        """Validate all planned datasets under datasets/raw/."""
        all_recordings: List[EEGRecordingMetadata] = []
        dataset_reports = []

        for spec in PLANNED_DATASETS:
            report = self.scan_dataset(spec)
            dataset_reports.append(report)
            all_recordings.extend(report["recordings"])

        return {
            "dataset_reports": dataset_reports,
            "all_recordings": all_recordings
        }

    def export_inventory_csv(self, recordings: List[EEGRecordingMetadata], output_file: Path = METADATA_DIR / "dataset_inventory.csv") -> Path:
        """Export list of EEGRecordingMetadata to CSV."""
        output_file.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "dataset_name",
            "class_category",
            "subject_id",
            "file_path",
            "file_format",
            "n_channels",
            "srate_hz",
            "duration_sec",
            "n_samples",
            "channel_names"
        ]

        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in recordings:
                writer.writerow(asdict(rec))

        logger.info(f"Exported dataset inventory ({len(recordings)} recordings) to: {output_file}")
        return output_file


# -----------------------------------------------------------------------------
# CONSOLE SUMMARY DISPLAY
# -----------------------------------------------------------------------------

def print_validation_summary(summary_data: Dict[str, Any]):
    """Print structured tabular report of dataset inventory validation."""
    reports = summary_data["dataset_reports"]
    all_recs = summary_data["all_recordings"]

    print("\n==================================================================================")
    print("                      EEG DATASET INVENTORY & VALIDATION REPORT                   ")
    print("==================================================================================")
    print(f"{'Dataset':<15} | {'Folder':<12} | {'Status':<22} | {'Recordings':<10} | {'Formats':<10}")
    print("----------------------------------------------------------------------------------")

    for rep in reports:
        formats = list(set([r.file_format for r in rep["recordings"]])) if rep["recordings"] else ["N/A"]
        fmt_str = ",".join(formats)
        rec_count = len(rep["recordings"])
        status = rep["status"]
        print(f"{rep['name']:<15} | {rep['folder']:<12} | {status:<22} | {rec_count:<10} | {fmt_str:<10}")

    print("----------------------------------------------------------------------------------")
    print(f"Total Local Verified EEG Recordings: {len(all_recs)}")
    print("==================================================================================")

    # Breakdown per category
    if all_recs:
        cat_df = pd.DataFrame([asdict(r) for r in all_recs])
        print("\nRecordings Breakdown by Class / Category:")
        print(cat_df["class_category"].value_counts().to_string())

        print("\nRecordings Breakdown by Sampling Rate (Hz):")
        print(cat_df["srate_hz"].value_counts().to_string())

        print("\nRecordings Breakdown by Channel Count:")
        print(cat_df["n_channels"].value_counts().to_string())

        # Show representative sample files
        print("\nSample Verified Recordings (First 5):")
        sample_cols = ["dataset_name", "class_category", "subject_id", "file_format", "n_channels", "srate_hz", "duration_sec"]
        print(cat_df[sample_cols].head(5).to_string(index=False))

    print("==================================================================================\n")


def run_validation() -> Dict[str, Any]:
    """Execute complete dataset validation and export metadata CSV."""
    validator = DatasetValidator()
    summary = validator.validate_all()
    csv_path = validator.export_inventory_csv(summary["all_recordings"])
    print_validation_summary(summary)
    return summary


if __name__ == "__main__":
    run_validation()
