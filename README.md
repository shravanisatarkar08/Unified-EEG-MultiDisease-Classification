# Transformer-Based Deep Learning Framework for Robust EEG Signal Classification

A unified, multi-disease EEG classification framework using EEGNet-style CNN feature extraction followed by a Transformer encoder for robust classification across multiple neurological conditions.

## Architecture

`
Raw EEG (dataset-specific)
    |
    v
Dataset-Specific Preprocessing
(filtering, resampling, channel selection)
    |
    v
Common Harmonization
(5-second windows @ 256 Hz, 19 channels, z-score normalization)
    |
    v
EEGNet-style CNN Feature Extractor
[B, 19, 1280] --> [B, 40, 64] feature tokens
    |
    v
Transformer Encoder (CLS-token + self-attention)
[B, 40, 64] --> [B, num_classes] logits
`

### CNN (EEGFeatureExtractor)
- Temporal convolution (64-point kernel)
- Depthwise spatial convolution (across channels)
- Separable convolution + average pooling
- Linear projection to embed_dim=64
- Output: [Batch, T//32, embed_dim] feature tokens

### Transformer Encoder (EEGTransformerEncoder)
- Learnable CLS token prepended to feature sequence
- Sinusoidal positional encoding
- Pre-norm TransformerEncoder layers (GELU activation)
- CLS token output fed to classification head
- 81,412 total parameters (lightweight, CPU-friendly)

### Full Classifier (EEGClassifier)
- Combines EEGFeatureExtractor + EEGTransformerEncoder
- Supports variable number of output classes

---

## Preprocessing Pipeline

Each dataset goes through **dataset-specific preprocessing** followed by **common harmonization**:

1. **Dataset-specific**: Load EDF/SET/FIF/BDF, notch filter, bandpass filter
2. **Harmonization**: Resample to 256 Hz, select/reorder to 19-channel standard 10-20 montage
3. **Segmentation**: 5-second windows, 50% overlap
4. **Normalization**: Per-channel z-score normalization
5. **Output**: .npy files (shape: [19, 1280]) + metadata.csv

### CHB-MIT (Epilepsy)
- Bipolar montage (23 channels) -- uses 19 fixed bipolar channels
- 256 Hz native (no resampling needed)
- Labels: seizure / 
on_seizure (both from epilepsy patients -- NOT healthy controls)
- Seizure annotations from chbXX-summary.txt

### OpenNeuro ds004504 (Alzheimer's)
- Subjects: AD (A) + Healthy Control (C)
- **FTD subjects (F) are explicitly excluded**
- Labels: lzheimer / healthy
- BIDS format with participants.tsv

### OpenNeuro ds004584 (Parkinson's)
- Subjects: PD + Healthy Control
- Group codes detected from participants.tsv (group or diagnosis column)
- Labels: parkinson / healthy

### MODMA (Depression)
- **DATA ACCESS BLOCKER**: Dataset distributed as split archives (.zip.001, .zip.002, ...)
- Archives are NOT modified or deleted
- Requires extraction on Linux + Microsoft Visual C++ build tools for pyEDFlib on Windows
- Module documents blocker status; returns empty until data is manually extracted

### TUAB / TUH (Normal/Abnormal EEG)
- **UNAVAILABLE**: Requires institutional registration with Temple University
- Access URL: https://isip.piconepress.com/projects/tuh_eeg/
- Stub module documents status; returns empty

---

## Supported Datasets

| Dataset | Status | Labels | Blocker |
|---------|--------|--------|---------|
| CHB-MIT (Epilepsy) | ✅ Preprocessor implemented | seizure / non_seizure | Requires local EDF files |
| OpenNeuro ds004504 (Alzheimer's) | ✅ Preprocessor implemented | alzheimer / healthy | Requires local EDF/SET files |
| OpenNeuro ds004584 (Parkinson's) | ✅ Preprocessor implemented | parkinson / healthy | Requires local EDF/SET files |
| MODMA (Depression) | ⚠️ Data access blocker | depression / healthy | Split archives cannot be extracted on Windows without MSVC build tools |
| TUAB/TUH | ❌ Unavailable | normal / abnormal | Requires institutional registration |

---

## Current Prototype Status

**What is implemented and tested:**
- ✅ CHB-MIT preprocessing module with correct relative imports
- ✅ Alzheimer's preprocessor (AD + HC, FTD excluded)
- ✅ Parkinson's preprocessor (PD + HC, group auto-detected from participants.tsv)
- ✅ MODMA stub (safe inspection, no destructive operations)
- ✅ TUAB stub (documented as unavailable)
- ✅ EEGNet-style CNN feature extractor
- ✅ Transformer encoder with CLS token
- ✅ Full EEGClassifier end-to-end pipeline
- ✅ Segmentation (continuous + labeled intervals)
- ✅ Z-score normalization
- ✅ Metadata tracking (CSV)
- ✅ 23 smoke tests passing (0 failures)

**What remains pending:**
- Training pipeline (DataLoader, loss, optimizer, train loop)
- Grad-CAM explainability integration
- Multi-dataset combined training (cross-disease)
- Evaluation metrics (accuracy, F1, sensitivity, specificity)
- MODMA integration (pending data extraction on Linux)
- TUAB integration (pending institutional access)
- Hyperparameter tuning

**Important**: The model has been smoke-tested for correct tensor shapes and NaN-free outputs only. No training has been performed. No accuracy/F1/sensitivity/specificity metrics are reported.

---

## Project Structure

`
Unified-EEG-MultiDisease-Classification/
├── models/
│   ├── eeg_cnn.py          # EEGNet-style CNN feature extractor
│   ├── eeg_classifier.py   # Full CNN + Transformer classifier
│   ├── transformer.py      # Transformer encoder with CLS token
│   └── __init__.py
├── preprocessing/
│   ├── config.py           # Shared constants (srate, channels, paths)
│   ├── common.py           # Shared utilities (filter, normalize, channel mapping)
│   ├── harmonize.py        # Common harmonization pipeline
│   ├── loader.py           # Basic EDF loader
│   ├── metadata.py         # WindowMetadata dataclass + CSV I/O
│   ├── segment.py          # Windowing (continuous + labeled intervals)
│   └── datasets/
│       ├── chbmit.py       # CHB-MIT epilepsy preprocessor
│       ├── alzheimers.py   # OpenNeuro ds004504 Alzheimer's preprocessor
│       ├── parkinsons.py   # OpenNeuro ds004584 Parkinson's preprocessor
│       ├── modma.py        # MODMA depression stub (data blocker documented)
│       ├── tuab.py         # TUAB stub (unavailable)
│       └── __init__.py
├── test_pipeline.py        # 23-test smoke test suite
├── requirements.txt
└── README.md
`

---

## How to Run

### Prerequisites

`ash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install mne numpy scipy pandas scikit-learn einops tqdm
`

### Run Smoke Tests

`ash
python test_pipeline.py
`

Expected output:
`
Ran 23 tests in ~0.4s

OK
`

### Compile All Modules

`ash
python -m compileall preprocessing models
`

### Run CHB-MIT Preprocessing (requires local EDF files)

`ash
python -m preprocessing.datasets.chbmit
`

Data must be at datasets/raw/epilepsy/ with chbXX-summary.txt files.

### Run Alzheimer's Preprocessing (requires local dataset)

`ash
python -m preprocessing.datasets.alzheimers
`

Data must be at datasets/raw/alzheimers/ in BIDS format with participants.tsv.

---

## Dataset Rules (Enforced)

- CHB-MIT non-seizure recordings are labeled 
on_seizure (epilepsy interictal), **NOT healthy controls**
- Alzheimer's ds004504: AD + Healthy only. FTD is **explicitly excluded**
- Labels are derived from actual dataset metadata only — no invented labels
- No metrics are reported from un-trained models

---

## Requirements

See equirements.txt for full dependency list.

Key dependencies:
- PyTorch 2.x (CPU)
- MNE-Python 1.x
- NumPy, SciPy, pandas, scikit-learn

Note: pyEDFlib requires Microsoft Visual C++ build tools on Windows. MNE-Python's built-in EDF reader is used instead.
