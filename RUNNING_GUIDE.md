# Running Guide: Unified EEG Multi-Disease Classification Framework

This guide provides step-by-step instructions for running, testing, training, and benchmarking the **Unified EEG Multi-Disease Classification Framework** across all supported modalities.

---

## Table of Contents
1. [Quick Start (TL;DR)](#quick-start-tldr)
2. [Environment Setup & Installation](#1-environment-setup--installation)
3. [Running the Model](#2-running-the-model)
   - [3.1 Forward Inference Demo](#21-forward-inference-demo)
   - [3.2 GPU Acceleration](#22-gpu-acceleration)
   - [3.3 Synthetic Training Step Demo](#23-synthetic-training-step-demo)
4. [Running the Verification Test Suite](#3-running-the-verification-test-suite)
5. [Running Individual Model Modules](#4-running-individual-model-modules)
6. [Dataset Preprocessing Guide](#5-dataset-preprocessing-guide)
   - [6.1 Directory Setup](#51-directory-setup)
   - [6.2 CHB-MIT (Epilepsy)](#52-chb-mit-epilepsy)
   - [6.3 OpenNeuro ds004504 (Alzheimer's)](#53-openneuro-ds004504-alzheimers)
   - [6.4 OpenNeuro ds004584 (Parkinson's)](#54-openneuro-ds004584-parkinsons)
7. [Real-Time Classification (RTC) Execution](#6-real-time-classification-rtc-execution)
8. [Comprehensive Command Reference](#7-comprehensive-command-reference)
9. [Troubleshooting & FAQ](#8-troubleshooting--faq)

---

## Quick Start (TL;DR)

Open PowerShell or terminal in the project root directory and run:

```powershell
# 1. Run inference on simulated 19-channel EEG
python run_model.py

# 2. Run a 5-step training demonstration
python run_model.py --train-demo

# 3. Run all 23 smoke & shape tests
python test_pipeline.py
```

---

## 1. Environment Setup & Installation

### 1.1 Python Requirement
- **Python 3.10, 3.11, 3.12, or 3.13** is recommended.

### 1.2 Install Core Dependencies

#### Standard CPU Setup (Recommended for lightweight inference):
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install mne numpy scipy pandas scikit-learn einops tqdm
```

#### Full Requirements File:
```powershell
pip install -r requirements.txt
```

#### GPU (CUDA) Setup (Optional):
If you have an NVIDIA GPU with CUDA installed:
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 1.3 Verify Installation
Verify that PyTorch and MNE are correctly recognized:
```powershell
python -c "import torch, mne; print(f'PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})'); print(f'MNE: {mne.__version__}')"
```

---

## 2. Running the Model

### 2.1 Forward Inference Demo
Run `run_model.py` without arguments to simulate an EEG window batch (4 samples × 19 channels × 1280 timepoints):

```powershell
python run_model.py
```

#### What this performs:
1. Instantiates the full **`EEGClassifier`** (CNN Feature Extractor + Transformer Encoder).
2. Generates a batch of 5-second EEG windows at 256 Hz across standard 10-20 channels (`[4, 19, 1280]`).
3. Passes signals through the temporal, spatial, and separable convolutions of `EEGFeatureExtractor` into 40 temporal tokens (`[4, 40, 64]`).
4. Prepends the learnable `[CLS]` token and applies multi-head self-attention via `EEGTransformerEncoder`.
5. Projects to target disease classes:
   - **Class 0**: Healthy Control
   - **Class 1**: Epilepsy (Seizure/Non-Seizure)
   - **Class 2**: Alzheimer's Disease
   - **Class 3**: Parkinson's Disease
6. Computes softmax probabilities, argmax classification, latency per sample, and parameter counts.

#### Sample Output:
```text
======================================================================
 UNIFIED EEG MULTI-DISEASE CLASSIFIER - FORWARD INFERENCE DEMO
======================================================================
[*] Device:             cpu
[*] Batch size:         4
[*] EEG Montage:        19 channels (10-20 system)
[*] Window duration:    5 seconds (1280 samples @ 256 Hz)
[*] Target Classes (4):
    Class 0: Healthy Control
    Class 1: Epilepsy (Seizure/Non-Seizure)
    Class 2: Alzheimer's Disease
    Class 3: Parkinson's Disease
----------------------------------------------------------------------
[*] Initializing EEGClassifier (CNN Feature Extractor + Transformer)...
    - CNN Extractor Params:     5,700
    - Transformer Params:       76,100
    - Total Model Parameters:   81,800
----------------------------------------------------------------------
[*] Input Tensor Shape: [B, C, T] = [4, 19, 1280]
[*] Stage 1 (CNN): Feature Tokens Shape: [B, N, D] = [4, 40, 64]
    (40 time-step tokens with 64 embedding dimensions per window)
[*] Stage 2 (Transformer): Logits Shape: [B, num_classes] = [4, 4]
[*] Forward pass latency: 24.88 ms/sample
----------------------------------------------------------------------
[*] Sample Predictions across the batch:
    Sample #1: Predicted -> [Epilepsy (Seizure/Non-Seizure)] (Confidence: 31.4%)
              Distribution: [C0: 22.4%, C1: 31.4%, C2: 23.9%, C3: 22.3%]
======================================================================
[SUCCESS] Model executed successfully!
======================================================================
```

### 2.2 GPU Acceleration
To force execution on CUDA or CPU explicitly:

```powershell
# Automatically use CUDA if available, else CPU:
python run_model.py --device auto

# Force execution on CPU:
python run_model.py --device cpu

# Force execution on GPU (requires CUDA):
python run_model.py --device cuda
```

### 2.3 Synthetic Training Step Demo
To verify that gradients flow properly through the CNN and Transformer layers back to all weights:

```powershell
python run_model.py --train-demo
```

#### What this performs:
- Configures the model in `train()` mode.
- Initializes the `AdamW` optimizer (`lr=1e-3, weight_decay=1e-2`) and `nn.CrossEntropyLoss`.
- Runs 5 training iterations on synthetic mini-batches.
- Prints loss values and batch training accuracy at each step to verify backpropagation and weight updating.

---

## 3. Running the Verification Test Suite

Run the comprehensive 23-test test suite to check tensor consistency, layer transformations, and dataset loader compatibility:

```powershell
python test_pipeline.py
```

To run with detailed verbosity showing every single test case:
```powershell
python -m unittest test_pipeline.py -v
```

### Tests Covered:
- `TestConfig`: Target sampling rates (256 Hz), window lengths (5s), channel count (19), filter cutoffs.
- `TestSegmentation`: Continuous segmentation, labeled interval slicing, and short-recording handling.
- `TestNormalization`: Per-channel zero-mean and unit-variance z-score normalization.
- `TestChannelNormalization`: 10-20 channel alias mapping (`T7->T3`, `P7->T5`, `CZ->Cz`, etc.).
- `TestCNNForwardPass`: Conv2D temporal, depthwise spatial, separable conv, and output shapes (`[B, 40, 64]`).
- `TestTransformerForwardPass`: CLS token concatenation, sinusoidal positional encoding, attention block.
- `TestEEGClassifierEndToEnd`: Binary (2-class) and multi-class (5-class) forward pass and feature extraction.
- `TestDatasetModuleImports`: Import and callable validation for CHB-MIT, Alzheimer's, Parkinson's, MODMA, and TUAB modules.
- `TestChbmitParseSummary`: Parsing seizure annotation intervals from summary text.

---

## 4. Running Individual Model Modules

You can execute each submodule independently as a standalone test:

### Full Combined Classifier
```powershell
python -m models.eeg_classifier
```

### CNN Feature Extractor Only
```powershell
python -m models.eeg_cnn
```

### Transformer Encoder Only
```powershell
python -m models.transformer
```

### Syntax and Bytecode Validation
Compile all packages to ensure zero compilation or syntax issues:
```powershell
python -m compileall preprocessing models
```

---

## 5. Dataset Preprocessing Guide

### 5.1 Directory Setup
The pipeline expects raw datasets under the `datasets/raw/` directory structure:

```
datasets/
├── raw/
│   ├── epilepsy/       # CHB-MIT EDF files + chbXX-summary.txt
│   ├── alzheimers/     # OpenNeuro ds004504 BIDS dataset + participants.tsv
│   ├── parkinsons/     # OpenNeuro ds004584 BIDS dataset + participants.tsv
│   ├── depression/     # MODMA split archives (.zip.001, etc.)
│   └── tuab/           # TUAB dataset (requires institutional login)
└── processed/
    ├── chbmit/         # Harmonized .npy windows + metadata.csv
    ├── alzheimer/      # Harmonized .npy windows + metadata.csv
    └── parkinson/      # Harmonized .npy windows + metadata.csv
```

To create this folder hierarchy in PowerShell:
```powershell
New-Item -ItemType Directory -Force -Path "datasets/raw/epilepsy", "datasets/raw/alzheimers", "datasets/raw/parkinsons", "datasets/processed"
```

---

### 5.2 CHB-MIT (Epilepsy)
Processes CHB-MIT recordings with 19-channel bipolar montage:

```powershell
python -m preprocessing.datasets.chbmit
```

- **Prerequisites**: Place `chb01_01.edf`, `chb01-summary.txt`, etc., in `datasets/raw/epilepsy/`.
- **Output**: Generates `.npy` window files `[19, 1280]` and a unified `metadata.csv` labeled as `seizure` or `non_seizure` (interictal).

---

### 5.3 OpenNeuro ds004504 (Alzheimer's)
Harmonizes Alzheimer's recordings into standard 10-20 format:

```powershell
python -m preprocessing.datasets.alzheimers
```

- **Prerequisites**: Extract `ds004504` into `datasets/raw/alzheimers/`. Must contain `participants.tsv`.
- **Filtering**: Automatically includes AD and Healthy Controls (HC). **FTD subjects are strictly excluded**.

---

### 5.4 OpenNeuro ds004584 (Parkinson's)
Harmonizes Parkinson's recordings into standard 10-20 format:

```powershell
python -m preprocessing.datasets.parkinsons
```

- **Prerequisites**: Extract `ds004584` into `datasets/raw/parkinsons/`.
- **Labels**: Auto-detects Parkinson's (`PD`) and Healthy (`HC`) labels from `participants.tsv`.

---

## 6. Real-Time Classification (RTC) Execution

The model is optimized for **Real-Time Classification (RTC)** in continuous EEG monitoring.

### 6.1 Real-Time Streaming Parameters
- **Window Length**: 5.0 seconds (1,280 samples at 256 Hz)
- **Hop / Step Size**: 2.5 seconds (50% window overlap)
- **CPU Inference Time**: ~24.8 ms
- **Real-Time Factor (RTF)**:
  $$\text{RTF} = \frac{24.8\text{ ms}}{5000\text{ ms}} = 0.00496 \approx 0.50\%$$

### 6.2 Streaming Loop Pseudocode
To integrate the model into a live EEG hardware feed (e.g. via Lab Streaming Layer or serial stream):

```python
import collections
import numpy as np
import torch
from models.eeg_classifier import EEGClassifier
from preprocessing.common import normalize_windows

# 1. Initialize model
model = EEGClassifier(n_channels=19, n_samples=1280, num_classes=4)
model.eval()

# 2. Setup sliding ring buffer (19 channels x 1280 samples)
BUFFER_SIZE = 1280
STEP_SIZE = 640  # 2.5 seconds @ 256 Hz
ring_buffer = np.zeros((19, BUFFER_SIZE), dtype=np.float32)

def on_eeg_chunk_received(chunk_2_5s):
    global ring_buffer
    # Slide buffer by 2.5s and append new chunk
    ring_buffer = np.roll(ring_buffer, -STEP_SIZE, axis=1)
    ring_buffer[:, -STEP_SIZE:] = chunk_2_5s

    # Normalize per-window
    normed = normalize_windows(ring_buffer[np.newaxis, :, :])
    input_tensor = torch.from_numpy(normed).float()

    # Model inference (< 25 ms)
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1).item()

    return prediction, probabilities
```

---

## 7. Comprehensive Command Reference

| Action | Command | Purpose |
|--------|---------|---------|
| **Run Inference** | `python run_model.py` | Run forward pass on sample EEG batch |
| **Run Training Demo** | `python run_model.py --train-demo` | Run 5 synthetic training steps with AdamW |
| **Run on GPU** | `python run_model.py --device cuda` | Run model on NVIDIA GPU |
| **Run Smoke Tests** | `python test_pipeline.py` | Run 23 automated pipeline tests |
| **Run Tests (Verbose)** | `python -m unittest test_pipeline.py -v` | Run tests with per-test details |
| **Run Classifier Module** | `python -m models.eeg_classifier` | Test end-to-end model directly |
| **Run CNN Module** | `python -m models.eeg_cnn` | Test CNN feature extractor directly |
| **Run Transformer Module** | `python -m models.transformer` | Test Transformer encoder directly |
| **Compile All Modules** | `python -m compileall preprocessing models` | Verify bytecode syntax integrity |
| **CHB-MIT Preprocessing** | `python -m preprocessing.datasets.chbmit` | Preprocess CHB-MIT epilepsy EDFs |
| **Alzheimer's Preprocessing** | `python -m preprocessing.datasets.alzheimers` | Preprocess ds004504 Alzheimer's data |
| **Parkinson's Preprocessing** | `python -m preprocessing.datasets.parkinsons` | Preprocess ds004584 Parkinson's data |

---

## 8. Troubleshooting & FAQ

### Q1: `UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True`
- **Cause**: PyTorch's `nn.TransformerEncoder` issues this warning when using `norm_first=True` (Pre-LN Transformer).
- **Status**: Harmless informational warning. Pre-LN is intentionally chosen for training stability.

### Q2: How do I handle missing Microsoft Visual C++ build tools on Windows?
- **Solution**: The pipeline uses MNE-Python's native EDF reader (`mne.io.read_raw_edf`) instead of `pyEDFlib`, so MSVC build tools are not required for standard preprocessing and inference.

### Q3: What if my raw EEG does not have all 19 channels?
- **Solution**: The `preprocessing/common.py` module includes channel aliasing and reordering. Channels not present in the recording are identified, and files with fewer than 14 recognized channels are safely rejected to prevent degradation.

### Q4: Can I run this on a Raspberry Pi or low-power edge device?
- **Yes**: Total model parameter count is only **81,800 parameters** (~0.32 MB). Memory consumption during inference is under 3 MB, comfortably running within the specs of Raspberry Pi 4/5 or NVIDIA Jetson Nano.
