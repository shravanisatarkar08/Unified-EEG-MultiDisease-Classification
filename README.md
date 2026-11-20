# Transformer-Based Deep Learning Framework for Robust EEG Signal Classification

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-ee4c2c.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.65-ff4b4b.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Unified Explainable CNN–Transformer Framework for Multi-Disease Scalp EEG Classification**

A deep learning research system that classifies scalp EEG recordings into five neurological categories using a CNN–Transformer architecture with gradient-based explainability.

---

## Overview

This project implements an end-to-end pipeline for automated EEG classification:

1. **EEG Harmonization** — Standardizes diverse EEG recordings to a common 19-channel, 256 Hz representation
2. **CNN Feature Extraction** — EEGNet-inspired spatial-temporal convolutional backbone
3. **Transformer Encoding** — Multi-head self-attention for temporal dependency modeling
4. **Explainability** — Grad-CAM, input gradient saliency, and attention visualization
5. **Interactive Dashboard** — Professional Streamlit application for research demonstration

### Classification Categories

| Class | Condition | Data Source |
|:---:|:---|:---|
| 0 | Healthy Control | SRM + OpenNeuro ds004504 + ds004584 + NEMAR |
| 1 | Epilepsy | CHB-MIT Scalp EEG Database |
| 2 | Alzheimer's Disease | OpenNeuro ds004504 |
| 3 | Parkinson's Disease | OpenNeuro ds004584 |
| 4 | Depression (MDD) | NEMAR nm000114 / MODMA |

---

## Model Architecture

```
Input: Raw EEG [B, 19, 1280] (19 channels × 5 seconds @ 256 Hz)
  │
  ├─ CNN Feature Extractor (EEGNet-inspired)
  │   ├─ Temporal Conv2d (1→16, kernel 1×64)
  │   ├─ Depthwise Conv2d (16→32, kernel 19×1)
  │   ├─ Separable Conv2d (32→32, kernel 1×16)
  │   └─ Feature Projection → [B, 40, 64]
  │
  ├─ Transformer Encoder
  │   ├─ CLS Token + Sinusoidal Positional Encoding
  │   ├─ 2× TransformerEncoderLayer (4 heads, 128 FFN, GELU, pre-norm)
  │   └─ CLS output → [B, 64]
  │
  └─ Classification Head
      └─ LayerNorm → Linear(128) → GELU → Dropout → Linear(5)
          → [B, 5] logits
```

**Total parameters:** ~57,000

---

## Evaluation Results

Evaluated on **1,525 held-out test windows** from subjects never seen during training.
Strict subject-level partitioning with zero subject overlap.

| Metric | Score |
|:---|:---:|
| **Overall Accuracy** | 82.49% |
| **Balanced Accuracy** | 86.15% |
| **Macro F1-Score** | 0.8518 |
| **Weighted F1-Score** | 0.8271 |
| **Macro ROC-AUC (OVR)** | 0.9648 |
| **Cohen's Kappa** | 0.7572 |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| Healthy | 0.769 | 0.722 | 0.745 | 540 |
| Epilepsy | 1.000 | 1.000 | 1.000 | 55 |
| Alzheimer's | 0.623 | 0.917 | 0.742 | 180 |
| Parkinson's | 0.907 | 0.902 | 0.904 | 540 |
| Depression | 1.000 | 0.767 | 0.868 | 210 |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/shravanisatarkar08/Unified-EEG-MultiDisease-Classification.git
cd Unified-EEG-MultiDisease-Classification

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.11+
- PyTorch 2.6+ (CPU or CUDA)
- Streamlit 1.30+
- MNE-Python 1.12+
- NumPy, Pandas, Matplotlib, SciPy

---

## Usage

### Launch the Dashboard

```bash
streamlit run app/app.py
```

Opens at **http://localhost:8501** in your browser.

### Supported Upload Formats

| Format | Extension | Notes |
|:---|:---|:---|
| European Data Format | `.edf` | Recommended; full channel auto-detection |
| BioSemi Data Format | `.bdf` | Supported via MNE |
| EEGLAB Dataset | `.set` | Requires EEGLAB-compatible file |
| CSV | `.csv` | Columns = channel names (or 19 numeric columns) |
| NumPy Array | `.npy` | Shape: `[19, 1280]` or `[N, 19, 1280]` |

### Expected EEG Input

- **Channels:** Standard 10-20 montage (19 channels)
  - `Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2`
  - 10-10 names (T7, T8, P7, P8) are automatically aliased
- **Sampling Rate:** Any (auto-resampled to 256 Hz)
- **Minimum Channels:** 10 of 19 (missing channels zero-padded)
- **Window Size:** 5 seconds (1,280 samples at 256 Hz)

### Run Tests

```bash
python -m pytest tests/test_inference_pipeline.py -v
```

---

## Project Structure

```
Unified-EEG-MultiDisease-Classification/
├── app/
│   └── app.py                    # Streamlit dashboard application
├── models/
│   ├── eeg_cnn.py                # EEGNet-style CNN feature extractor
│   ├── transformer.py            # Transformer encoder with CLS token
│   ├── eeg_classifier.py         # Combined CNN-Transformer classifier
│   ├── spectral_transformer.py   # Spectral features variant
│   └── eegnet.py                 # Standalone EEGNet baseline
├── preprocessing/
│   ├── config.py                 # Pipeline configuration constants
│   ├── common.py                 # Filtering, resampling, normalization
│   ├── harmonize.py              # EEG harmonization pipeline
│   ├── segment.py                # Windowing and segmentation
│   └── datasets/                 # Dataset-specific preprocessors
├── training/
│   ├── trainer.py                # Training loop with cosine schedule
│   ├── evaluator.py              # Evaluation and metrics computation
│   ├── dataset.py                # PyTorch dataset with lazy loading
│   ├── explainability.py         # Grad-CAM, saliency, attention
│   └── feature_extraction.py     # CNN feature extraction utilities
├── results/
│   ├── metrics.json              # Complete evaluation metrics
│   ├── confusion_matrix.png      # Test set confusion matrix
│   ├── training_curves.png       # Loss/accuracy trajectories
│   ├── classification_report.csv # Per-class precision/recall/F1
│   ├── gradcam_analysis.png      # Multi-disease Grad-CAM composite
│   ├── gradcam/                  # Individual class Grad-CAM images
│   └── experiments/              # Loss function experiment results
├── checkpoints/
│   └── best_model.pt             # Trained model checkpoint (~1.1 MB)
├── datasets/
│   └── metadata/                 # Dataset inventory and index files
├── tests/
│   └── test_inference_pipeline.py # 26 comprehensive tests
├── scripts/                      # Training and analysis scripts
├── requirements.txt              # Python dependencies
├── DEMO_GUIDE.md                 # Thesis presentation walkthrough
└── README.md                     # This file
```

---

## Dataset

**Total:** 10,488 harmonized windows across 5 classes

| Split | Windows | Protocol |
|:---|:---:|:---|
| Training | 7,466 | Subject-level, balanced class weighting |
| Validation | 1,497 | Subject-level, early stopping criterion |
| Test | 1,525 | Held-out subjects, final evaluation |

All recordings harmonized to: 19-channel 10-20 montage, 256 Hz, 5-second windows, 50% overlap, z-score normalized.

---

## Known Limitations

1. **Epilepsy test cohort:** Test partition contains 55 windows from one held-out subject (CHB-MIT chb01)
2. **Multi-center confounding:** Acquisition hardware differences across cohorts contribute to classification signals
3. **Clinical scope:** Research prototype; not validated for clinical diagnosis
4. **Model capacity:** Lightweight architecture (~57K parameters) designed for efficiency
5. **Healthy reference:** Drawn from 4 diverse sources introducing acquisition heterogeneity

---

## Citation

If you use this work, please cite:

```
@thesis{satarkar2026eeg,
  title={Transformer-Based Deep Learning Framework for Robust EEG Signal Classification},
  author={Satarkar, Shravani},
  year={2026},
  type={Thesis}
}
```

---

## Disclaimer

> **⚠️ Research Prototype:** This software is developed for academic research and computational
> neuroscience investigation. It is **not** approved as a medical device, is **not** certified
> for clinical diagnosis, and must **not** be used as a substitute for professional medical evaluation.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
