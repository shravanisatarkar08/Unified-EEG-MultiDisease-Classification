# Unified Explainable CNN–Transformer Framework for Multi-Disease EEG Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Dashboard-FF4B4B.svg)](http://localhost:8501)
[![Tests: 81 Passed](https://img.shields.io/badge/Tests-81%20Passed-brightgreen.svg)](test_pipeline.py)

A unified deep learning framework combining an **EEGNet-style CNN spatial-temporal backbone** with a **Multi-Head Self-Attention Transformer Encoder** for robust multi-disease classification and biomarker attribution across five clinical EEG categories:
1. **Healthy Controls** (`healthy`, Class 0)
2. **Epilepsy / Seizure EEG** (`epilepsy`, Class 1)
3. **Alzheimer’s Disease** (`alzheimers`, Class 2)
4. **Parkinson’s Disease** (`parkinsons`, Class 3)
5. **Major Depressive Disorder** (`depression`, Class 4)

The framework incorporates full end-to-end data harmonization (standard 19-channel 10–20 montage, 256 Hz, 5-second overlapping windows), rigorous **subject-level partitioning with zero data leakage**, balanced loss optimization, **1-D CNN Grad-CAM explainability**, and an interactive Streamlit clinical dashboard.

---

## Table of Contents
- [Architecture](#architecture)
- [Multi-Cohort Dataset Provenance](#multi-cohort-dataset-provenance)
- [Real 5-Class Experimental Results](#real-5-class-experimental-results)
- [Visual Explainability & Biomarker Attribution (Grad-CAM)](#visual-explainability--biomarker-attribution-grad-cam)
- [Interactive Streamlit Clinical Dashboard](#interactive-streamlit-clinical-dashboard)
- [Quick Start & Reproduction Commands](#quick-start--reproduction-commands)
- [Real-Time Classification (RTC) Profile](#real-time-classification-rtc-profile)
- [Repository Structure](#repository-structure)
- [Scientific Limitations & Methodology](#scientific-limitations--methodology)

---

## Architecture

```
Raw Multi-Channel Scalp EEG (EDF / BDF / SET / FDT)
                     │
                     ▼
       Common Preprocessing & Harmonization
  • 0.5–45.0 Hz Bandpass Filter + 50/60 Hz Notch Filter
  • Resampled to 256 Hz
  • Spatial Montage: 19 Standard 10–20 Channels (Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2)
  • 5-Second Segmentation: [19 Channels × 1,280 Time Samples]
  • Per-Channel Z-Score Normalization (μ = 0, σ = 1)
                     │
                     ▼
      EEGNet-Style CNN Feature Extractor (EEGFeatureExtractor)
  • 1D Temporal Convolution across time (1 × 64, F1=16)
  • Depthwise Spatial Convolution across 19 channels (D=2, F1*D=32)
  • Separable Convolution + ELU Activation + AvgPool2D (1 × 8)
  • Feature Projection: [Batch, 19, 1280] ──> [Batch, 40, 64] Tokens
                     │
                     ▼
      Transformer Encoder with CLS Token (EEGTransformerEncoder)
  • Learnable [CLS] Token prepended: [Batch, 41, 64]
  • Sinusoidal Positional Encoding
  • 2-Layer Pre-Norm Transformer Encoder (4 Attention Heads, FF-Dim=128)
  • Multi-Class Linear Projection Head: [Batch, num_classes=5]
                     │
                     ▼
         Explainable Diagnosis & Attribution
  • Posterior Probabilities for 5 Classes: [Healthy, Epilepsy, Alzheimer's, Parkinson's, Depression]
  • True 1-D CNN Grad-CAM Activation Maps over Temporal Waves
  • Spatial 19-Electrode Gradient Saliency Attributions
```

### Computational Specifications
- **Total Model Parameters**: ~128,000 float32 parameters (~1.1 MB checkpoint size)
- **CPU Forward Inference Latency**: ~24.8 ms per 5-second window (< 0.5% Real-Time Factor)
- **Memory Footprint**: < 2.5 MB peak activation memory for single-window inference

---

## Multi-Cohort Dataset Provenance

| Disease / Condition | Class ID | Clinical Origin / Accession | Total Windows | Train Split | Validation Split | Test Split (Held-Out) | Split Protocol |
|:---|:---:|:---|:---:|:---:|:---:|:---:|:---|
| **Healthy Control** | 0 | SRM Healthy + OpenNeuro `ds004504` Controls + `ds004584` Controls + NEMAR Controls | 3,110 | 2,090 | 480 | 540 | Subject-Level (Zero Overlap) |
| **Epilepsy** | 1 | CHB-MIT (`chb01`, 6 EDF sessions) | 55 | 55 | 0 | 0 | Subject-Level (Train cohort) |
| **Alzheimer's Disease** | 2 | OpenNeuro `ds004504` (AD Patient Cohort) | 1,080 | 720 | 180 | 180 | Subject-Level (Zero Overlap) |
| **Parkinson's Disease** | 3 | OpenNeuro `ds004584` (PD Patient Cohort) | 3,000 | 2,070 | 390 | 540 | Subject-Level (Zero Overlap) |
| **Depression (MDD)** | 4 | NEMAR `nm000114` / MODMA (Depression Cohort) | 1,860 | 1,260 | 390 | 210 | Subject-Level (Zero Overlap) |
| **TOTAL** | - | **Unified Multi-Disease Cohort** | **10,105** | **7,195** | **1,440** | **1,470** | **Strict Zero Subject Leakage** |

---

## Real 5-Class Experimental Results

The unified CNN–Transformer classifier was trained with AdamW optimizer, cosine learning rate scheduling with warmup, and class-weighted CrossEntropy loss to handle cohort balance.

### Quantitative Performance (1,470 Independent Held-Out Test Windows)

| Metric | Score | Clinical / Technical Interpretation |
|:---|:---:|:---|
| **Overall Accuracy** | **78.44%** | Multi-class overall correct diagnostic classification rate |
| **Balanced Accuracy** | **82.27%** | Macro-average of recall across all disease classes |
| **Macro ROC-AUC (OVR)** | **0.9498** | High discriminative capacity across disease vs control curves |
| **Weighted F1-Score** | **0.7824** | Multi-class support-weighted harmonic mean |
| **Cohen's Kappa ($\kappa$)** | **0.6959** | Substantial inter-class agreement exceeding chance |
| **Validation Accuracy** | **74.72%** | Subject-level validation cohort performance |

### Per-Class Performance Breakdown (Held-Out Test Set)

```
Class                 Precision     Recall   F1-Score    Support
-----------------------------------------------------------------
Healthy                  0.7671     0.6037     0.6756        540
Epilepsy*                0.0000     0.0000     0.0000          0
Alzheimer's              0.6020     1.0000     0.7516        180
Parkinson's              0.8360     0.8870     0.8607        540
Depression               0.9825     0.8000     0.8819        210
-----------------------------------------------------------------
Macro Average                                  0.6340       1470
Weighted Average                               0.7824       1470
```
*\*Note: CHB-MIT epilepsy data currently consists of 1 subject (`chb01`), so all 55 epilepsy windows are allocated strictly to the training cohort. Test set evaluates Healthy, Alzheimer's, Parkinson's, and Depression across independent patients.*

---

## Visual Explainability & Biomarker Attribution (Grad-CAM)

The framework includes true **1-D CNN Grad-CAM** and input-gradient saliency attribution:
- **Temporal Grad-CAM**: Computes backpropagation gradients through the CNN feature projection layer to reveal which transient waveform phases triggered the prediction.
- **Electrode Spatial Saliency**: Quantifies attribution across all 19 scalp channels (`Fp1`, `Fp2`, `F7`, `F3`, `Fz`, `F4`, `F8`, `T3`, `C3`, `Cz`, `C4`, `T4`, `T5`, `P3`, `Pz`, `P4`, `T6`, `O1`, `O2`).

Artifacts generated in `results/`:
- **5-Class Confusion Matrix Heatmap**: [`results/confusion_matrix.png`](results/confusion_matrix.png)
- **Training & Validation Trajectories**: [`results/training_curves.png`](results/training_curves.png)
- **Multi-Disease Grad-CAM & Saliency Analysis**: [`results/gradcam_analysis.png`](results/gradcam_analysis.png)
- **Machine-Readable Metrics**: [`results/metrics.json`](results/metrics.json)
- **Experimental Summary**: [`results/FINAL_EXPERIMENT_SUMMARY.md`](results/FINAL_EXPERIMENT_SUMMARY.md)

---

## Interactive Streamlit Clinical Dashboard

Launch the interactive web demonstration:
```bash
streamlit run app/app.py
```

### Dashboard Capabilities:
1. **Interactive Signal Viewer**: Visualizes 19-channel EEG signals over 5-second windows.
2. **5-Class Clinical Predictor**: Loads `checkpoints/best_model.pt` and outputs real-time posterior probabilities.
3. **Spatio-Temporal Biomarker Attribution**: Overlays gradient saliency attributions to highlight salient electrodes and temporal peak activations.
4. **Empirical Validation Panel**: Displays test confusion matrix, training loss/accuracy trajectories, and held-out metrics table.

---

## Quick Start & Reproduction Commands

### 1. Environment Setup

```bash
git clone https://github.com/shravanisatarkar08/Unified-EEG-MultiDisease-Classification.git
cd Unified-EEG-MultiDisease-Classification

# Activate virtual environment
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Build 5-Class Dataset Index

```bash
python scripts/build_new_index.py
```

### 3. Run Full 5-Class Training & Evaluation Pipeline

```bash
python run_new_training.py
```

### 4. Run Test Suite (81 Passed)

```bash
python test_pipeline.py
```

---

## Repository Structure

```
Unified-EEG-MultiDisease-Classification/
├── run_new_training.py        # Real 5-class CNN-Transformer training, metrics & Grad-CAM pipeline
├── run_model.py               # CLI runner for quick inference & demonstration
├── test_pipeline.py           # 81-test validation suite (100% passing)
├── requirements.txt           # Python dependency manifest
├── README.md                  # Project documentation & benchmark results
├── app/
│   └── app.py                 # Streamlit interactive 5-class clinical dashboard
├── checkpoints/
│   └── best_model.pt          # Best trained PyTorch model checkpoint (74.72% val acc)
├── results/
│   ├── confusion_matrix.png   # 5-class normalized test confusion matrix heatmap
│   ├── training_curves.png    # Loss and accuracy trajectories across epochs
│   ├── gradcam_analysis.png   # Multi-class Grad-CAM & 19-channel electrode saliency maps
│   ├── classification_report.csv # Per-class precision, recall, F1-scores
│   ├── classification_report.txt # Text classification report
│   ├── metrics.json           # Machine-readable test metrics summary
│   └── FINAL_EXPERIMENT_SUMMARY.md # Comprehensive experimental research report
├── models/
│   ├── eeg_cnn.py             # EEGNet-style 1D/2D CNN spatial-temporal feature extractor
│   ├── transformer.py         # Multi-head self-attention Transformer encoder
│   └── eeg_classifier.py      # Unified CNN + Transformer 5-class classifier
├── preprocessing/
│   ├── config.py              # Standard 10-20 channel montage, sampling rate (256 Hz), window settings
│   ├── harmonization.py       # Harmonization pipeline (bandpass, notch, 19 channels, z-score)
│   ├── dataset_preprocessors.py # Dataset-specific preprocessors
│   └── dataset_split.py       # Subject-level split generator (zero leakage)
├── scripts/
│   ├── build_new_index.py     # Deterministic 5-class dataset index generator
│   └── run_full_preprocessing.py # Batch dataset preprocessing runner
└── training/
    ├── dataset.py             # PyTorch EEGDataset with fast direct .npy loading
    ├── trainer.py             # Training engine with AdamW, cosine LR, early stopping
    ├── evaluator.py           # Multi-class evaluation metrics & confusion matrix computation
    └── explainability.py      # True CNN Grad-CAM & input gradient saliency
```

---

## Scientific Limitations & Methodology

1. **Epilepsy Cohort Scope**: CHB-MIT scalp EEG currently comprises subject `chb01` (55 windows), which was assigned strictly to the training cohort. Cross-subject epilepsy generalization should be verified on additional CHB-MIT subjects (`chb02`–`chb24`).
2. **Healthy Reference Distribution**: Healthy control data is drawn from four diverse clinical origins (SRM, ds004504, ds004584, NEMAR), strengthening the normative EEG baseline across age and recording hardware variations.
3. **FTD Exclusion**: Frontotemporal Dementia cases in ds004504 were strictly excluded to ensure clean diagnostic separation between Alzheimer's Disease and healthy aging controls.
4. **Subject Leakage Guarantee**: Zero subject overlap exists across training, validation, and test splits (`train_subjects ∩ val_subjects = ∅`, `train_subjects ∩ test_subjects = ∅`, `val_subjects ∩ test_subjects = ∅`).

---

## Citation

```bibtex
@article{unified_eeg_classification_2026,
  title={Unified Explainable CNN--Transformer Framework for Multi-Disease EEG Classification},
  author={Satarkar, Shravani and Kale, Arpit},
  journal={GitHub Repository},
  year={2026},
  url={https://github.com/shravanisatarkar08/Unified-EEG-MultiDisease-Classification}
}
```
