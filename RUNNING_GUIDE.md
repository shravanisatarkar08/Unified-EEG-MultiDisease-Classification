# Running Guide: Unified Explainable CNN–Transformer EEG Framework

Complete step-by-step instructions for data downloading, preprocessing, training on real EEG datasets, evaluating classification performance, generating true CNN Grad-CAM explainability maps, and running the interactive web application.

---

## Table of Contents
1. [Quick Start (Execution of Real Training Pipeline)](#quick-start-execution-of-real-training-pipeline)
2. [Environment Setup & Installation](#1-environment-setup--installation)
3. [Real Data Ingestion & Index Generation](#2-real-data-ingestion--index-generation)
   - [2.1 Download Healthy Baseline EEG (EEGMMIDB)](#21-download-healthy-baseline-eeg-eegmmidb)
   - [2.2 Download Epilepsy Recordings (CHB-MIT)](#22-download-epilepsy-recordings-chb-mit)
   - [2.3 Generate Balanced Dataset Index (Zero Leakage)](#23-generate-balanced-dataset-index-zero-leakage)
4. [Training & Evaluating the CNN–Transformer Model](#3-training--evaluating-the-cnn-transformer-model)
   - [3.1 Full End-to-End Training on Real Data](#31-full-end-to-end-training-on-real-data)
   - [3.2 Fast Evaluation & Grad-CAM from Checkpoint](#32-fast-evaluation--grad-cam-from-checkpoint)
   - [3.3 Output Artifacts & Results Directory](#33-output-artifacts--results-directory)
5. [Interactive Streamlit Clinical Dashboard](#4-interactive-streamlit-clinical-dashboard)
6. [Standalone Demos & Unit Testing](#5-standalone-demos--unit-testing)
7. [Comprehensive Command Reference](#6-comprehensive-command-reference)
8. [Troubleshooting & Common Questions](#7-troubleshooting--common-questions)

---

## Quick Start (Execution of Real Training Pipeline)

To run the complete real research pipeline on real scalp EEG data:

```powershell
# 1. Download real healthy baseline EEG (PhysioNet EEGMMIDB, 38 subjects, runs 1-2)
python scripts/download_eegmmidb_fast.py

# 2. Download CHB-MIT epilepsy recordings (chb01, 17 EDF recordings)
python scripts/download_chbmit.py

# 3. Build unified 19-channel 256 Hz dataset index (1,756 balanced windows)
python scripts/build_new_index.py

# 4. Run real CNN-Transformer training, metrics calculation, and true Grad-CAM
python run_new_training.py

# 5. Launch interactive diagnosis & biomarker dashboard
streamlit run app/app.py
```

---

## 1. Environment Setup & Installation

### 1.1 Python Version
- **Python 3.10 to 3.13** (64-bit recommended).

### 1.2 Install Required Dependencies

#### Standard CPU Setup:
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

#### GPU (CUDA) Acceleration (Optional):
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 1.3 Verify Environment
```powershell
python -c "import torch, mne, sklearn, matplotlib; print(f'PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()} | MNE: {mne.__version__}')"
```

---

## 2. Real Data Ingestion & Index Generation

### 2.1 Download Healthy Baseline EEG (EEGMMIDB)
The EEG Motor Movement/Imagery Database (EEGMMIDB) from PhysioNet provides high-density scalp recordings from 109 healthy volunteers. Runs 1 and 2 represent clean baseline resting-state EEG (eyes-open and eyes-closed).

Run the concurrent downloader (downloads 38 subjects, 76 EDF files in < 1 minute):
```powershell
python scripts/download_eegmmidb_fast.py
```
*Files are saved to standard MNE cache directory: `~/mne_data/MNE-eegbci-data/files/eegmmidb/1.0.0/`.*

### 2.2 Download Epilepsy Recordings (CHB-MIT)
The CHB-MIT Scalp EEG Database contains long-term pediatric continuous recordings from Children's Hospital Boston. Subject `chb01` has 17 recordings (~1 hour each), including 7 seizure discharge sessions with verified clinical onset/offset annotations.

```powershell
python scripts/download_chbmit.py
```
*Files are saved to `datasets/raw/epilepsy/chb01/`.*

### 2.3 Generate Balanced Dataset Index (Zero Leakage)
Builds the unified index `datasets/metadata/model_dataset_index.csv`:
```powershell
python scripts/build_new_index.py
```

#### What this script enforces:
1. **Spatial Harmonization**: Harmonizes electrode channels to the standard 19-channel 10–20 system (`Fp1`, `Fp2`, `F7`, `F3`, `Fz`, `F4`, `F8`, `T3`, `C3`, `Cz`, `C4`, `T4`, `T5`, `P3`, `Pz`, `P4`, `T6`, `O1`, `O2`).
2. **Frequency Standardization**: Standardizes sampling rates to 256 Hz.
3. **Windowing**: Creates non-overlapping 5.0-second windows (1,280 samples).
4. **Class Balance**: Samples ~45 windows per CHB-MIT recording (prioritizing confirmed seizure episodes) to match healthy baseline windows (912 Healthy vs 844 Epilepsy).
5. **Strict Partitioning**:
   - **Train (70%)**: 1,154 windows (624 Healthy, 530 Epilepsy)
   - **Val (15%)**: 301 windows (144 Healthy, 157 Epilepsy)
   - **Test (15%)**: 301 windows (144 Healthy, 157 Epilepsy)
   *Zero subject overlap in Healthy, and separate recording sessions in Epilepsy.*

---

## 3. Training & Evaluating the CNN–Transformer Model

### 3.1 Full End-to-End Training on Real Data
Runs the AdamW optimizer with cosine learning rate warm-up, early stopping (patience=10), best-checkpoint tracking, full held-out test evaluation, confusion matrix plotting, and true CNN Grad-CAM generation:

```powershell
python run_new_training.py
```

#### Training Profile & Trajectory:
- **Batch Size**: 32
- **Initial LR**: 3e-4 with 2-epoch linear warmup
- **Epoch Time**: ~25–35 seconds per epoch on CPU
- **Convergence**: Reaches 100.0% validation accuracy by Epoch 3; triggers early stopping at Epoch 13.
- **Total Training Duration**: ~6.6 minutes.

### 3.2 Fast Evaluation & Grad-CAM from Checkpoint
If the model checkpoint already exists in `checkpoints/`, you can evaluate test metrics, re-plot the confusion matrix, and generate Grad-CAM explainability maps in seconds without re-training:

```powershell
python run_new_training.py --eval-only
```

### 3.3 Output Artifacts & Results Directory
After execution, all deliverables are automatically saved in `results/` and `checkpoints/`:

| Artifact | File Path | Description |
|:---|:---|:---|
| **Confusion Matrix** | `results/confusion_matrix.png` | Normalized & count 2×2 confusion matrix on held-out test split |
| **Training Curves** | `results/training_curves.png` | Loss and accuracy curves over all epochs |
| **Grad-CAM Analysis** | `results/gradcam_analysis.png` | True 1-D CNN Grad-CAM waveform activation & channel saliency bar chart |
| **Classification Report** | `results/classification_report.txt` | Precision, recall, and F1-score breakdown |
| **Metrics Summary** | `results/metrics.json` | JSON format with accuracy, sensitivity, specificity, kappa, and AUC |
| **Paper Summary** | `results/research_summary.md` | Markdown formatted summary ready for publication |
| **Trained Weights** | `checkpoints/best_model_ep003_acc1.0000.pt` | PyTorch checkpoint dictionary |

---

## 4. Interactive Streamlit Clinical Dashboard

Launch the browser-based clinical prototype dashboard:

```powershell
streamlit run app/app.py
```
*Access the live dashboard at: **[http://localhost:8501](http://localhost:8501)***

### Dashboard Features:
1. **Real Data Explorer (Sidebar)**: Seamlessly select from 17 local CHB-MIT epilepsy recordings or 38 local EEGMMIDB healthy control subjects.
2. **Interactive Signal Viewer**: Visualizes 19-channel EEG signals over 5-second windows.
3. **Real-Time Classification**: Automatically loads our trained checkpoint (`best_model_ep003_acc1.0000.pt`) and performs forward inference in < 25 ms.
4. **Spatio-Temporal Biomarker Attribution**: Overlays gradient saliency attributions to highlight salient electrodes (e.g. Fp1, F3, C3, Cz, T3) and plots the true 1-D CNN Grad-CAM temporal activation curve.
5. **Real Empirical Validation Suite (Section 10)**: Displays test confusion matrix, training loss/accuracy trajectories, and held-out metrics table directly in the browser.

---

## 5. Standalone Demos & Unit Testing

### 5.1 Quick Forward Pass Demo
Simulates a forward pass with latency benchmarks:
```powershell
python run_model.py
```

### 5.2 Synthetic Training Demonstration
Verifies backpropagation gradients:
```powershell
python run_model.py --train-demo
```

### 5.3 Explainability Smoke Test
Computes input gradients, integrated gradients, and attention:
```powershell
python run_model.py --explain-demo
```

### 5.4 Run Automated Unit Test Suite
Executes the comprehensive validation suite:
```powershell
pytest
```

---

## 6. Comprehensive Command Reference

| Task / Goal | Command Line | Notes |
|:---|:---|:---|
| **Install Dependencies** | `pip install -r requirements.txt` | Installs PyTorch, MNE, scikit-learn, etc. |
| **Download Healthy Data** | `python scripts/download_eegmmidb_fast.py` | Downloads 38 subjects from PhysioNet |
| **Download Epilepsy Data**| `python scripts/download_chbmit.py` | Downloads CHB-MIT chb01 recordings |
| **Build Dataset Index** | `python scripts/build_new_index.py` | Indexes 1,756 windows into 70/15/15 splits |
| **Train Real Model** | `python run_new_training.py` | Full training, evaluation & Grad-CAM |
| **Evaluate Checkpoint** | `python run_new_training.py --eval-only` | Instant test metrics & Grad-CAM plots |
| **Launch Dashboard** | `streamlit run app/app.py` | Web UI for live testing & explainability |
| **Forward Pass Demo** | `python run_model.py` | Latency and tensor shape verification |
| **Run Unit Tests** | `pytest` | Validates models, layers & preprocessors |

---

## 7. Troubleshooting & Common Questions

#### 1. Why use MNE's built-in EDF reader instead of pyEDFlib on Windows?
`pyEDFlib` requires Microsoft Visual C++ build tools to compile on Windows. MNE-Python includes a pure-Python EDF reader (`mne.io.read_raw_edf`) that does not require compilation.

#### 2. How is data leakage prevented between train, val, and test splits?
For healthy controls (EEGMMIDB), subjects are partitioned into non-overlapping groups (26 train, 6 val, 6 test). For epilepsy recordings (CHB-MIT chb01), 17 multi-hour recordings are partitioned into separate sessions (11 train, 3 val, 3 test) so test evaluation occurs on independent recording sessions.

#### 3. Where are the trained model checkpoints saved?
Trained model checkpoints are stored in `checkpoints/best_model_ep003_acc1.0000.pt`.
