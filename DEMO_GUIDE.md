# Demo Guide — Thesis Presentation

## Transformer-Based Deep Learning Framework for Robust EEG Signal Classification

**Interactive Demonstration Steps for Thesis Evaluation Panel**

---

## Quick Start

```bash
# 1. Navigate to project directory
cd Unified-EEG-MultiDisease-Classification

# 2. Ensure dependencies are installed
pip install -r requirements.txt

# 3. Verify checkpoint exists
ls checkpoints/best_model.pt

# 4. Launch the application
streamlit run app/app.py
```

The app will open automatically at **http://localhost:8501**

---

## Demonstration Walkthrough

### Step 1: Home Page (2 minutes)

1. The dashboard opens showing the **project overview**
2. Point out the **five classification categories** with disease cards
3. Show the **CNN–Transformer architecture diagram**
4. Highlight the **evaluation metrics** from the held-out test set:
   - 82.49% accuracy
   - 86.15% balanced accuracy
   - 0.8518 macro F1
   - 0.9648 ROC-AUC
5. Note the **research disclaimer** at the bottom

### Step 2: EEG Prediction — Main Demo (5 minutes)

1. Click **🎯 EEG Prediction** in the sidebar
2. **Upload an EEG file** (EDF, BDF, SET, CSV, or NPY format)
   - If you have an EDF file from one of the research datasets, use that
   - If not, use the **Quick Demo** button with synthetic data
3. Watch the **preprocessing pipeline steps** expand:
   - File loading
   - Channel mapping to 10-20 montage
   - Resampling to 256 Hz
   - Windowing into 5-second segments
   - Z-score normalization
4. Show the **EEG signal preview** — toggle channel count
5. If multiple windows exist, demonstrate the **window slider**
6. Click **Run Classification** and show:
   - Predicted class with confidence
   - Full probability distribution bar chart
   - Detailed probability table
7. Point out the **research disclaimer** under predictions

### Step 3: Explainable AI (3 minutes)

1. Click **🔍 Explainable AI** in the sidebar
2. Show the **pre-computed Grad-CAM analysis** (5-class composite)
3. Use the **class selector** to view individual condition profiles
4. Click **Generate Live Attribution** to demonstrate real-time:
   - Grad-CAM temporal activation
   - Channel × time saliency heatmap
   - Transformer attention weights
5. Explain what each visualization represents (see expansion panel)

### Step 4: Model Performance (2 minutes)

1. Click **📊 Model Performance** in the sidebar
2. Show the **six evaluation metrics** cards
3. Display the **confusion matrix** — point out strong diagonal
4. Show **training curves** (loss and accuracy over epochs)
5. Review the **per-class classification report**
6. Scroll to the **dataset composition table** showing all cohorts
7. If time allows, show the **loss function experiments** comparison

### Step 5: About & Research (2 minutes)

1. Click **🔬 About & Research** in the sidebar
2. Walk through:
   - Problem statement
   - Proposed architecture
   - EEG harmonization pipeline
   - Five-class classification design
   - Explainability methods
3. Discuss **limitations** honestly:
   - Small epilepsy test cohort
   - Multi-center confounding
   - Not clinically validated
4. Mention **future work** directions

---

## Troubleshooting

| Issue | Solution |
|:---|:---|
| "Checkpoint not found" | Copy `best_model.pt` to `checkpoints/` |
| "No module named 'mne'" | Run `pip install mne` |
| App won't start | Check `streamlit` is installed: `pip install streamlit` |
| Slow loading on first run | Model loads once and is cached for the session |
| EDF upload fails | Ensure the file has standard 10-20 channel names |

## Supported EEG Formats

| Format | Extension | Notes |
|:---|:---|:---|
| European Data Format | `.edf` | Recommended format; full channel detection |
| BioSemi Data Format | `.bdf` | Supported via MNE |
| EEGLAB Dataset | `.set` | Requires EEGLAB-compatible export |
| CSV | `.csv` | Columns = channels, rows = time samples |
| NumPy Array | `.npy` | Shape: [19, 1280] or [N, 19, 1280] |

## Expected Input Specification

- **Channels:** 19 standard 10-20 (Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2)
- **Sampling Rate:** Any (auto-resampled to 256 Hz)
- **Minimum Channels:** 10 of 19 (missing channels are zero-padded)
- **Window Size:** 5 seconds (1,280 samples at 256 Hz)

---

## Key Talking Points for Panel Questions

1. **"Why CNN + Transformer?"** — CNN captures local spatial-temporal patterns; Transformer captures long-range temporal dependencies via self-attention
2. **"How do you handle different EEG setups?"** — Harmonization pipeline maps diverse channel names (10-10 → 10-20), resamples frequencies, and standardizes montage
3. **"What about data leakage?"** — Strict subject-level partitioning with zero overlap between train/val/test
4. **"Why only 82% accuracy?"** — Honest multi-cohort evaluation without inflating metrics; competitive for heterogeneous multi-center EEG data
5. **"Is this clinically usable?"** — No. It's a research prototype demonstrating the approach; clinical validation would require prospective trials
