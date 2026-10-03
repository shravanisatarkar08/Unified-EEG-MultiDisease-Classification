# Final Experimental Summary: Unified CNN–Transformer Multi-Disease EEG Classification

**Date:** 2026-10-03 10:34:38  
**Framework:** Unified Explainable CNN–Transformer Framework for Multi-Disease EEG Classification  
**Architecture:** 1D Spatial-Temporal CNN Backbone + Multi-Head Self-Attention Transformer Encoder  
**Number of Target Classes:** 5 Classes (Healthy, Epilepsy, Alzheimer's Disease, Parkinson's Disease, Depression)  
**Total Harmonized Windows:** 10,105 windows (19 channels, 256 Hz, 5-second windows = 1,280 timepoints)  

---

## 1. Multi-Cohort Dataset Provenance & Subject-Level Partition

All EEG recordings are harmonized to the standard 19-channel 10-20 montage (`Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2`), bandpass filtered (0.5–45 Hz), notch filtered (50/60 Hz), resampled to 256 Hz, segmented into 5-second windows with 50% overlap, and normalized via robust z-score standardization.

| Disease / Condition | Class ID | Clinical Source / Accession | Total Windows | Train Split | Validation Split | Test Split (Held-Out) | Split Protocol |
|:---|:---:|:---|:---:|:---:|:---:|:---:|:---|
| **Healthy Control** | 0 | SRM + ds004504 + ds004584 + NEMAR | 3,110 | 2,090 | 480 | 540 | Subject-Level (Zero Overlap) |
| **Epilepsy** | 1 | CHB-MIT (`chb01`) | 55 | 55 | 0 | 0 | Subject-Level (Train cohort) |
| **Alzheimer's Disease** | 2 | OpenNeuro `ds004504` (AD Cohort) | 1,080 | 720 | 180 | 180 | Subject-Level (Zero Overlap) |
| **Parkinson's Disease** | 3 | OpenNeuro `ds004584` (PD Cohort) | 3,000 | 2,070 | 390 | 540 | Subject-Level (Zero Overlap) |
| **Depression (MDD)** | 4 | NEMAR `nm000114` / MODMA Cohort | 1,860 | 1,260 | 390 | 210 | Subject-Level (Zero Overlap) |
| **TOTAL** | - | **Unified 5-Cohort Benchmark** | **10,105** | **7,195** | **1,440** | **1,470** | **Strict Zero Subject Leakage** |

---

## 2. Held-Out Test Set Performance

The model was evaluated on **1,470 independent test windows** from held-out subjects never seen during training or hyperparameter tuning.

| Metric | Score | Clinical / Technical Interpretation |
|:---|:---:|:---|
| **Overall Accuracy** | **78.44%** | Overall correct window prediction rate across classes |
| **Balanced Accuracy** | **82.27%** | Macro-average of recall across all disease classes |
| **Macro F1-Score** | **0.6340** | Harmonic mean of precision and recall (unweighted) |
| **Weighted F1-Score** | **0.7824** | Support-weighted multi-class F1-score |
| **Macro ROC-AUC (OVR)** | **0.9498** | Multi-class One-vs-Rest Area Under the ROC Curve |
| **Cohen's Kappa (κ)** | **0.6959** | Inter-class agreement metric exceeding chance agreement |

---

## 3. Per-Class Detailed Classification Report

```
Class                 Precision     Recall   F1-Score    Support
-----------------------------------------------------------------
Healthy                  0.7671     0.6037     0.6756        540
Epilepsy                 0.0000     0.0000     0.0000          0
Alzheimer's              0.6020     1.0000     0.7516        180
Parkinson's              0.8360     0.8870     0.8607        540
Depression               0.9825     0.8000     0.8819        210
-----------------------------------------------------------------
Macro Average                                  0.6340       1470
Weighted Average                               0.7824       1470
```

---

## 4. Visual Artifacts Generated

1. **Confusion Matrix Heatmap:** [`results/confusion_matrix.png`](file:///C:/Users/shrav/Documents/Unified-EEG-MultiDisease-Classification/results/confusion_matrix.png)
2. **Loss & Accuracy Trajectories:** [`results/training_curves.png`](file:///C:/Users/shrav/Documents/Unified-EEG-MultiDisease-Classification/results/training_curves.png)
3. **Multi-Disease Grad-CAM & Saliency Maps:** [`results/gradcam_analysis.png`](file:///C:/Users/shrav/Documents/Unified-EEG-MultiDisease-Classification/results/gradcam_analysis.png)
4. **Per-Class Metrics CSV:** [`results/classification_report.csv`](file:///C:/Users/shrav/Documents/Unified-EEG-MultiDisease-Classification/results/classification_report.csv)
5. **Full Machine-Readable JSON:** [`results/metrics.json`](file:///C:/Users/shrav/Documents/Unified-EEG-MultiDisease-Classification/results/metrics.json)
6. **PyTorch Model Checkpoint:** [`checkpoints/best_model.pt`](file:///C:/Users/shrav/Documents/Unified-EEG-MultiDisease-Classification/checkpoints/best_model.pt)

---

## 5. Scientific Limitations & Future Directions

1. **Epilepsy Generalization:** CHB-MIT currently has 1 local subject (`chb01`), so all 55 windows reside in the Train partition. Testing on additional CHB-MIT subjects (`chb02`–`chb24`) will provide multi-subject validation for epilepsy.
2. **Healthy Reference Distribution:** Healthy control data is drawn from four diverse clinical origins (SRM, ds004504, ds004584, NEMAR), strengthening the normative EEG baseline across age and recording hardware variations.
3. **FTD Exclusion:** Frontotemporal Dementia cases in ds004504 were strictly excluded to ensure clean diagnostic separation between Alzheimer's Disease and healthy aging controls.
