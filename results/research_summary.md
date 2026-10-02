# Experimental Results: Unified CNN–Transformer Multi-Disease EEG Classification

**Date:** 2026-10-02 14:46:58  
**Model Architecture:** EEGNet-style CNN Backbone + Multi-Head Self-Attention Transformer Encoder  
**Active Classes:** Healthy (EEGMMIDB) vs. Epilepsy (CHB-MIT chb01)  

---

## 1. Dataset Partition & Cohort Statistics

| Partition | Total Windows | Healthy (EEGMMIDB) | Epilepsy (CHB-MIT) | Split Protocol |
|:---|:---:|:---:|:---:|:---|
| **Train** | 1154 | 624 | 530 | Subject-level (Healthy) / Session-level (Epilepsy, 11 files) |
| **Validation** | 301 | 144 | 157 | Subject-level (Healthy) / Session-level (Epilepsy, 3 files) |
| **Test (Held-out)** | 301 | 144 | 157 | Subject-level (Healthy) / Session-level (Epilepsy, 3 files) |
| **Total** | 1756 | 912 | 844 | 19 Channels, 256 Hz, 5-second windows (1280 samples) |

---

## 2. Test Set Classification Performance

| Metric | Score | Clinical Interpretation |
|:---|:---:|:---|
| **Accuracy** | **100.00%** | Overall sample classification rate |
| **Balanced Accuracy** | **100.00%** | Macro-average of class recalls |
| **Macro F1-Score** | **1.0000** | Unweighted mean of Healthy and Epilepsy F1 |
| **Weighted F1-Score** | **1.0000** | Support-weighted harmonic mean |
| **Sensitivity (Recall)** | **100.00%** | True positive rate for epileptic seizure / abnormal EEG |
| **Specificity** | **100.00%** | True negative rate for healthy control EEG |
| **ROC-AUC** | **1.0000** | Area under the ROC curve |
| **Cohen's Kappa (κ)** | **1.0000** | Inter-rater agreement above chance |

---

## 3. Confusion Matrix (Test Split)

```
                 Predicted Healthy    Predicted Epilepsy
Actual Healthy          144                  0
Actual Epilepsy           0                157
```

- **True Negatives (Healthy correctly identified):** 144 / 144 (100.0%)
- **True Positives (Epilepsy correctly identified):** 157 / 157 (100.0%)
- **False Positives:** 0
- **False Negatives:** 0

---

## 4. Visual Artifacts Produced

1. **Confusion Matrix Heatmap:** [`results/confusion_matrix.png`](file:///C:/Users/ARPIT KALE/Desktop/Unified-EEG-MultiDisease-Classification/results/confusion_matrix.png)
2. **Loss & Accuracy Trajectories:** [`results/training_curves.png`](file:///C:/Users/ARPIT KALE/Desktop/Unified-EEG-MultiDisease-Classification/results/training_curves.png)
3. **True CNN Grad-CAM & Saliency Analysis:** [`results/gradcam_analysis.png`](file:///C:/Users/ARPIT KALE/Desktop/Unified-EEG-MultiDisease-Classification/results/gradcam_analysis.png)
4. **Machine-Readable Metrics:** [`results/metrics.json`](file:///C:/Users/ARPIT KALE/Desktop/Unified-EEG-MultiDisease-Classification/results/metrics.json)
5. **Model Checkpoint:** [`checkpoints/best_model.pt`](file:///C:/Users/ARPIT KALE/Desktop/Unified-EEG-MultiDisease-Classification/checkpoints/best_model.pt)

---

## 5. Methodological & Scientific Limitations (For Research Paper)

1. **Epilepsy Cohort Scope:** CHB-MIT data currently comprises 17 multi-hour recordings from subject `chb01`. Recordings were partitioned across train (11), val (3), and test (3) splits, ensuring seizure discharge events exist in all sets. Cross-patient generalization requires incorporating additional CHB subjects.
2. **Absent Disease Classes:** Alzheimer's Disease and Parkinson's Disease EEG cohorts were not present in the workspace data repositories; the 5-class model architecture is preserved, and remaining classes can be introduced without structural modifications.
3. **Depression Proxy:** The DEAP dataset requires credentialed access via the QMUL data agreement. The preprocessor (`preprocessing/datasets/deap_preprocessor.py`) is implemented and tested to ingest DEAP `.dat` files once downloaded.
