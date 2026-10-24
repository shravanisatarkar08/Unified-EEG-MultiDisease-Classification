# Experiment 10: Targeted Healthy-Control Improvement Report

**Date:** 2026-10-24 20:18:17  
**Branch:** `exp10/targeted-improvement`  
**Strategy:** Healthy Margin Disentanglement (HMD) Fine-Tuning on Exp 6 DANN  
**Safety Protocol:** Isolated execution; zero modifications to canonical files or main branch; no remote push.

---

## 1. Motivation & Technical Rationale

Inspection in Step 1 revealed that **77.7% of all validation errors (265 / 341)** are Healthy controls misclassified as their hospital's co-located disease (157 Parkinson Healthy $\rightarrow$ Parkinson's; 108 Alzheimer Healthy $\rightarrow$ Alzheimer's).
This failure mode is exacerbated by canonical loss weights ($w_{\text{AD}} = 1.1589$ vs. $w_{\text{Healthy}} = 0.5594$), which impose a $2.07\times$ higher penalty for false negatives on Alzheimer's than on Healthy.

To counteract this without altering data distribution or discarding samples, **Exp 10** introduces a Healthy Margin Disentanglement (HMD) loss:
$$\mathcal{L} = \mathcal{L}_{\text{CE}} + 0.10 \mathcal{L}_{\text{DANN}} + 0.2 \mathcal{L}_{\text{margin}}$$
where $\mathcal{L}_{\text{margin}} = \text{ReLU}(z_{\text{disease}} - z_{\text{healthy}} + 0.5)$ penalizes margin violations on cohort-matched pairs during training.

---

## 2. Validation Results Comparison

| Model | Val Accuracy | Balanced Acc | Macro-F1 | AD Healthy Rec | PD Healthy Rec | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **canonical_baseline** | 76.62% | 83.17% | 0.7996 | 3.3% | 44.8% |
| **exp6_dann_alpha0.15** | 77.15% | 84.49% | 0.8036 | 6.7% | 38.9% |
| **exp9_ensemble_w0.7** | 77.22% | 83.51% | 0.8055 | 13.3% | 42.6% |
| **exp10_hmd_finetuning** | 76.82% | 83.08% | 0.8042 | 5.8% | 56.3% |

---

## 3. Final Candidate Metrics (`exp10_hmd_finetuning`)

- **Validation Accuracy:** **76.82%** (Subject-level 95% Bootstrap CI: [66.59%, 86.22%])
- **Validation Balanced Accuracy:** **83.08%** (Subject-level 95% Bootstrap CI: [74.52%, 88.37%])
- **Validation Macro-F1:** **0.8042** (Subject-level 95% Bootstrap CI: [0.5463, 0.8671])
- **Validation Weighted-F1:** **0.7643**

### Per-Class Metrics:
```json
{
  "healthy": {
    "precision": 0.693,
    "recall": 0.5125,
    "f1_score": 0.5892,
    "support": 480
  },
  "epilepsy": {
    "precision": 1.0,
    "recall": 0.9474,
    "f1_score": 0.973,
    "support": 57
  },
  "alzheimers": {
    "precision": 0.589,
    "recall": 0.9556,
    "f1_score": 0.7288,
    "support": 180
  },
  "parkinsons": {
    "precision": 0.7287,
    "recall": 0.8128,
    "f1_score": 0.7685,
    "support": 390
  },
  "depression": {
    "precision": 1.0,
    "recall": 0.9256,
    "f1_score": 0.9614,
    "support": 390
  }
}
```

### Confusion Matrix:
```text
               Healthy   Epilepsy  Alzheimers  Parkinsons  Depression
Healthy            246        0          116        118        0         
Epilepsy           1          54         2          0          0         
Alzheimers         8          0          172        0          0         
Parkinsons         71         0          2          317        0         
Depression         29         0          0          0          361       
```

---

## 4. Decision & Safety Conclusion

- **Decision:** **REJECTED**
- **Canonical Files:** Bit-for-bit unchanged.
- **Test Set Protection:** Held-out test set remained strictly quarantined (zero access).
