# Experiment 9: Safe Accuracy Improvement Report (Zero-Regression Protocol)

**Date:** 2026-10-24 19:40:27  
**Branch:** `exp9/safe-accuracy-improvement`  
**Evaluation Protocol:** Strict Subject-Level Validation Split (1,497 samples, zero subject overlap)  
**Safety Protocol:** Zero file overwrites, external timestamped backups created and verified before execution.

---

## 1. Safety and Protection Verification

1. **Working Tree at Start:** Clean working tree at commit `3f2b41e810bd67ec688b7025fd2f56f89b895602`.
2. **Canonical Baseline Check:**
   - `checkpoints/best_model.pt`: 1,171,262 bytes (SHA-256: `e8cbd5bd9ec79b6e46524fbc2cad7b84a59c22d83c07a5fe2d335f3021441111`)
   - `results/metrics.json`: 3,744 bytes (SHA-256: `5f0a10003b0fe06b383614c097ab5287ae526a073a90dd420dda2bec1862ff3a`)
3. **External Backup:**
   - Destination: `C:\Users\shrav\Documents\unified_eeg_backups\backup_20261024_1932\`
   - File hashes verified identical to working tree.
4. **Isolated Branch:** All changes made on branch `exp9/safe-accuracy-improvement`.

---

## 2. Complementarity Analysis (Exp 6 DANN vs. Exp 7 Spectral Hybrid)

Analysis of predictions on the 1,497 validation windows revealed significant predictive diversity:
- **Both Models Correct:** 1051 / 1,497 (70.2%)
- **Only Exp 6 Correct:** 104 / 1,497 (6.9%)
- **Only Exp 7 Correct:** 72 / 1,497 (4.8%)
- **Both Models Wrong:** 270 / 1,497 (18.0%)
- **Theoretical Union Oracle Accuracy:** **81.96%**

Exp 7 Spectral Hybrid correctly identified **32.5%** of Alzheimer Healthy controls and **49.6%** of Parkinson Healthy controls, whereas Exp 6 correctly identified higher fractions of disease categories.

---

## 3. Ensemble Results Across Weight Grid

Ensemble formula: $P_{\text{ens}} = w \cdot P_{\text{Exp6}} + (1 - w) \cdot P_{\text{Exp7}}$

| Model / Ensemble | Val Accuracy | Val Balanced Acc | Val Macro-F1 | AD Healthy Rec | PD Healthy Rec |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ensemble_w0.7** | 77.22% | 83.51% | 0.8055 | 13.3% | 42.6% |
| **ensemble_w0.8** | 77.22% | 83.68% | 0.8048 | 10.0% | 41.9% |
| **ensemble_w0.6** | 77.15% | 82.96% | 0.8049 | 17.5% | 43.7% |
| **exp6_dann_alpha0.15** | 77.15% | 84.49% | 0.8036 | 6.7% | 38.9% |
| **ensemble_w0.9** | 76.95% | 84.11% | 0.8037 | 6.7% | 40.4% |
| **ensemble_w0.5** | 76.75% | 82.19% | 0.8013 | 20.8% | 43.7% |
| **canonical_baseline** | 76.62% | 83.17% | 0.7996 | 3.3% | 44.8% |
| **ensemble_w0.4** | 76.02% | 80.92% | 0.7940 | 22.5% | 44.8% |
| **ensemble_w0.3** | 75.28% | 79.06% | 0.7826 | 24.2% | 45.6% |
| **ensemble_w0.1** | 75.15% | 77.74% | 0.7775 | 30.8% | 48.1% |
| **ensemble_w0.2** | 75.08% | 77.86% | 0.7769 | 28.3% | 47.0% |
| **exp7_spectral_hybrid_balanced** | 75.02% | 77.28% | 0.7754 | 32.5% | 49.6% |

---

## 4. Best Ensemble Performance & Confidence Intervals

- **Selected Ensemble:** `ensemble_w0.8` (0.8 * Exp 6 + 0.2 * Exp 7)
- **Validation Accuracy:** **77.22%** (Subject-level 95% Bootstrap CI: [66.44%, 86.81%])
- **Balanced Accuracy:** **83.68%** (Subject-level 95% Bootstrap CI: [75.32%, 88.67%])
- **Macro-F1:** **0.8048** (Subject-level 95% Bootstrap CI: [0.5417, 0.8683])

### Per-Class Metrics (`ensemble_w0.8`):
```json
{
  "healthy": {
    "precision": 0.7465,
    "recall": 0.4479,
    "f1_score": 0.5599,
    "support": 480
  },
  "epilepsy": {
    "precision": 1.0,
    "recall": 0.9649,
    "f1_score": 0.9821,
    "support": 57
  },
  "alzheimers": {
    "precision": 0.6007,
    "recall": 0.9278,
    "f1_score": 0.7293,
    "support": 180
  },
  "parkinsons": {
    "precision": 0.6951,
    "recall": 0.9179,
    "f1_score": 0.7912,
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

### Confusion Matrix (`ensemble_w0.8`):
```text
               Healthy   Epilepsy  Alzheimers  Parkinsons  Depression
Healthy            215        0          108        157        0         
Epilepsy           0          55         2          0          0         
Alzheimers         13         0          167        0          0         
Parkinsons         31         0          1          358        0         
Depression         29         0          0          0          361       
```

---

## 5. Decision & Baseline Comparison

- **Exp 6 Champion Baseline:** 77.15% Val Acc, 84.49% Bal Acc, 0.8036 Macro-F1
- **Canonical Baseline:** 76.62% Val Acc, 83.17% Bal Acc, 0.7996 Macro-F1
- **Ensemble Result:** **77.22%** Val Acc, **83.68%** Bal Acc, **0.8048** Macro-F1
- **Improvement over Exp 6:** **+0.0007** (+0.07%)
- **Improvement over Canonical Baseline:** **+0.0060** (+0.60%)
- **Acceptance Decision:** **ACCEPTED**

---

## 6. Test-Set Protection Disclosure

The held-out test set ($N=1,525$ windows from 42 subjects) was **NOT used** for ensemble weight selection, tuning, or threshold adjustment. It remains strictly quarantined.
