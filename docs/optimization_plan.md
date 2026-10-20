# EEG Multi-Disease Optimization Plan & Baseline Audit

## 1. Verified Repository Baseline (Commit `9bc1e53`)

- **Classes (5)**: Healthy (0), Epilepsy (1), Alzheimer's (2), Parkinson's (3), Depression (4)
- **Data Configuration**: 19 channels (10-20 system), 256 Hz, 5-second windows (1,280 samples), 50% overlap.
- **Cohort Windows**: Total = 10,488
  - Healthy: 4,110
  - Epilepsy: 438
  - Alzheimer's: 1,080
  - Parkinson's: 3,000
  - Depression: 1,860
- **Partitioning**: Deterministic subject-level split:
  - Train: 7,466 windows (201 subjects)
  - Validation: 1,497 windows (42 subjects)
  - Test: 1,525 windows (42 subjects)
  - Subject Leakage (dataset-qualified): **0** (Train ∩ Val = ∅, Train ∩ Test = ∅, Val ∩ Test = ∅).
  - Epilepsy Partitioning:
    - Train: `chb02`, `chb03`, `chb04`, `chb05`, `chb08` (326 windows)
    - Val: `chb07` (57 windows)
    - Test: `chb01` (55 windows) — *Single held-out subject limitation noted.*
- **Model Architecture**:
  - EEGNet-inspired 1D Spatial-Temporal CNN Backbone + 2-layer Multi-Head Self-Attention Transformer Encoder.
  - Parameters: 81,994 (all trainable).
- **Verified Test Metrics (Held-out Test Split, 1,525 windows)**:
  - Accuracy: **82.49%**
  - Balanced Accuracy: **86.15%**
  - Macro F1-Score: **0.8518**
  - Weighted F1-Score: **0.8271**
  - Macro ROC-AUC: **0.9648**
  - Cohen's Kappa: **0.7572**
- **Per-Class Metrics**:
  - Healthy: Precision 76.92%, Recall 72.22%, F1 0.7450 (Support 540)
  - Epilepsy: Precision 100.0%, Recall 100.0%, F1 1.0000 (Support 55)
  - Alzheimer's: Precision 62.26%, Recall 91.67%, F1 0.7416 (Support 180)
  - Parkinson's: Precision 90.69%, Recall 90.19%, F1 0.9044 (Support 540)
  - Depression: Precision 100.0%, Recall 76.67%, F1 0.8679 (Support 210)
- **Unit Test Suite**: 81/81 unit tests passing (`test_pipeline.py`).

---

## 2. Diagnostic Analysis of Error Patterns

Analysis of the test confusion matrix (`results/confusion_matrix.png`):
- **Zero Disease-to-Disease Confusion**: Not a single sample of Alzheimer's was predicted as Parkinson's, Depression, or Epilepsy; similarly, Parkinson's and Depression are never confused with each other.
- **Confusion is Exclusively Healthy ↔ Disease**:
  - 100 Healthy test windows predicted as Alzheimer's (depressing Alzheimer's precision to 62.26%).
  - 50 Healthy test windows predicted as Parkinson's.
  - 53 Parkinson's test windows predicted as Healthy.
  - 49 Depression test windows predicted as Healthy.
  - 15 Alzheimer's test windows predicted as Healthy.
- **Root Cause**: The Healthy class aggregates 4 different clinical sources (SRM, ds004504, ds004584, NEMAR). The boundary between normative aging EEG and mild neurodegenerative/depressive alterations is subtle, leading to conservative or biased predictions around the Healthy class.

---

## 3. Prioritized Experiment Hypotheses

1. **Exp 1: Loss Function Refinement (Focal Loss / Softmax Temperature)**
   - *Rationale*: Focus hard-example learning on the ambiguous Healthy ↔ Alzheimer's boundary without degrading minority-class recall.
   - *Validation metric*: Validation Macro F1 and Balanced Accuracy.
2. **Exp 2: Class-Weight Smoothing & Label Smoothing**
   - *Rationale*: Mitigate overconfidence on dominant Healthy windows (3,090 in train) while retaining calibrated probability estimates.
3. **Exp 3: Learning Rate & Optimizer Regularization**
   - *Rationale*: Baseline trained for 10 epochs before early stopping (best epoch 4). Slight learning rate schedule adjustments (e.g., warmup 3 epochs, weight decay 5e-3) may yield smoother convergence and avoid premature plateauing on validation.

---

## 4. Execution Protocol & Safeguards

- **Strict Subject Isolation**: No alterations to dataset split or subject assignments.
- **Model Selection Protocol**: Hyperparameters and model variants selected exclusively via Validation Macro F1 and Validation Balanced Accuracy.
- **Final Test Evaluation**: Only the single selected best model will be evaluated once on the test partition.
- **Checkpoint Safety**: `checkpoints/best_model.pt` is backed up before any candidate evaluation.
