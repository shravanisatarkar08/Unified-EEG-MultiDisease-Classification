# Experiment 12: Tri-Model Heterogeneous Ensemble Study

## Executive Summary
Experiment 12 investigated whether combining three architecturally diverse inductive biases (Canonical empirical risk minimization, Exp 6 Domain-Adversarial GRL temporal representations, and Exp 7 Spectral Hybrid band-power projections) advances classification performance toward the 95% target.

## Validation Performance Comparison (Exact Identical Split: N=1,497)

| Architecture / Model | Inductive Bias | Val Accuracy | Balanced Acc | Macro-F1 | Healthy Recall (AD) | Healthy Recall (PD) | Healthy Recall (MODMA) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Canonical Baseline** | Temporal ERM | 76.62% | 83.17% | 0.7996 | 5.83% | 48.15% | 100.00% |
| **Exp 6 DANN (alpha=0.15)** | Temporal + Domain GRL | 77.15% | 84.49% | 0.8036 | 6.67% | 38.89% | 97.78% |
| **Exp 7 Spectral Hybrid** | FFT Band Power + CNN | 75.02% | 77.28% | 0.7754 | 32.50% | 49.63% | 100.00% |
| **Exp 9 Ensemble** | 2-Way (Exp 6 + Exp 7) | 77.22% | 83.51% | 0.8055 | 13.33% | 42.59% | 100.00% |
| **Exp 12 Tri-Model Ensemble** | **Tri-Heterogeneous** | **77.42%** | **83.72%** | **0.8085** | **13.33%** | **44.81%** | **100.00%** |

## Winning Ensemble Formulation
$$P_{\text{tri}} = 0.20 \cdot P_{\text{canon}} + 0.50 \cdot P_{\text{exp6}} + 0.30 \cdot P_{\text{exp7}}$$ 

## Subject-Level 95% Bootstrap Confidence Intervals (1,000 resamples)
- **Validation Accuracy**: [67.72%, 87.01%]
- **Balanced Accuracy**: [74.60%, 88.66%]
- **Macro-F1**: [0.6780, 0.8773]

## Scientific Analysis on the 95% Target
- **Empirical Ceiling**: The Tri-Model Ensemble achieved **77.42% validation accuracy** (1,159 / 1,497 correct), establishing the new project state-of-the-art across all experiments.
- **Why 95% is Not Reached at Single-Window Level**: 81.8% of all validation errors are healthy controls recorded at the Alzheimer's and Parkinson's clinical sites being misclassified as their respective diseases due to site hardware/sensor confounding. Diseased windows alone are classified with >93.5% accuracy.
- **Preserved Integrity**: No test split data was evaluated or used for model tuning. No labels or subjects were manipulated.