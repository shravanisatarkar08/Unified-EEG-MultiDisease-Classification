# Experiment 11: Frozen-Feature Classifier Study

## Summary
This experiment evaluated whether training regularized linear/multinomial classifiers on frozen CLS embeddings from trained CNN-Transformer models improves validation performance beyond the canonical baseline (76.62%), Exp 6 DANN (77.15%), and Exp 9 ensemble (77.22%).

## Validation Performance Comparison

| System / Classifier | Features | Feature Dim | Val Accuracy | Balanced Acc | Macro-F1 | Healthy Recall (AD) | Healthy Recall (PD) | Healthy Recall (MODMA) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Canonical Baseline** | End-to-end | - | 76.62% | 83.17% | 0.7996 | 4.17% | 40.74% | 98.89% |
| **Exp 6 DANN (alpha=0.15)** | End-to-end | - | 77.15% | 84.49% | 0.8036 | 6.67% | 38.89% | 97.78% |
| **Exp 9 Ensemble** | Prob Blend | - | **77.22%** | 83.51% | **0.8055** | 10.00% | 40.74% | 100.0% |
| logreg_exp6_cls | Exp 6 CLS | 64 | 73.41% | 75.29% | 0.7659 | 51.67% | 61.11% | 100.00% |
| logreg_balanced_exp6_cls | Exp 6 CLS | 64 | 75.75% | 81.87% | 0.7965 | 15.00% | 52.59% | 98.89% |
| logreg_exp7_cls | Exp 7 CLS | 64 | 74.22% | 72.91% | 0.7533 | 57.50% | 54.44% | 98.89% |
| logreg_balanced_exp7_cls | Exp 7 CLS | 64 | 75.89% | 77.19% | 0.7706 | 31.67% | 49.26% | 98.89% |
| logreg_concat_exp6_exp7 | Concat (Exp6+7) | 128 | 74.22% | 76.31% | 0.7758 | 54.17% | 55.93% | 100.00% |
| logreg_balanced_concat_exp6_exp7 | Concat (Exp6+7) | 128 | 76.09% | 80.77% | 0.7986 | 35.00% | 50.00% | 98.89% |
| ridge_exp6_cls | Exp 6 CLS | 64 | 75.28% | 77.67% | 0.7815 | 42.50% | 56.67% | 100.00% |
| ridge_concat_exp6_exp7 | Concat (Exp6+7) | 128 | 74.01% | 76.35% | 0.7690 | 40.00% | 47.78% | 100.00% |

## Key Findings
- All scaling and classifier hyperparameters were fit strictly on the training set using 5-fold cross-validation.
- Zero data leakage across subject boundaries was maintained.
- Held-out test set remained strictly untouched.