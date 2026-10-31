# Experiment 13: Pz-Harmonization Cached-Artifact Feasibility Record

## Decision
**No new validation experiment was run; no candidate is accepted.**

## Highest-impact evidence-supported fix
Harmonize the Parkinson's cohort's missing **Pz** channel before model inference (deterministic scalp interpolation or consistent masking across cohorts). Exp13 established that Pz is zeroed for the Parkinson's cohort and that this acquisition footprint biases healthy controls toward Parkinson's disease.

## Why cached artifacts cannot validly test it
- `exp12_accuracy95/validation_predictions.csv` stores only final tri-ensemble probabilities/predictions, not model inputs or counterfactual Pz-harmonized outputs.
- `exp11_frozen_features/frozen_embeddings_{train,val}.npz` stores post-encoder embeddings. Pz harmonization must occur before encoding, so these embeddings cannot represent the intervention.
- Selecting a probability adjustment using the validation labels would tune on the validation split and is rejected.
- The test split was not read, evaluated, or used.

## Required comparison (no new measured candidate)
| System | Validation accuracy | Balanced accuracy | Macro-F1 | Healthy recall: AD | Healthy recall: PD | Healthy recall: MODMA |
|---|---:|---:|---:|---:|---:|---:|
| Exp12 tri-model ensemble (existing benchmark) | 77.42% | 83.72% | 0.8085 | 13.33% | 44.81% | 100.00% |
| Exp13 Pz-harmonization candidate | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured |

## Integrity
Canonical checkpoint and canonical metrics hashes were verified unchanged before and after this record was written. No checkpoints, prior experiment artifacts, metrics, or backups were overwritten.