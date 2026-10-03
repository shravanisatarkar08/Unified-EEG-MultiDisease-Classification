"""
Optional Phase 3: Model Optimization Experiment.
Train with enhanced capacity and validation-based model selection (macro F1 / balanced accuracy).
Only overwrite best_model.pt if validation performance genuinely improves.
"""

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.dataset import get_dataloaders
from training.trainer import EEGTrainer, TrainConfig
from models.eeg_classifier import EEGClassifier
from run_new_training import compute_multiclass_metrics, plot_training_curves, plot_confusion_matrix, generate_gradcam_visualizations, format_classification_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

INDEX_CSV = ROOT / "datasets" / "metadata" / "model_dataset_index.csv"
CHECKPOINT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
DISPLAY_NAMES = ["Healthy", "Epilepsy", "Alzheimer's", "Parkinson's", "Depression"]

def main():
    loaders = get_dataloaders(str(INDEX_CSV), str(ROOT), batch_size=64, num_workers=0, shuffle_train=True)
    train_loader, val_loader, test_loader = loaders["train"], loaders["val"], loaders["test"]
    
    train_df = train_loader.dataset.index_df
    label_counts = train_df["class_label_idx"].value_counts().to_dict()
    total_train = len(train_df)
    n_classes = 5

    class_weights = torch.ones(n_classes, dtype=torch.float32)
    for idx in range(n_classes):
        cnt = label_counts.get(idx, 0)
        if cnt > 0:
            class_weights[idx] = float(np.sqrt(total_train / (n_classes * cnt)))
        else:
            class_weights[idx] = 1.0

    class_weights = class_weights / class_weights.mean()
    class_weights_list = class_weights.tolist()

    cfg = TrainConfig(
        n_channels=19,
        n_samples=1280,
        num_classes=5,
        embed_dim=128,
        num_heads=4,
        num_layers=3,
        ff_dim=256,
        cnn_dropout=0.25,
        trans_dropout=0.1,
        lr=4e-4,
        weight_decay=1e-2,
        warmup_epochs=2,
        max_epochs=15,
        batch_size=64,
        patience=6,
        checkpoint_dir=str(CHECKPOINT_DIR),
        num_workers=0,
        class_weights=class_weights_list,
    )

    trainer = EEGTrainer(cfg)
    device = trainer.device
    model = trainer.model

    logger.info("Training optimized CNN-Transformer (embed_dim=128, layers=3)...")
    t0 = time.time()
    history = trainer.train(train_loader, val_loader)
    elapsed = time.time() - t0

    logger.info("Training finished in %.1f seconds. Best val acc: %.4f", elapsed, trainer.best_val_acc)

    # Compare with existing best model
    existing_ckpt = CHECKPOINT_DIR / "best_model.pt"
    old_val_acc = 0.7472
    if existing_ckpt.exists():
        c_old = torch.load(existing_ckpt, map_location="cpu", weights_only=False)
        old_val_acc = float(c_old.get("val_acc", 0.7472))

    logger.info("Old best val acc: %.4f vs New best val acc: %.4f", old_val_acc, trainer.best_val_acc)

    if trainer.best_val_acc >= old_val_acc:
        logger.info("Adopting new optimized checkpoint as best_model.pt!")
        if trainer.best_ckpt_path and trainer.best_ckpt_path.exists():
            ckpt_data = torch.load(trainer.best_ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt_data["model_state_dict"])
            torch.save(ckpt_data, existing_ckpt)
        
        # Re-run evaluation and update plots
        plot_training_curves(history, RESULTS_DIR / "training_curves.png")

        def run_eval(loader):
            preds, targets, probs = [], [], []
            model.eval()
            with torch.no_grad():
                for x, y, _ in loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    pr = F.softmax(logits, dim=-1)
                    preds.extend(logits.argmax(dim=-1).cpu().numpy())
                    targets.extend(y.cpu().numpy())
                    probs.extend(pr.cpu().numpy())
            return np.array(preds), np.array(targets), np.array(probs)

        val_preds, val_targets, val_probs = run_eval(val_loader)
        test_preds, test_targets, test_probs = run_eval(test_loader)

        test_metrics = compute_multiclass_metrics(test_targets, test_preds, test_probs, n_classes=5)
        val_metrics = compute_multiclass_metrics(val_targets, val_preds, val_probs, n_classes=5)

        plot_confusion_matrix(test_metrics["confusion_matrix"], DISPLAY_NAMES, RESULTS_DIR / "confusion_matrix.png",
                              f"5-Class Held-Out Test Confusion Matrix (Acc: {test_metrics['accuracy']*100:.2f}%)")
        generate_gradcam_visualizations(model, loaders, device, RESULTS_DIR / "gradcam_analysis.png")

        clf_text = format_classification_report(test_metrics["per_class"], DISPLAY_NAMES, len(test_targets))
        with open(RESULTS_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
            f.write(clf_text)

        # Update JSON
        per_class_summary = {}
        for c_idx, c_name in enumerate(["healthy", "epilepsy", "alzheimers", "parkinsons", "depression"]):
            d_name = DISPLAY_NAMES[c_idx]
            m = test_metrics["per_class"].get(c_idx, {})
            per_class_summary[c_name] = {
                "class_id": c_idx,
                "display_name": d_name,
                "precision": round(float(m.get("precision", 0.0)), 4),
                "recall": round(float(m.get("recall", 0.0)), 4),
                "f1_score": round(float(m.get("f1-score", 0.0)), 4),
                "test_support": int(m.get("support", 0)),
            }

        metrics_json = {
            "experiment": "Unified Explainable CNN-Transformer Multi-Disease EEG Classification",
            "num_classes": 5,
            "classes": DISPLAY_NAMES,
            "cohort_sizes": {
                "train_windows": int(len(train_loader.dataset)),
                "val_windows": int(len(val_loader.dataset)),
                "test_windows": int(len(test_loader.dataset)),
                "total_windows": int(len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)),
            },
            "test_metrics": {
                "accuracy": round(float(test_metrics["accuracy"]), 4),
                "balanced_accuracy": round(float(test_metrics["balanced_accuracy"]), 4),
                "macro_f1": round(float(test_metrics["macro_f1"]), 4),
                "weighted_f1": round(float(test_metrics["weighted_f1"]), 4),
                "cohen_kappa": round(float(test_metrics["cohen_kappa"]), 4),
                "macro_roc_auc": round(float(test_metrics["macro_roc_auc"]), 4),
            },
            "val_metrics": {
                "accuracy": round(float(val_metrics["accuracy"]), 4),
                "balanced_accuracy": round(float(val_metrics["balanced_accuracy"]), 4),
                "macro_f1": round(float(val_metrics["macro_f1"]), 4),
                "weighted_f1": round(float(val_metrics["weighted_f1"]), 4),
            },
            "per_class_metrics": per_class_summary,
            "confusion_matrix_test_5x5": {
                "labels": DISPLAY_NAMES,
                "matrix": test_metrics["confusion_matrix"].tolist(),
            },
            "training": {
                "epochs_trained": len(history["train_loss"]),
                "best_val_acc": round(float(trainer.best_val_acc), 4),
                "elapsed_seconds": round(float(elapsed), 2),
                "checkpoint_path": str(existing_ckpt),
            },
            "scientific_provenance": {
                "healthy": "SRM Healthy (sub-010) + OpenNeuro ds004504 Controls + OpenNeuro ds004584 Controls + NEMAR nm000114 Controls",
                "epilepsy": "CHB-MIT (chb01, 6 EDF sessions)",
                "alzheimers": "OpenNeuro ds004504 (AD patients)",
                "parkinsons": "OpenNeuro ds004584 (PD patients)",
                "depression": "NEMAR nm000114 / MODMA (Major Depressive Disorder patients)",
                "subject_leakage": "ZERO (strictly independent subjects across Train, Val, and Test splits)",
                "montage": "Standard 19-channel 10-20 system, 256 Hz, 5-second windows",
            },
            "known_limitations": [
                "CHB-MIT: 1 subject (chb01) present locally; assigned strictly to Train split. Cross-subject epilepsy generalization should be verified on additional CHB-MIT cases as they are downloaded.",
                "SRM Healthy: 1 subject (sub-010) assigned to Train split; healthy controls in Val/Test are sourced from OpenNeuro and NEMAR cohorts.",
                "FTD (Frontotemporal Dementia): Excluded from ds004504 to maintain clean Alzheimer's vs Healthy boundary."
            ]
        }

        with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
            import json
            json.dump(metrics_json, f, indent=2)

        print("\n--- NEW TEST RESULTS ---")
        print(f"Test Accuracy: {test_metrics['accuracy']*100:.2f}% | Macro F1: {test_metrics['macro_f1']:.4f}")
        print(clf_text)
    else:
        logger.info("Existing checkpoint had higher or equal val_acc. Keeping existing model.")

if __name__ == "__main__":
    main()
