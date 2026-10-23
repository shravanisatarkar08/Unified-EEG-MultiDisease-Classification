"""
scripts/run_exp1_full.py — Full 8-Epoch Experiment 1
(Pz weighted-neighbor interpolation for Parkinson's windows)

SAFEGUARDS:
  - Preserves checkpoints/best_model.pt and results/metrics.json unchanged.
  - Never evaluates the held-out test partition.
  - All outputs to: results/experiments/exp1_full_pz_interpolation/
  - Applies identical deterministic Pz interpolation to train AND val loaders.
  - Reports per-class metrics and healthy-control recall by source dataset.
  - Fixed seed=42 for reproducibility.
"""

import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(PROJECT_ROOT)

from models.eeg_classifier import EEGClassifier
from training.dataset import get_dataloaders, IDX_TO_CLASS, CLASS_TO_IDX
from training.trainer import EEGTrainer, TrainConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("exp1_full")

EXPERIMENT_DIR = PROJECT_ROOT / "results" / "experiments" / "exp1_full_pz_interpolation"
INDEX_CSV = PROJECT_ROOT / "datasets" / "metadata" / "model_dataset_index.csv"

# Canonical sqrt-smoothed class weights matching baseline
SQRT_WEIGHTS = [0.5594, 1.7222, 1.1589, 0.6835, 0.8760]


def run_full():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("EXPERIMENT 1 — FULL 8-EPOCH RUN (Pz Weighted-Neighbor Interpolation)")
    logger.info("Output directory: %s", EXPERIMENT_DIR)
    logger.info("=" * 80)

    # -------------------------------------------------------------------
    # 1. DataLoaders with interpolate_missing=True
    # -------------------------------------------------------------------
    logger.info("Loading DataLoaders with interpolate_missing=True...")
    loaders = get_dataloaders(
        index_csv=str(INDEX_CSV),
        project_root=str(PROJECT_ROOT),
        batch_size=64,
        num_workers=0,
        shuffle_train=True,
        interpolate_missing=True,
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    logger.info("Train windows: %d | Val windows: %d", n_train, n_val)

    # -------------------------------------------------------------------
    # 2. Configuration
    # -------------------------------------------------------------------
    cfg = TrainConfig(
        n_channels=19,
        n_samples=1280,
        num_classes=5,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        ff_dim=128,
        cnn_dropout=0.25,
        trans_dropout=0.1,
        lr=5e-4,
        weight_decay=1e-2,
        warmup_epochs=2,
        max_epochs=8,
        batch_size=64,
        patience=5,
        eval_metric="val_composite",
        loss_type="ce",
        label_smoothing=0.0,
        checkpoint_dir=str(EXPERIMENT_DIR),
        class_weights=SQRT_WEIGHTS,
        num_workers=0,
    )

    cfg_dict = asdict(cfg)
    cfg_dict["seed"] = seed
    cfg_dict["interpolation_method"] = "distance-weighted 5-electrode nearest-neighbor average"
    cfg_dict["interpolation_formula"] = "Pz_hat = 0.25*Cz + 0.25*P3 + 0.25*P4 + 0.125*O1 + 0.125*O2"
    cfg_dict["interpolation_note"] = (
        "NOT spherical-spline interpolation. This is a hand-crafted weighted average "
        "of the 5 nearest 10-20 electrodes surrounding Pz, followed by z-score normalization."
    )
    cfg_dict["n_train_windows"] = n_train
    cfg_dict["n_val_windows"] = n_val

    cfg_path = EXPERIMENT_DIR / "config.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, indent=2)
    logger.info("Configuration saved to %s", cfg_path)

    # -------------------------------------------------------------------
    # 3. Training
    # -------------------------------------------------------------------
    trainer = EEGTrainer(cfg)
    t0 = time.time()
    history = trainer.train(train_loader, val_loader)
    elapsed = time.time() - t0

    logger.info("Training completed in %.1f seconds (%.1f min)", elapsed, elapsed / 60)

    # -------------------------------------------------------------------
    # 4. Detailed Validation Diagnostics
    # -------------------------------------------------------------------
    logger.info("Running detailed validation diagnostics...")
    device = trainer.device
    model = trainer.model

    # Load best checkpoint if it was saved
    if trainer.best_ckpt_path and trainer.best_ckpt_path.exists():
        logger.info("Loading best checkpoint: %s", trainer.best_ckpt_path)
        ckpt = torch.load(str(trainer.best_ckpt_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    model.eval()

    records = []
    with torch.no_grad():
        for eeg, labels, metas in val_loader:
            eeg = eeg.to(device)
            logits = model(eeg)
            probs = F.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1).cpu().numpy()
            labels_np = labels.numpy()

            for i in range(len(labels_np)):
                meta_i = metas[i]
                records.append({
                    "dataset_name": meta_i["dataset_name"],
                    "subject_id": meta_i["subject_id"],
                    "true_label_idx": int(labels_np[i]),
                    "true_class": IDX_TO_CLASS[int(labels_np[i])],
                    "pred_label_idx": int(preds[i]),
                    "pred_class": IDX_TO_CLASS[int(preds[i])],
                    "correct": int(labels_np[i] == preds[i]),
                })

    df_res = pd.DataFrame(records)
    val_acc = float(df_res["correct"].mean())

    # Confusion matrix
    n_classes = 5
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(df_res["true_label_idx"], df_res["pred_label_idx"]):
        cm[t, p] += 1

    class_names = [IDX_TO_CLASS[i] for i in range(5)]

    # Per-class metrics
    per_class_metrics = {}
    recalls_list = []
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        sup = int(cm[c, :].sum())
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class_metrics[class_names[c]] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "support": sup,
        }
        if sup > 0:
            recalls_list.append(rec)

    bal_acc = float(np.mean(recalls_list)) if recalls_list else 0.0
    f1_arr = np.array([per_class_metrics[class_names[c]]["f1_score"] for c in range(n_classes)])
    sup_arr = np.array([per_class_metrics[class_names[c]]["support"] for c in range(n_classes)], dtype=float)
    macro_f1 = float(f1_arr.mean())
    weighted_f1 = float((f1_arr * sup_arr).sum() / max(1.0, sup_arr.sum()))

    # Healthy-control recall by source dataset
    healthy_df = df_res[df_res["true_class"] == "healthy"]
    healthy_by_ds = {}
    for ds in sorted(healthy_df["dataset_name"].unique()):
        sub = healthy_df[healthy_df["dataset_name"] == ds]
        if len(sub) > 0:
            healthy_by_ds[ds] = {
                "support": len(sub),
                "correct": int(sub["correct"].sum()),
                "recall": round(float(sub["correct"].mean()), 4),
            }

    # -------------------------------------------------------------------
    # 5. Save Results
    # -------------------------------------------------------------------
    results = {
        "experiment": "exp1_full_pz_interpolation",
        "description": (
            "Full 8-Epoch Training: Distance-weighted 5-electrode Pz interpolation "
            "for ds004584 Parkinson's windows"
        ),
        "epochs_trained": len(history["train_loss"]),
        "elapsed_seconds": round(elapsed, 2),
        "elapsed_minutes": round(elapsed / 60, 2),
        "seed": seed,
        # Training history (final epoch)
        "final_train_loss": round(float(history["train_loss"][-1]), 4),
        "final_train_acc": round(float(history["train_acc"][-1]), 4),
        # Best-checkpoint validation metrics
        "val_accuracy": round(val_acc, 4),
        "val_balanced_accuracy": round(bal_acc, 4),
        "val_macro_f1": round(macro_f1, 4),
        "val_weighted_f1": round(weighted_f1, 4),
        "val_composite_score": round(0.5 * (bal_acc + macro_f1), 4),
        # Per-class
        "per_class": per_class_metrics,
        # Healthy breakdown
        "healthy_recall_by_dataset": healthy_by_ds,
        # Confusion matrix
        "confusion_matrix": cm.tolist(),
        # Training history
        "training_history": {
            "train_loss": [round(v, 4) for v in history["train_loss"]],
            "train_acc": [round(v, 4) for v in history["train_acc"]],
            "val_loss": [round(v, 4) for v in history["val_loss"]],
            "val_acc": [round(v, 4) for v in history["val_acc"]],
            "val_bal_acc": [round(v, 4) for v in history["val_bal_acc"]],
            "val_macro_f1": [round(v, 4) for v in history["val_macro_f1"]],
            "val_score": [round(v, 4) for v in history["val_score"]],
        },
        # Baseline comparison
        "baseline_val_comparison": {
            "baseline_val_accuracy": 0.7662,
            "baseline_val_balanced_acc": 0.8317,
            "baseline_val_macro_f1": 0.7996,
            "baseline_val_composite_score": 0.8157,
            "baseline_healthy_alzheimer_recall": 0.0583,
            "baseline_healthy_parkinson_recall": 0.4815,
            "baseline_healthy_modma_recall": 1.0000,
        },
    }

    out_json = EXPERIMENT_DIR / "val_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Also save the predictions for later analysis
    df_res.to_csv(EXPERIMENT_DIR / "val_predictions.csv", index=False)

    # -------------------------------------------------------------------
    # 6. Print Summary
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPERIMENT 1 RESULTS (VALIDATION SET ONLY)")
    print("=" * 70)
    print(f"  Epochs trained:         {results['epochs_trained']}")
    print(f"  Runtime:                {results['elapsed_minutes']:.1f} min")
    print(f"  Accuracy:               {results['val_accuracy']:.4f}  (baseline: 0.7662)")
    print(f"  Balanced Accuracy:      {results['val_balanced_accuracy']:.4f}  (baseline: 0.8317)")
    print(f"  Macro F1:               {results['val_macro_f1']:.4f}  (baseline: 0.7996)")
    print(f"  Weighted F1:            {results['val_weighted_f1']:.4f}")
    print(f"  Composite Score:        {results['val_composite_score']:.4f}  (baseline: 0.8157)")
    print("-" * 70)
    print(f"  {'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 70)
    for c in range(n_classes):
        m = per_class_metrics[class_names[c]]
        print(f"  {class_names[c]:<15} {m['precision']:>10.4f} {m['recall']:>10.4f} "
              f"{m['f1_score']:>10.4f} {m['support']:>10d}")
    print("-" * 70)
    print("  Healthy-control recall by source dataset:")
    for ds, info in healthy_by_ds.items():
        print(f"    {ds:<15}: {info['recall']:.4f}  ({info['correct']}/{info['support']})")
    print("=" * 70)
    print(f"  Results saved to: {out_json}")
    print(f"  Predictions saved to: {EXPERIMENT_DIR / 'val_predictions.csv'}")
    print(f"  Config saved to: {cfg_path}")
    print(f"\n  NOTE: checkpoints/best_model.pt and results/metrics.json are UNCHANGED.")
    print(f"  NOTE: Held-out test set was NOT evaluated.")
    print("=" * 70)

    return results


if __name__ == "__main__":
    run_full()
