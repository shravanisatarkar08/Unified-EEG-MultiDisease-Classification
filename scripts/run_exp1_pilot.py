"""
scripts/run_exp1_pilot.py - 1-Epoch Pilot for Experiment 1 (Pz Topological Interpolation).

SAFEGUARDS:
  - Preserves checkpoints/best_model.pt and results/metrics.json unchanged.
  - Never touches or evaluates the held-out test partition.
  - Runs in separate directory: results/experiments/exp1_pilot_pz_interpolation/
  - Applies identical deterministic Pz topological interpolation to train and val loaders.
  - Detailed breakdown of healthy-control recall across source datasets.
"""

import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.eeg_classifier import EEGClassifier
from training.dataset import get_dataloaders, IDX_TO_CLASS, CLASS_TO_IDX
from training.trainer import EEGTrainer, TrainConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("exp1_pilot")

EXPERIMENT_DIR = PROJECT_ROOT / "results" / "experiments" / "exp1_pilot_pz_interpolation"
INDEX_CSV = PROJECT_ROOT / "datasets" / "metadata" / "model_dataset_index.csv"

# Canonical sqrt-smoothed class weights matching baseline
SQRT_WEIGHTS = [0.5594, 1.7222, 1.1589, 0.6835, 0.8760]


def run_pilot():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("================================================================================")
    logger.info("STARTING EXPERIMENT 1 PILOT (1-EPOCH: PZ TOPOLOGICAL INTERPOLATION)")
    logger.info("Output directory: %s", EXPERIMENT_DIR)
    logger.info("================================================================================")

    # 1. Load DataLoaders with interpolate_missing=True for train and val
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

    logger.info("Train windows: %d | Val windows: %d", len(train_loader.dataset), len(val_loader.dataset))

    # 2. Configure 1-epoch pilot training
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
        warmup_epochs=1,
        max_epochs=1,
        batch_size=64,
        patience=2,
        eval_metric="val_composite",
        loss_type="ce",
        label_smoothing=0.0,
        checkpoint_dir=str(EXPERIMENT_DIR),
        class_weights=SQRT_WEIGHTS,
        num_workers=0,
    )

    # Save run config
    cfg_path = EXPERIMENT_DIR / "config.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    # 3. Execute 1 Epoch of Training
    trainer = EEGTrainer(cfg)
    t0 = time.time()
    history = trainer.train(train_loader, val_loader)
    elapsed = time.time() - t0

    # 4. Detailed Validation Diagnostics with Per-Dataset Breakdown
    logger.info("Running detailed validation breakdown on pilot model...")
    device = trainer.device
    model = trainer.model
    model.eval()

    records = []
    with torch.no_grad():
        for eeg, labels, metas in val_loader:
            eeg = eeg.to(device)
            logits = model(eeg)
            probs = F.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1).cpu().numpy()
            labels_np = labels.numpy()
            probs_np = probs.cpu().numpy()

            batch_size = len(labels_np)
            for i in range(batch_size):
                meta_i = metas[i] if isinstance(metas, list) else {k: v[i] for k, v in metas.items()}
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
    recalls = []
    precisions = []
    f1s = []
    supports = []
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        sup = int(cm[c, :].sum())
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        recalls.append(rec)
        precisions.append(prec)
        f1s.append(f1)
        supports.append(sup)

    bal_acc = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))
    weighted_f1 = float(np.sum(np.array(f1s) * np.array(supports)) / np.sum(supports))

    # Healthy breakdown by source dataset
    healthy_df = df_res[df_res["true_class"] == "healthy"]
    healthy_by_ds = {}
    for ds in ["alzheimer", "parkinson", "modma"]:
        sub = healthy_df[healthy_df["dataset_name"] == ds]
        if len(sub) > 0:
            rec_ds = float(sub["correct"].mean())
            healthy_by_ds[ds] = {
                "support": len(sub),
                "correct": int(sub["correct"].sum()),
                "recall": round(rec_ds, 4),
            }

    pilot_results = {
        "experiment": "exp1_pilot_pz_interpolation",
        "description": "1-Epoch Pilot: Topological 4-Direction Pz Scalp Surface Interpolation",
        "epochs_trained": 1,
        "elapsed_seconds": round(elapsed, 2),
        "train_loss": round(float(history["train_loss"][-1]), 4),
        "val_loss": round(float(history["val_loss"][-1]), 4),
        "val_accuracy": round(val_acc, 4),
        "val_balanced_accuracy": round(bal_acc, 4),
        "val_macro_f1": round(macro_f1, 4),
        "val_weighted_f1": round(weighted_f1, 4),
        "val_score": round(0.5 * (bal_acc + macro_f1), 4),
        "per_class": {
            class_names[c]: {
                "precision": round(precisions[c], 4),
                "recall": round(recalls[c], 4),
                "f1_score": round(f1s[c], 4),
                "support": supports[c],
            }
            for c in range(n_classes)
        },
        "healthy_recall_by_dataset": healthy_by_ds,
        "confusion_matrix": cm.tolist(),
        "baseline_val_comparison": {
            "baseline_val_accuracy": 0.7662,
            "baseline_val_balanced_acc": 0.8317,
            "baseline_val_macro_f1": 0.7996,
            "baseline_val_score": 0.8157,
            "baseline_healthy_alzheimer_recall": 0.0583,
            "baseline_healthy_parkinson_recall": 0.4815,
            "baseline_healthy_modma_recall": 1.0000,
        },
    }

    out_json = EXPERIMENT_DIR / "pilot_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(pilot_results, f, indent=2)

    logger.info("Pilot completed successfully in %.1f seconds.", elapsed)
    logger.info("Results saved to: %s", out_json)
    return pilot_results


if __name__ == "__main__":
    run_pilot()
