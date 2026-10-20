"""
scripts/run_experiments.py - Controlled Loss & Optimization Experiments.

Sequential, reproducible comparison of loss function formulations:
  - Baseline (Run 0): Weighted CrossEntropy (sqrt-smoothed weights) [reference recorded]
  - Run 1 (unweighted_ce): Unweighted standard CrossEntropy
  - Run 2 (label_smoothing): Label-Smoothed CrossEntropy (smoothing=0.1, weighted)
  - Run 3 (focal_loss): Multi-Class Focal Loss (gamma=1.5, weighted)

Evaluation Protocol:
  - Candidate selection is based strictly on Validation Macro F1 and Validation Balanced Accuracy.
  - Composite Validation Score = 0.5 * (val_macro_f1 + val_bal_acc).
  - The held-out test partition is NEVER evaluated during candidate selection.
  - Results saved under results/experiments/<run_id>/, preserving checkpoints/best_model.pt.

Usage:
  python scripts/run_experiments.py --pilot          # Quick smoke-test (1-2 epochs)
  python scripts/run_experiments.py --run run1       # Run specific experiment
  python scripts/run_experiments.py --all            # Run all candidate experiments sequentially
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.eeg_classifier import EEGClassifier
from training.dataset import get_dataloaders
from training.trainer import EEGTrainer, TrainConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("run_experiments")

EXPERIMENTS_DIR = PROJECT_ROOT / "results" / "experiments"
INDEX_CSV = PROJECT_ROOT / "datasets" / "metadata" / "model_dataset_index.csv"

# Canonical weights from dataset distribution
SQRT_WEIGHTS = [0.5594, 1.7222, 1.1589, 0.6835, 0.8760]

EXPERIMENT_REGISTRY = {
    "run1_unweighted_ce": {
        "description": "Standard Unweighted CrossEntropy Loss",
        "loss_type": "ce",
        "class_weights": None,
        "label_smoothing": 0.0,
        "focal_gamma": 2.0,
    },
    "run2_label_smoothing": {
        "description": "Label-Smoothed CrossEntropy Loss (smoothing=0.1, weighted)",
        "loss_type": "label_smoothing",
        "class_weights": SQRT_WEIGHTS,
        "label_smoothing": 0.1,
        "focal_gamma": 2.0,
    },
    "run3_focal_loss": {
        "description": "Multi-Class Focal Loss (gamma=1.5, weighted)",
        "loss_type": "focal",
        "class_weights": SQRT_WEIGHTS,
        "label_smoothing": 0.0,
        "focal_gamma": 1.5,
    },
}


def run_experiment(
    run_id: str,
    spec: Dict[str, Any],
    max_epochs: int = 10,
    patience: int = 5,
    batch_size: int = 64,
    lr: float = 5e-4,
    weight_decay: float = 1e-2,
    seed: int = 42,
    pilot: bool = False,
) -> Dict[str, Any]:
    """Execute a single controlled experiment on validation data."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    run_dir = EXPERIMENTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("STARTING EXPERIMENT: %s", run_id)
    logger.info("Description: %s", spec.get("description", ""))
    logger.info("Output directory: %s", run_dir)
    logger.info("=" * 80)

    # 1. Load DataLoaders (Train and Val ONLY)
    loaders = get_dataloaders(
        index_csv=str(INDEX_CSV),
        project_root=str(PROJECT_ROOT),
        batch_size=batch_size,
        num_workers=0,
        shuffle_train=True,
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    epochs = 1 if pilot else max_epochs
    eff_patience = 2 if pilot else patience

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
        lr=lr,
        weight_decay=weight_decay,
        warmup_epochs=1 if pilot else 2,
        max_epochs=epochs,
        batch_size=batch_size,
        patience=eff_patience,
        eval_metric="val_composite",  # select checkpoint by 0.5 * (val_macro_f1 + val_bal_acc)
        loss_type=spec.get("loss_type", "ce"),
        label_smoothing=spec.get("label_smoothing", 0.0),
        focal_gamma=spec.get("focal_gamma", 2.0),
        checkpoint_dir=str(run_dir),
        class_weights=spec.get("class_weights"),
        num_workers=0,
    )

    # Save run config
    cfg_path = run_dir / "config.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    trainer = EEGTrainer(cfg)
    t0 = time.time()
    history = trainer.train(train_loader, val_loader)
    elapsed = time.time() - t0

    # Best validation metrics achieved
    best_idx = int(np.argmax(history["val_score"])) if len(history["val_score"]) > 0 else -1
    best_epoch = best_idx + 1 if best_idx >= 0 else 0
    best_val_score = float(history["val_score"][best_idx]) if best_idx >= 0 else 0.0
    best_val_acc = float(history["val_acc"][best_idx]) if best_idx >= 0 else 0.0
    best_val_bal_acc = float(history["val_bal_acc"][best_idx]) if best_idx >= 0 else 0.0
    best_val_macro_f1 = float(history["val_macro_f1"][best_idx]) if best_idx >= 0 else 0.0

    val_summary = {
        "run_id": run_id,
        "description": spec.get("description", ""),
        "loss_type": cfg.loss_type,
        "class_weights": cfg.class_weights,
        "label_smoothing": cfg.label_smoothing,
        "focal_gamma": cfg.focal_gamma,
        "epochs_trained": len(history["train_loss"]),
        "best_epoch": best_epoch,
        "elapsed_seconds": round(elapsed, 2),
        "best_val_score": round(best_val_score, 4),
        "best_val_acc": round(best_val_acc, 4),
        "best_val_bal_acc": round(best_val_bal_acc, 4),
        "best_val_macro_f1": round(best_val_macro_f1, 4),
        "checkpoint_path": str(trainer.best_ckpt_path) if trainer.best_ckpt_path else "",
    }

    # Save validation metrics and history
    with open(run_dir / "val_metrics.json", "w", encoding="utf-8") as f:
        json.dump(val_summary, f, indent=2)
    with open(run_dir / "val_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    logger.info("=" * 80)
    logger.info("EXPERIMENT COMPLETE: %s", run_id)
    logger.info("Best Epoch: %d | Val Score: %.4f | Val Acc: %.4f | Val Bal Acc: %.4f | Val Macro F1: %.4f",
                best_epoch, best_val_score, best_val_acc, best_val_bal_acc, best_val_macro_f1)
    logger.info("=" * 80)

    return val_summary


def update_manifest(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Save or update the central experiment manifest."""
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_csv = EXPERIMENTS_DIR / "manifest.csv"
    manifest_json = EXPERIMENTS_DIR / "manifest.json"

    # Reference Baseline Entry
    baseline_entry = {
        "run_id": "baseline_run0",
        "description": "Baseline Weighted CrossEntropy (sqrt-smoothed weights)",
        "loss_type": "ce",
        "class_weights": SQRT_WEIGHTS,
        "label_smoothing": 0.0,
        "focal_gamma": 2.0,
        "epochs_trained": 10,
        "best_epoch": 4,
        "elapsed_seconds": 734.96,
        "best_val_score": 0.8157,  # 0.5 * (0.7996 + 0.8317)
        "best_val_acc": 0.7662,
        "best_val_bal_acc": 0.8317,
        "best_val_macro_f1": 0.7996,
        "checkpoint_path": "checkpoints/best_model.pt",
    }

    existing: List[Dict[str, Any]] = [baseline_entry]
    if manifest_json.exists():
        try:
            with open(manifest_json, "r", encoding="utf-8") as f:
                saved = json.load(f)
                for item in saved:
                    if item.get("run_id") != "baseline_run0":
                        existing.append(item)
        except Exception as e:
            logger.warning("Could not read existing manifest: %s", e)

    # Merge by run_id
    run_map = {r["run_id"]: r for r in existing}
    for r in records:
        run_map[r["run_id"]] = r

    final_list = list(run_map.values())
    df = pd.DataFrame(final_list)
    df.to_csv(manifest_csv, index=False)
    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump(final_list, f, indent=2)

    logger.info("Updated manifest at %s (%d records)", manifest_csv, len(df))
    return df


def main():
    parser = argparse.ArgumentParser(description="Run controlled EEG optimization experiments.")
    parser.add_argument("--pilot", action="store_true", help="Run 1-epoch pilot test")
    parser.add_argument("--run", type=str, default="", help="Specific run ID to execute")
    parser.add_argument("--all", action="store_true", help="Execute all registry experiments")
    parser.add_argument("--epochs", type=int, default=10, help="Max training epochs")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    args = parser.parse_args()

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.pilot:
        logger.info("RUNNING PILOT TEST...")
        spec = EXPERIMENT_REGISTRY["run1_unweighted_ce"]
        res = run_experiment("pilot_test", spec, max_epochs=1, patience=1, pilot=True)
        print("\nPILOT SUCCESSFUL:", res)
        return

    runs_to_execute = []
    if args.run:
        if args.run not in EXPERIMENT_REGISTRY:
            print(f"Unknown run ID: {args.run}. Available: {list(EXPERIMENT_REGISTRY.keys())}")
            sys.exit(1)
        runs_to_execute.append((args.run, EXPERIMENT_REGISTRY[args.run]))
    elif args.all:
        runs_to_execute = list(EXPERIMENT_REGISTRY.items())
    else:
        print("Please specify --pilot, --run <id>, or --all. Registry options:")
        for k, v in EXPERIMENT_REGISTRY.items():
            print(f"  {k}: {v['description']}")
        return

    results = []
    for r_id, r_spec in runs_to_execute:
        res = run_experiment(r_id, r_spec, max_epochs=args.epochs, patience=args.patience)
        results.append(res)

    df = update_manifest(results)
    print("\n" + "=" * 90)
    print("EXPERIMENT COMPARISON MANIFEST (VALIDATION CRITERION: MACRO F1 & BALANCED ACC)")
    print("=" * 90)
    display_cols = ["run_id", "loss_type", "best_epoch", "best_val_score", "best_val_macro_f1", "best_val_bal_acc", "best_val_acc", "elapsed_seconds"]
    print(df[[c for c in display_cols if c in df.columns]].to_string(index=False))
    print("=" * 90)


if __name__ == "__main__":
    main()
