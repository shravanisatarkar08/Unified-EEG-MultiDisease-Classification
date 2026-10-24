"""
scripts/run_exp4_source_balanced.py - Source-Balanced Training Experiment.

Diagnosis Summary
-----------------
The baseline model (76.62% val acc) misclassifies 266/350 errors as
Healthy → Disease, driven by dataset-source confounding:
  - 116/120 alzheimer-source healthy → predicted alzheimers (96.7% error)
  - 149/270 parkinson-source healthy → predicted parkinsons (55.2% error)
  - modma-source healthy: 98.9% correct (not confounded)

Root Cause: Each dataset contributes healthy controls + one disease.
The model learns recording-site features instead of disease features.
With standard random sampling, batches are dominated by the largest
source (parkinson, 3150/7466 = 42%), so the model sees mostly
parkinson-healthy and parkinson-disease together, encouraging the shortcut.

Intervention: Source-Balanced Sampling
--------------------------------------
Build mini-batches so every batch contains roughly equal representation
from each dataset source. This forces the model to see healthy controls
from ALL sources in every batch, preventing it from relying on
source-specific features to distinguish healthy from disease.

Implementation: WeightedRandomSampler with per-sample weights inversely
proportional to their dataset-source frequency. This ensures each
source contributes ~equally to each epoch.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.eeg_classifier import EEGClassifier
from training.dataset import (
    EEGDataset, get_dataloaders, IDX_TO_CLASS, CLASS_TO_IDX,
    _eeg_collate_fn,
)
from training.trainer import get_cosine_schedule_with_warmup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("exp4_source_balanced")

EXPERIMENT_DIR = PROJECT_ROOT / "results" / "experiments" / "exp4_source_balanced"
INDEX_CSV = PROJECT_ROOT / "datasets" / "metadata" / "model_dataset_index.csv"

# Canonical sqrt-smoothed class weights (same as baseline)
CANONICAL_CLASS_WEIGHTS = [0.5594, 1.7222, 1.1589, 0.6835, 0.8760]


def build_source_balanced_loader(
    index_df: pd.DataFrame,
    project_root: Path,
    batch_size: int = 64,
    num_workers: int = 0,
) -> Tuple[DataLoader, Dict[str, float]]:
    """Build a DataLoader with source-balanced sampling.

    Each sample gets weight = 1 / (number of samples from same dataset_name).
    This makes each dataset source contribute equally to each epoch in expectation.

    Returns the DataLoader and the per-source weight dict for logging.
    """
    dataset = EEGDataset(
        index_df=index_df.reset_index(drop=True),
        project_root=project_root,
        cache_size=2,
        interpolate_missing=False,
    )

    # Compute per-sample weights based on dataset source
    source_counts = index_df["dataset_name"].value_counts().to_dict()
    n_sources = len(source_counts)
    total_samples = len(index_df)

    # Weight = total / (n_sources * source_count)
    # This means each source has equal total weight
    source_weights = {}
    for src, cnt in source_counts.items():
        source_weights[src] = total_samples / (n_sources * cnt)

    logger.info("Source-balanced weights:")
    for src, w in sorted(source_weights.items()):
        logger.info("  %s: count=%d, weight=%.4f", src, source_counts[src], w)

    sample_weights = []
    for _, row in index_df.reset_index(drop=True).iterrows():
        sample_weights.append(source_weights[row["dataset_name"]])

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=total_samples,  # same epoch length
        replacement=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=_eeg_collate_fn,
        drop_last=False,
    )

    return loader, source_weights


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for eeg, labels, _ in loader:
        eeg = eeg.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(eeg)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        preds = logits.argmax(dim=-1)
        total_loss += loss.item()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    n = max(1, len(loader))
    return total_loss / n, correct / max(1, total)


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes=5):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_metas = []

    for eeg, labels, metas in loader:
        eeg = eeg.to(device)
        labels = labels.to(device)
        logits = model(eeg)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_metas.extend(metas)

    n = max(1, len(loader))
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    # Per-class metrics
    recalls = []
    f1s = []
    per_class = {}
    class_names = [IDX_TO_CLASS[i] for i in range(num_classes)]

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    supports = []
    for c, name in enumerate(class_names):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        sup = int(cm[c, :].sum())
        prec = float(tp) / max(1, tp + fp) if (tp + fp) > 0 else 0.0
        rec = float(tp) / max(1, tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / max(1e-9, prec + rec)) if (prec + rec) > 0 else 0.0
        per_class[name] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "support": sup,
        }
        recalls.append(rec)
        f1s.append(f1)
        supports.append(sup)

    acc = float((y_true == y_pred).mean())
    bal_acc = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))
    weighted_f1 = float(np.sum(np.array(f1s) * np.array(supports)) / np.sum(supports))

    # Healthy recall by dataset
    healthy_by_ds = {}
    for i in range(len(y_true)):
        if y_true[i] == 0:  # healthy
            ds = all_metas[i]["dataset_name"]
            if ds not in healthy_by_ds:
                healthy_by_ds[ds] = {"support": 0, "correct": 0}
            healthy_by_ds[ds]["support"] += 1
            if y_pred[i] == 0:
                healthy_by_ds[ds]["correct"] += 1
    for ds in healthy_by_ds:
        s = healthy_by_ds[ds]
        s["recall"] = round(float(s["correct"] / s["support"]), 4)

    # Save predictions
    val_records = []
    for i in range(len(y_true)):
        val_records.append({
            "dataset_name": all_metas[i]["dataset_name"],
            "subject_id": all_metas[i]["subject_id"],
            "class_category": all_metas[i]["class_category"],
            "true_label_idx": int(y_true[i]),
            "true_class": IDX_TO_CLASS[int(y_true[i])],
            "pred_label_idx": int(y_pred[i]),
            "pred_class": IDX_TO_CLASS[int(y_pred[i])],
            "correct": int(y_true[i] == y_pred[i]),
        })

    return {
        "validation_accuracy": round(acc, 4),
        "validation_balanced_accuracy": round(bal_acc, 4),
        "validation_macro_f1": round(macro_f1, 4),
        "validation_weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "healthy_recall_by_dataset": healthy_by_ds,
        "confusion_matrix": cm.tolist(),
        "val_loss": total_loss / n,
        "predictions": val_records,
    }


def run_experiment():
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Running Source-Balanced Training Experiment on device: %s", device)

    # ---- Integrity checks ----
    df_index = pd.read_csv(INDEX_CSV)
    tr_subjs = set(zip(
        df_index[df_index["split"] == "train"]["dataset_name"],
        df_index[df_index["split"] == "train"]["subject_id"],
    ))
    val_subjs = set(zip(
        df_index[df_index["split"] == "val"]["dataset_name"],
        df_index[df_index["split"] == "val"]["subject_id"],
    ))
    assert len(tr_subjs & val_subjs) == 0, f"Subject leakage: {tr_subjs & val_subjs}"
    logger.info("Subject leakage check PASSED (0 overlap between train and val)")

    # ---- Build source-balanced train loader ----
    train_df = df_index[df_index["split"] == "train"].copy()
    train_loader, source_weights = build_source_balanced_loader(
        train_df, PROJECT_ROOT, batch_size=64, num_workers=0,
    )

    # Standard val loader (no source balancing)
    loaders = get_dataloaders(
        index_csv=str(INDEX_CSV),
        project_root=str(PROJECT_ROOT),
        batch_size=64,
        num_workers=0,
        shuffle_train=False,
        interpolate_missing=False,
    )
    val_loader = loaders["val"]

    # ---- Model, optimizer, loss (same as baseline) ----
    torch.manual_seed(42)
    np.random.seed(42)

    model = EEGClassifier(
        n_channels=19,
        n_samples=1280,
        num_classes=5,
        cnn_dropout=0.25,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        ff_dim=128,
        trans_dropout=0.10,
    ).to(device)

    weights = torch.tensor(CANONICAL_CLASS_WEIGHTS, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-4,
        weight_decay=1e-2,
    )

    max_epochs = 8
    patience = 4
    warmup_epochs = 2
    steps_per_epoch = len(train_loader)
    total_steps = max_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ---- Config ----
    config = {
        "experiment": "exp4_source_balanced",
        "intervention": "WeightedRandomSampler with inverse source-frequency weights",
        "lr": 5e-4,
        "weight_decay": 1e-2,
        "cnn_dropout": 0.25,
        "trans_dropout": 0.10,
        "max_epochs": max_epochs,
        "patience": patience,
        "batch_size": 64,
        "seed": 42,
        "class_weights": CANONICAL_CLASS_WEIGHTS,
        "source_weights": source_weights,
    }
    with open(EXPERIMENT_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # ---- Training loop ----
    best_val_acc = 0.0
    best_epoch = 0
    best_ckpt_path = None
    no_improve = 0
    history = {"train_loss": [], "train_acc": [], "val_acc": [], "val_bal_acc": [], "val_macro_f1": []}

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_result = evaluate(model, val_loader, criterion, device)
        v_acc = val_result["validation_accuracy"]
        v_bal = val_result["validation_balanced_accuracy"]
        v_f1 = val_result["validation_macro_f1"]

        for _ in range(steps_per_epoch):
            scheduler.step()

        elapsed = time.time() - t0

        history["train_loss"].append(round(t_loss, 4))
        history["train_acc"].append(round(t_acc, 4))
        history["val_acc"].append(round(v_acc, 4))
        history["val_bal_acc"].append(round(v_bal, 4))
        history["val_macro_f1"].append(round(v_f1, 4))

        logger.info(
            "Ep %d/%d | loss=%.4f acc=%.3f | val_acc=%.3f bal=%.3f f1=%.3f | %.1fs",
            epoch, max_epochs, t_loss, t_acc, v_acc, v_bal, v_f1, elapsed,
        )

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_epoch = epoch
            if best_ckpt_path and Path(best_ckpt_path).exists():
                Path(best_ckpt_path).unlink()
            best_ckpt_path = EXPERIMENT_DIR / f"best_model_exp4_ep{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": v_acc,
                "val_bal_acc": v_bal,
                "val_macro_f1": v_f1,
                "config": config,
            }, best_ckpt_path)
            logger.info("  -> New best checkpoint: %s (val_acc=%.4f)", best_ckpt_path.name, v_acc)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    # ---- Final evaluation with best checkpoint ----
    logger.info("Final evaluation with best checkpoint (epoch %d)", best_epoch)
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    final_result = evaluate(model, val_loader, criterion, device)

    # Save predictions
    df_preds = pd.DataFrame(final_result["predictions"])
    df_preds.to_csv(EXPERIMENT_DIR / "val_predictions.csv", index=False)

    # Save metrics
    metrics = {
        "run_name": "exp4_source_balanced",
        "intervention": "source-balanced sampling (WeightedRandomSampler)",
        "provenance": "freshly_trained",
        "training_completed": True,
        "selected_epoch": best_epoch,
        "checkpoint_path": str(best_ckpt_path.relative_to(PROJECT_ROOT)),
        "validation_accuracy": final_result["validation_accuracy"],
        "validation_balanced_accuracy": final_result["validation_balanced_accuracy"],
        "validation_macro_f1": final_result["validation_macro_f1"],
        "validation_weighted_f1": final_result["validation_weighted_f1"],
        "per_class": final_result["per_class"],
        "healthy_recall_by_dataset": final_result["healthy_recall_by_dataset"],
        "confusion_matrix": final_result["confusion_matrix"],
        "training_history": history,
    }
    with open(EXPERIMENT_DIR / "val_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # ---- Baseline comparison ----
    baseline = {
        "val_accuracy": 0.7662,
        "val_balanced_acc": 0.8317,
        "val_macro_f1": 0.7996,
        "healthy_recall": {"alzheimer": 0.0333, "modma": 0.9889, "parkinson": 0.4481},
    }

    candidate_acc = final_result["validation_accuracy"]
    delta = candidate_acc - baseline["val_accuracy"]

    print("\n" + "=" * 80)
    print("EXP4 SOURCE-BALANCED TRAINING RESULTS")
    print("=" * 80)
    print(f"{'Metric':<30} {'Baseline':>12} {'Candidate':>12} {'Delta':>10}")
    print("-" * 80)
    print(f"{'Val Accuracy':<30} {baseline['val_accuracy']*100:>11.2f}% {candidate_acc*100:>11.2f}% {delta*100:>+9.2f}%")
    print(f"{'Balanced Accuracy':<30} {baseline['val_balanced_acc']*100:>11.2f}% {final_result['validation_balanced_accuracy']*100:>11.2f}%")
    print(f"{'Macro-F1':<30} {baseline['val_macro_f1']:>12.4f} {final_result['validation_macro_f1']:>12.4f}")
    print()

    print("Per-class Recall:")
    for cls_name, cls_data in final_result["per_class"].items():
        print(f"  {cls_name:<15} recall={cls_data['recall']:.4f}  f1={cls_data['f1_score']:.4f}  support={cls_data['support']}")
    print()

    print("Healthy Recall by Dataset:")
    for ds, data in sorted(final_result["healthy_recall_by_dataset"].items()):
        base_r = baseline["healthy_recall"].get(ds, 0.0)
        print(f"  {ds:<15} {base_r*100:>7.1f}% -> {data['recall']*100:>7.1f}%  (n={data['support']})")

    print()
    print("Confusion Matrix:")
    cm = np.array(final_result["confusion_matrix"])
    class_names = [IDX_TO_CLASS[i] for i in range(5)]
    print(f"{'':>15}", end="")
    for name in class_names:
        print(f"{name:>12}", end="")
    print()
    for i, name in enumerate(class_names):
        print(f"{name:>15}", end="")
        for j in range(5):
            print(f"{cm[i,j]:>12d}", end="")
        print()

    print("=" * 80)
    if candidate_acc > 0.7662:
        print(f"CANDIDATE ACCEPTED: {candidate_acc*100:.2f}% > 76.62% baseline")
    else:
        print(f"CANDIDATE REJECTED: {candidate_acc*100:.2f}% <= 76.62% baseline")
    print("=" * 80)

    return metrics


if __name__ == "__main__":
    run_experiment()
