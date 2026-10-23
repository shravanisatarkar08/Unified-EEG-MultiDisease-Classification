"""
scripts/run_eegnet_baseline.py - Controlled Training and Evaluation of Compact EEGNet Baseline.

Context & Motivation:
---------------------
The canonical baseline (CNN-Transformer, 81,994 parameters) achieved 76.62% validation accuracy
and 82.49% test accuracy, but diagnostic probes revealed that intermediate representations
encode recording site (dataset_name) with 98.60% accuracy. The model's errors are heavily
concentrated in misclassifying healthy controls as disease cases from the same recording site
(e.g., Alzheimer's healthy controls misclassified as Alzheimer's 95% of the time).

This experiment tests the hypothesis that a standard compact EEGNet architecture
(Lawhern et al., 2018) with ~4,613 parameters (~17.8x fewer parameters) regularizes against
high-capacity site-memorization and evaluates genuine generalization.

Safeguards:
-----------
1. Strictly evaluates on the existing VALIDATION split only. The held-out test split is never touched.
2. Identical subject-level split, identical labels, identical class ordering, identical preprocessing
   (interpolate_missing=False, strictly matching baseline).
3. All checkpoints and metric artifacts saved to `results/experiments/eegnet_baseline/`.
4. Canonical baseline artifacts (checkpoints/best_model.pt, results/metrics.json) are NEVER overwritten.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.eegnet import EEGNet
from training.dataset import get_dataloaders, IDX_TO_CLASS, CLASS_TO_IDX
from training.trainer import get_cosine_schedule_with_warmup, EpochMetrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("eegnet_baseline")

EXPERIMENT_DIR = PROJECT_ROOT / "results" / "experiments" / "eegnet_baseline"
INDEX_CSV = PROJECT_ROOT / "datasets" / "metadata" / "model_dataset_index.csv"

# Canonical sqrt-smoothed class weights matching baseline exactly
CANONICAL_CLASS_WEIGHTS = [0.5594, 1.7222, 1.1589, 0.6835, 0.8760]


@dataclass
class EEGNetTrainConfig:
    """Hyperparameters and metadata for EEGNet baseline run."""
    model_name: str = "Compact-EEGNet"
    n_channels: int = 19
    n_samples: int = 1280
    num_classes: int = 5
    F1: int = 8
    D: int = 2
    F2: int = 16
    kernel_length: int = 64
    dropout_rate: float = 0.25

    # Optimization
    lr: float = 1e-3
    weight_decay: float = 1e-2
    warmup_epochs: int = 2
    max_epochs: int = 12
    batch_size: int = 64
    patience: int = 6
    eval_metric: str = "val_composite"  # 0.5 * (bal_acc + macro_f1)
    seed: int = 42

    # Preprocessing check
    interpolate_missing: bool = False  # Explicitly False: canonical baseline preprocessing


class EEGNetTrainer:
    """Training manager specifically for compact EEGNet."""

    def __init__(self, config: EEGNetTrainConfig, device: torch.device):
        self.cfg = config
        self.device = device

        self.model = EEGNet(
            n_channels=config.n_channels,
            n_samples=config.n_samples,
            num_classes=config.num_classes,
            F1=config.F1,
            D=config.D,
            F2=config.F2,
            kernel_length=config.kernel_length,
            dropout_rate=config.dropout_rate,
        ).to(device)

        self.param_count = sum(p.numel() for p in self.model.parameters())
        logger.info("EEGNet initialized: %d parameters (trainable: %d) on %s",
                    self.param_count,
                    sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                    device)

        # Loss with canonical class weights
        weights_tensor = torch.tensor(CANONICAL_CLASS_WEIGHTS, dtype=torch.float32).to(device)
        self.criterion = nn.CrossEntropyLoss(weight=weights_tensor)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        self.best_val_score = 0.0
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.best_ckpt_path: Optional[Path] = None

        self.history: Dict[str, List[float]] = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
            "val_bal_acc": [], "val_macro_f1": [], "val_score": [],
        }

    def _train_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        m = EpochMetrics()
        for eeg, labels, _ in loader:
            eeg = eeg.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(eeg)
            loss = self.criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            preds = logits.argmax(dim=-1)
            m.update(loss.item(), preds, labels)

        n = max(1, len(loader))
        return m.avg_loss(n), m.accuracy

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader) -> Tuple[float, float, float, float]:
        self.model.eval()
        m = EpochMetrics()
        all_preds: List[int] = []
        all_labels: List[int] = []

        for eeg, labels, _ in loader:
            eeg = eeg.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(eeg)
            loss = self.criterion(logits, labels)
            preds = logits.argmax(dim=-1)
            m.update(loss.item(), preds, labels)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        n = max(1, len(loader))
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        n_classes = self.cfg.num_classes

        recalls: List[float] = []
        f1s: List[float] = []
        for c in range(n_classes):
            tp = int(((y_true == c) & (y_pred == c)).sum())
            fn = int(((y_true == c) & (y_pred != c)).sum())
            fp = int(((y_true != c) & (y_pred == c)).sum())
            sup = tp + fn
            rec = float(tp) / max(1, sup) if sup > 0 else 0.0
            prec = float(tp) / max(1, tp + fp) if (tp + fp) > 0 else 0.0
            f1 = float(2 * prec * rec / max(1e-9, prec + rec)) if (prec + rec) > 0 else 0.0
            if sup > 0:
                recalls.append(rec)
            f1s.append(f1)

        bal_acc = float(np.mean(recalls)) if recalls else 0.0
        macro_f1 = float(np.mean(f1s)) if f1s else 0.0
        return m.avg_loss(n), m.accuracy, bal_acc, macro_f1

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List[float]]:
        steps_per_epoch = len(train_loader)
        total_steps = self.cfg.max_epochs * steps_per_epoch
        warmup_steps = self.cfg.warmup_epochs * steps_per_epoch

        scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps
        )

        no_improve = 0

        for epoch in range(1, self.cfg.max_epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self._train_epoch(train_loader)
            val_loss, val_acc, val_bal_acc, val_macro_f1 = self._val_epoch(val_loader)
            val_score = 0.5 * (val_macro_f1 + val_bal_acc)

            for _ in range(steps_per_epoch):
                scheduler.step()

            elapsed = time.time() - t0

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_bal_acc"].append(val_bal_acc)
            self.history["val_macro_f1"].append(val_macro_f1)
            self.history["val_score"].append(val_score)

            logger.info(
                "Epoch %2d/%d | train_loss=%.4f acc=%.3f | val_loss=%.4f acc=%.3f bal_acc=%.3f macro_f1=%.3f score=%.3f | %.1fs",
                epoch, self.cfg.max_epochs,
                train_loss, train_acc, val_loss, val_acc, val_bal_acc, val_macro_f1, val_score, elapsed,
            )

            is_better = val_score > self.best_val_score
            if is_better:
                self.best_val_score = val_score
                self.best_val_acc = val_acc
                self.best_epoch = epoch

                # Save checkpoint
                if self.best_ckpt_path and self.best_ckpt_path.exists():
                    self.best_ckpt_path.unlink()
                self.best_ckpt_path = EXPERIMENT_DIR / f"best_model_ep{epoch:03d}_acc{val_acc:.4f}.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_bal_acc": val_bal_acc,
                    "val_macro_f1": val_macro_f1,
                    "val_composite_score": val_score,
                    "config": asdict(self.cfg),
                }, self.best_ckpt_path)
                logger.info("  -> Saved new best checkpoint: %s", self.best_ckpt_path.name)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.cfg.patience:
                    logger.info("Early stopping triggered at epoch %d (patience=%d)", epoch, self.cfg.patience)
                    break

        return self.history


def run_eegnet_experiment():
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = EEGNetTrainConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    logger.info("=" * 80)
    logger.info("STARTING CONTROLLED COMPACT EEGNET BASELINE EXPERIMENT")
    logger.info("Architecture: Lawhern et al. (2018) EEGNet [F1=%d, D=%d, F2=%d]", cfg.F1, cfg.D, cfg.F2)
    logger.info("Output Directory: %s", EXPERIMENT_DIR)
    logger.info("=" * 80)

    # 1. Load DataLoaders (TRAIN and VAL only; test is NEVER accessed)
    logger.info("Loading DataLoaders with canonical baseline preprocessing (interpolate_missing=False)...")
    loaders = get_dataloaders(
        index_csv=str(INDEX_CSV),
        project_root=str(PROJECT_ROOT),
        batch_size=cfg.batch_size,
        num_workers=0,
        shuffle_train=True,
        interpolate_missing=cfg.interpolate_missing,
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    logger.info("Train windows: %d | Val windows: %d (Test split NOT loaded)",
                len(train_loader.dataset), len(val_loader.dataset))

    # Save run configuration
    with open(EXPERIMENT_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    # 2. Train EEGNet
    trainer = EEGNetTrainer(cfg, device)
    t_start = time.time()
    history = trainer.train(train_loader, val_loader)
    total_elapsed = time.time() - t_start

    logger.info("Training finished in %.1f seconds (%.2f minutes)", total_elapsed, total_elapsed / 60)

    # 3. Comprehensive Validation Diagnostics on Selected Best Checkpoint
    logger.info("Running detailed validation evaluation using selected best checkpoint (epoch %d)...",
                trainer.best_epoch)

    best_ckpt = torch.load(trainer.best_ckpt_path, map_location=device, weights_only=False)
    eval_model = EEGNet(
        n_channels=cfg.n_channels,
        n_samples=cfg.n_samples,
        num_classes=cfg.num_classes,
        F1=cfg.F1,
        D=cfg.D,
        F2=cfg.F2,
        kernel_length=cfg.kernel_length,
        dropout_rate=cfg.dropout_rate,
    ).to(device)
    eval_model.load_state_dict(best_ckpt["model_state_dict"])
    eval_model.eval()

    # Re-evaluate val loader to collect per-sample predictions and provenance
    val_records = []
    val_preds_list = []
    val_labels_list = []

    # Non-shuffled val loader
    val_eval_loader = DataLoader(
        val_loader.dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=val_loader.collate_fn,
    )

    with torch.no_grad():
        for eeg, labels, metas in val_eval_loader:
            eeg = eeg.to(device)
            logits = eval_model(eeg)
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1).cpu().numpy()
            labels_np = labels.numpy()

            val_preds_list.extend(preds.tolist())
            val_labels_list.extend(labels_np.tolist())

            for i in range(len(labels_np)):
                val_records.append({
                    "dataset_name": metas[i]["dataset_name"],
                    "subject_id": metas[i]["subject_id"],
                    "class_category": metas[i]["class_category"],
                    "true_label_idx": int(labels_np[i]),
                    "true_class": IDX_TO_CLASS[int(labels_np[i])],
                    "pred_label_idx": int(preds[i]),
                    "pred_class": IDX_TO_CLASS[int(preds[i])],
                    "correct": int(labels_np[i] == preds[i]),
                })

    df_preds = pd.DataFrame(val_records)
    df_preds.to_csv(EXPERIMENT_DIR / "val_predictions.csv", index=False)

    # Compute full confusion matrix & per-class metrics
    y_true = np.array(val_labels_list)
    y_pred = np.array(val_preds_list)
    n_classes = cfg.num_classes
    class_names = [IDX_TO_CLASS[i] for i in range(n_classes)]

    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    per_class = {}
    recalls = []
    f1s = []
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
    val_composite = 0.5 * (bal_acc + macro_f1)

    # Healthy-control recall by source dataset
    healthy_sub = df_preds[df_preds["true_class"] == "healthy"]
    healthy_by_ds = {}
    for ds in sorted(healthy_sub["dataset_name"].unique()):
        sub_ds = healthy_sub[healthy_sub["dataset_name"] == ds]
        corr = int(sub_ds["correct"].sum())
        n_ds = len(sub_ds)
        healthy_by_ds[ds] = {
            "support": n_ds,
            "correct": corr,
            "recall": round(float(corr / n_ds), 4),
        }

    # Comparison against canonical baseline on the EXACT SAME validation split
    baseline_val = {
        "val_accuracy": 0.7662,
        "val_balanced_accuracy": 0.8317,
        "val_macro_f1": 0.7996,
        "val_weighted_f1": 0.7678,
        "val_composite_score": 0.8157,
        "healthy_recall_by_dataset": {
            "alzheimer": 0.0583,
            "modma": 1.0000,
            "parkinson": 0.4815,
        },
    }

    results = {
        "experiment": "eegnet_baseline",
        "model_architecture": "Compact EEGNet (Lawhern et al., 2018)",
        "parameter_count": trainer.param_count,
        "baseline_parameter_count": 81994,
        "parameter_reduction_ratio": round(81994 / trainer.param_count, 1),
        "selected_epoch": trainer.best_epoch,
        "total_epochs_trained": len(history["train_loss"]),
        "runtime_seconds": round(total_elapsed, 2),
        "runtime_minutes": round(total_elapsed / 60, 2),
        "seed": cfg.seed,
        "evaluation_split": "validation (n=1497, subject-isolated)",
        "validation_metrics": {
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "composite_score": round(val_composite, 4),
        },
        "per_class_metrics": per_class,
        "healthy_recall_by_dataset": healthy_by_ds,
        "confusion_matrix": cm.tolist(),
        "training_history": {
            "train_loss": [round(v, 4) for v in history["train_loss"]],
            "train_acc": [round(v, 4) for v in history["train_acc"]],
            "val_loss": [round(v, 4) for v in history["val_loss"]],
            "val_acc": [round(v, 4) for v in history["val_acc"]],
            "val_bal_acc": [round(v, 4) for v in history["val_bal_acc"]],
            "val_macro_f1": [round(v, 4) for v in history["val_macro_f1"]],
            "val_score": [round(v, 4) for v in history["val_score"]],
        },
        "canonical_baseline_comparison_val_split": {
            "baseline_val_accuracy": baseline_val["val_accuracy"],
            "eegnet_val_accuracy": round(acc, 4),
            "delta_accuracy": round(acc - baseline_val["val_accuracy"], 4),
            "baseline_val_balanced_acc": baseline_val["val_balanced_accuracy"],
            "eegnet_val_balanced_acc": round(bal_acc, 4),
            "delta_balanced_accuracy": round(bal_acc - baseline_val["val_balanced_accuracy"], 4),
            "baseline_val_macro_f1": baseline_val["val_macro_f1"],
            "eegnet_val_macro_f1": round(macro_f1, 4),
            "delta_macro_f1": round(macro_f1 - baseline_val["val_macro_f1"], 4),
            "baseline_val_composite": baseline_val["val_composite_score"],
            "eegnet_val_composite": round(val_composite, 4),
            "delta_composite": round(val_composite - baseline_val["val_composite_score"], 4),
        },
    }

    out_json = EXPERIMENT_DIR / "val_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Results successfully saved to %s", out_json)

    # Print summary
    print("\n" + "=" * 75)
    print("COMPACT EEGNET BASELINE VALIDATION REPORT")
    print("=" * 75)
    print(f"  Model Parameters:       {trainer.param_count:,} (vs. Baseline: 81,994 -> {81994/trainer.param_count:.1f}x smaller)")
    print(f"  Best Checkpoint Epoch:  Epoch {trainer.best_epoch} / {len(history['train_loss'])} trained")
    print(f"  Runtime:                {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print("-" * 75)
    print(f"  Validation Accuracy:    {acc*100:.2f}%  (Baseline: {baseline_val['val_accuracy']*100:.2f}% | Delta: {(acc-baseline_val['val_accuracy'])*100:+.2f}%)")
    print(f"  Balanced Accuracy:      {bal_acc*100:.2f}%  (Baseline: {baseline_val['val_balanced_accuracy']*100:.2f}% | Delta: {(bal_acc-baseline_val['val_balanced_accuracy'])*100:+.2f}%)")
    print(f"  Macro-F1:               {macro_f1:.4f}  (Baseline: {baseline_val['val_macro_f1']:.4f} | Delta: {(macro_f1-baseline_val['val_macro_f1']):+.4f})")
    print(f"  Weighted-F1:            {weighted_f1:.4f}  (Baseline: {baseline_val['val_weighted_f1']:.4f})")
    print(f"  Composite Score:        {val_composite:.4f}  (Baseline: {baseline_val['val_composite_score']:.4f} | Delta: {(val_composite-baseline_val['val_composite_score']):+.4f})")
    print("-" * 75)
    print("  Per-Class Metrics:")
    for c_name in class_names:
        m = per_class[c_name]
        print(f"    {c_name:<12}: Precision={m['precision']*100:5.1f}% | Recall={m['recall']*100:5.1f}% | F1={m['f1_score']:.4f} | n={m['support']}")
    print("-" * 75)
    print("  Healthy-Control Recall by Recording Site (Source Dataset):")
    for ds, info in healthy_by_ds.items():
        base_rec = baseline_val["healthy_recall_by_dataset"].get(ds, 0.0)
        print(f"    {ds:<12}: Recall={info['recall']*100:5.1f}% ({info['correct']}/{info['support']})  [Baseline: {base_rec*100:5.1f}% | Delta: {(info['recall']-base_rec)*100:+.1f}%]")
    print("=" * 75)
    print(f"  Saved artifacts in: {EXPERIMENT_DIR}")
    print("  CANONICAL BASELINE ARTIFACTS AND TEST SET REMAIN COMPLETELY UNCHANGED.")
    print("=" * 75)

    return results


if __name__ == "__main__":
    run_eegnet_experiment()
