"""
scripts/run_hyperparam_tuning.py - Multi-Stage Automated Hyperparameter Optimization.

Workflow:
---------
Stage 1: Learning Rate Search (1e-4, 3e-4, 5e-4) with baseline wd=1e-2, drop=0.25.
         Automatically selects best LR by validation accuracy (tiebreak: macro-F1, balanced accuracy).
Stage 2: Regularization Sweep around best LR:
         Weight decay: 1e-3, 1e-2, 5e-2
         Dropout: 0.15, 0.25, 0.35
         Automatically selects best (LR, WD, Dropout) configuration.
Stage 3: Fine LR Search around best Stage 2 configuration:
         Nearby learning rates: 0.5x, 1.0x (cached), 1.5x best LR.
Consolidation:
         Generates unified summary JSON and CSV comparing all candidates against the 76.62% baseline.

Safeguards:
-----------
1. Preserves checkpoints/best_model.pt and results/metrics.json unchanged.
2. Training split only for optimization; validation split only for model selection.
3. Test split is NEVER loaded or evaluated.
4. Skips already completed runs if valid val_metrics.json exists.
"""

from __future__ import annotations

import argparse
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

from models.eeg_classifier import EEGClassifier
from training.dataset import get_dataloaders, IDX_TO_CLASS, CLASS_TO_IDX
from training.trainer import get_cosine_schedule_with_warmup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("hyperparam_tuning")

EXPERIMENT_DIR = PROJECT_ROOT / "results" / "experiments" / "exp3_hyperparam_tuning"
INDEX_CSV = PROJECT_ROOT / "datasets" / "metadata" / "model_dataset_index.csv"

# Canonical sqrt-smoothed class weights
CANONICAL_CLASS_WEIGHTS = [0.5594, 1.7222, 1.1589, 0.6835, 0.8760]


@dataclass
class TuningConfig:
    run_name: str
    stage: int = 1
    lr: float = 5e-4
    weight_decay: float = 1e-2
    cnn_dropout: float = 0.25
    trans_dropout: float = 0.10
    n_channels: int = 19
    n_samples: int = 1280
    num_classes: int = 5
    embed_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    ff_dim: int = 128
    warmup_epochs: int = 2
    max_epochs: int = 8
    batch_size: int = 64
    patience: int = 4
    eval_metric: str = "val_acc"
    seed: int = 42
    provenance: str = "freshly_trained"


class TuningRunManager:
    """Manages training and evaluation for one hyperparameter configuration."""

    def __init__(self, cfg: TuningConfig, run_dir: Path, device: torch.device):
        self.cfg = cfg
        self.run_dir = run_dir
        self.device = device
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.model = EEGClassifier(
            n_channels=cfg.n_channels,
            n_samples=cfg.n_samples,
            num_classes=cfg.num_classes,
            cnn_dropout=cfg.cnn_dropout,
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            ff_dim=cfg.ff_dim,
            trans_dropout=cfg.trans_dropout,
        ).to(device)

        weights = torch.tensor(CANONICAL_CLASS_WEIGHTS, dtype=torch.float32).to(device)
        self.criterion = nn.CrossEntropyLoss(weight=weights)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.best_ckpt_path: Optional[Path] = None

        self.history: Dict[str, List[float]] = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
            "val_bal_acc": [], "val_macro_f1": [],
        }

    def _train_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

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
            total_loss += loss.item()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        n = max(1, len(loader))
        return total_loss / n, correct / max(1, total)

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader) -> Tuple[float, float, float, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for eeg, labels, _ in loader:
            eeg = eeg.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(eeg)
            loss = self.criterion(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        n = max(1, len(loader))
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        n_classes = self.cfg.num_classes

        recalls = []
        f1s = []
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
        acc = float((y_true == y_pred).mean())

        return total_loss / n, acc, bal_acc, macro_f1

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
            t_loss, t_acc = self._train_epoch(train_loader)
            v_loss, v_acc, v_bal, v_f1 = self._val_epoch(val_loader)

            for _ in range(steps_per_epoch):
                scheduler.step()

            elapsed = time.time() - t0

            self.history["train_loss"].append(t_loss)
            self.history["train_acc"].append(t_acc)
            self.history["val_loss"].append(v_loss)
            self.history["val_acc"].append(v_acc)
            self.history["val_bal_acc"].append(v_bal)
            self.history["val_macro_f1"].append(v_f1)

            logger.info(
                "[%s] Ep %d/%d | loss=%.4f acc=%.3f | val_acc=%.3f (bal=%.3f f1=%.3f) | %.1fs",
                self.cfg.run_name, epoch, self.cfg.max_epochs,
                t_loss, t_acc, v_acc, v_bal, v_f1, elapsed,
            )

            if v_acc > self.best_val_acc:
                self.best_val_acc = v_acc
                self.best_epoch = epoch

                if self.best_ckpt_path and self.best_ckpt_path.exists():
                    self.best_ckpt_path.unlink()
                self.best_ckpt_path = self.run_dir / f"best_model_{self.cfg.run_name}_ep{epoch:03d}.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "val_acc": v_acc,
                    "val_bal_acc": v_bal,
                    "val_macro_f1": v_f1,
                    "config": asdict(self.cfg),
                }, self.best_ckpt_path)
                logger.info("  -> Saved new best checkpoint: %s (val_acc=%.4f)",
                            self.best_ckpt_path.name, v_acc)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.cfg.patience:
                    logger.info("[%s] Early stopping triggered at epoch %d", self.cfg.run_name, epoch)
                    break

        return self.history

    def evaluate_best_checkpoint(self, val_loader: DataLoader) -> dict:
        """Full validation diagnostic on the selected best model checkpoint."""
        logger.info("[%s] Evaluating selected checkpoint: %s", self.cfg.run_name, self.best_ckpt_path)
        ckpt = torch.load(self.best_ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        val_records = []
        with torch.no_grad():
            for eeg, labels, metas in val_loader:
                eeg = eeg.to(self.device)
                logits = self.model(eeg)
                preds = logits.argmax(dim=-1).cpu().numpy()
                labels_np = labels.numpy()

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
        df_preds.to_csv(self.run_dir / "val_predictions.csv", index=False)

        n_classes = self.cfg.num_classes
        class_names = [IDX_TO_CLASS[i] for i in range(n_classes)]
        cm = np.zeros((n_classes, n_classes), dtype=np.int64)
        for t, p in zip(df_preds["true_label_idx"], df_preds["pred_label_idx"]):
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

        acc = float(df_preds["correct"].mean())
        bal_acc = float(np.mean(recalls))
        macro_f1 = float(np.mean(f1s))
        weighted_f1 = float(np.sum(np.array(f1s) * np.array(supports)) / np.sum(supports))

        healthy_df = df_preds[df_preds["true_class"] == "healthy"]
        healthy_by_ds = {}
        for ds in sorted(healthy_df["dataset_name"].unique()):
            sub = healthy_df[healthy_df["dataset_name"] == ds]
            corr = int(sub["correct"].sum())
            n_sub = len(sub)
            healthy_by_ds[ds] = {
                "support": n_sub,
                "correct": corr,
                "recall": round(float(corr / n_sub), 4),
            }

        return {
            "run_name": self.cfg.run_name,
            "stage": self.cfg.stage,
            "lr": self.cfg.lr,
            "weight_decay": self.cfg.weight_decay,
            "cnn_dropout": self.cfg.cnn_dropout,
            "provenance": getattr(self.cfg, "provenance", "freshly_trained"),
            "training_completed": True,
            "checkpoint_path": str(self.best_ckpt_path.relative_to(PROJECT_ROOT)) if self.best_ckpt_path else None,
            "selected_epoch": self.best_epoch,
            "validation_accuracy": round(acc, 4),
            "validation_balanced_accuracy": round(bal_acc, 4),
            "validation_macro_f1": round(macro_f1, 4),
            "validation_weighted_f1": round(weighted_f1, 4),
            "per_class": per_class,
            "healthy_recall_by_dataset": healthy_by_ds,
            "confusion_matrix": cm.tolist(),
        }


def execute_or_load_run(
    cfg: TuningConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> dict:
    """Execute run or load existing metrics if genuine completed run exists."""
    run_dir = EXPERIMENT_DIR / cfg.run_name
    val_metrics_file = run_dir / "val_metrics.json"

    # Reuse ONLY if already completed with valid metrics AND actual checkpoint produced for this run
    if val_metrics_file.exists():
        try:
            with open(val_metrics_file, "r", encoding="utf-8") as f:
                res = json.load(f)
            ckpts = list(run_dir.glob("best_model_*.pt"))
            if ckpts and res.get("training_completed", False) and "validation_accuracy" in res:
                logger.info("[%s] Found genuine completed run (val_acc=%.4f). Skipping re-training.",
                            cfg.run_name, res["validation_accuracy"])
                res["run_name"] = cfg.run_name
                res["stage"] = cfg.stage
                res["lr"] = cfg.lr
                res["weight_decay"] = cfg.weight_decay
                res["cnn_dropout"] = cfg.cnn_dropout
                res["provenance"] = res.get("provenance", cfg.provenance)
                res["training_completed"] = True
                res["checkpoint_path"] = str(ckpts[0].relative_to(PROJECT_ROOT))
                return res
        except Exception as e:
            logger.warning("[%s] Failed reading existing metrics (%s); re-running.", cfg.run_name, e)

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    mgr = TuningRunManager(cfg, run_dir, device)
    t0 = time.time()
    history = mgr.train(train_loader, val_loader)
    elapsed = time.time() - t0

    res = mgr.evaluate_best_checkpoint(val_loader)
    res["runtime_seconds"] = round(elapsed, 2)
    res["training_history"] = {k: [round(x, 4) for x in v] for k, v in history.items()}
    res["provenance"] = cfg.provenance
    res["training_completed"] = True
    res["checkpoint_path"] = str(mgr.best_ckpt_path.relative_to(PROJECT_ROOT)) if mgr.best_ckpt_path else None

    with open(val_metrics_file, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    return res


def pick_best_candidate(results_dict: Dict[str, dict]) -> Tuple[str, dict]:
    """Select candidate with highest val_acc, breaking ties by macro-F1 then balanced accuracy."""
    sorted_candidates = sorted(
        results_dict.items(),
        key=lambda item: (
            item[1]["validation_accuracy"],
            item[1]["validation_macro_f1"],
            item[1]["validation_balanced_accuracy"],
        ),
        reverse=True,
    )
    return sorted_candidates[0]


def run_full_multi_stage_tuning():
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Executing Multi-Stage Hyperparameter Optimization on device: %s", device)

    # Integrity verification
    df_index = pd.read_csv(INDEX_CSV)
    tr_subjs = set(zip(df_index[df_index["split"] == "train"]["dataset_name"], df_index[df_index["split"] == "train"]["subject_id"]))
    val_subjs = set(zip(df_index[df_index["split"] == "val"]["dataset_name"], df_index[df_index["split"] == "val"]["subject_id"]))
    assert len(tr_subjs & val_subjs) == 0, f"Subject leakage detected: {tr_subjs & val_subjs}"
    logger.info("Integrity check PASSED: 0 overlapping subjects between train and val.")

    loaders = get_dataloaders(
        index_csv=str(INDEX_CSV),
        project_root=str(PROJECT_ROOT),
        batch_size=64,
        num_workers=0,
        shuffle_train=True,
        interpolate_missing=False,
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    baseline_val = {
        "val_accuracy": 0.7662,
        "val_balanced_acc": 0.8317,
        "val_macro_f1": 0.7996,
        "healthy_recall_by_dataset": {"alzheimer": 0.0583, "modma": 1.0000, "parkinson": 0.4815},
    }

    all_stage_results: Dict[str, dict] = {}

    # =========================================================================
    # STAGE 1: LEARNING RATE SWEEP (1e-4, 3e-4, 5e-4)
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info(">>> STARTING STAGE 1: LEARNING RATE SEARCH <<<")
    logger.info("=" * 80)

    # Load control run (lr=5e-4, wd=1e-2, drop=0.25) as cached control with genuine provenance
    control_dir = EXPERIMENT_DIR / "control_lr5e-4_wd1e-2_drop0.25"
    control_metrics_file = control_dir / "val_metrics.json"
    if control_metrics_file.exists():
        with open(control_metrics_file, "r", encoding="utf-8") as f:
            control_res = json.load(f)
        control_res["run_name"] = "control_lr5e-4_wd1e-2_drop0.25"
        control_res["provenance"] = "cached_control (canonical baseline reproduction on train split)"
        control_res["training_completed"] = True
        control_res["stage"] = 1
        control_ckpts = list(control_dir.glob("best_model_*.pt"))
        if control_ckpts:
            control_res["checkpoint_path"] = str(control_ckpts[0].relative_to(PROJECT_ROOT))
        all_stage_results["control_lr5e-4_wd1e-2_drop0.25"] = control_res
        logger.info("[control_lr5e-4_wd1e-2_drop0.25] Loaded cached control run (val_acc=%.4f, epoch=%d, provenance=%s)",
                    control_res["validation_accuracy"], control_res["selected_epoch"], control_res["provenance"])
    else:
        control_cfg = TuningConfig(
            run_name="control_lr5e-4_wd1e-2_drop0.25",
            stage=1,
            lr=5e-4,
            weight_decay=1e-2,
            cnn_dropout=0.25,
            provenance="freshly_trained_control",
        )
        control_res = execute_or_load_run(control_cfg, train_loader, val_loader, device)
        all_stage_results["control_lr5e-4_wd1e-2_drop0.25"] = control_res

    # Stage 1 fresh candidate runs: lr=3e-4, lr=1e-4
    stage1_cfgs = [
        TuningConfig(run_name="stage1_lr3e-4_wd1e-2_drop0.25", stage=1, lr=3e-4, weight_decay=1e-2, cnn_dropout=0.25, provenance="freshly_trained_stage1"),
        TuningConfig(run_name="stage1_lr1e-4_wd1e-2_drop0.25", stage=1, lr=1e-4, weight_decay=1e-2, cnn_dropout=0.25, provenance="freshly_trained_stage1"),
    ]

    for cfg in stage1_cfgs:
        res = execute_or_load_run(cfg, train_loader, val_loader, device)
        all_stage_results[cfg.run_name] = res

    best_s1_name, best_s1_res = pick_best_candidate({k: v for k, v in all_stage_results.items() if v["stage"] == 1})
    best_lr = best_s1_res["lr"]
    logger.info("\n>>> STAGE 1 BEST CANDIDATE: %s | Best LR: %.1e (Val Acc: %.4f, Macro-F1: %.4f) <<<",
                best_s1_name, best_lr, best_s1_res["validation_accuracy"], best_s1_res["validation_macro_f1"])

    # =========================================================================
    # STAGE 2: WEIGHT DECAY & DROPOUT SWEEP (around best LR)
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info(">>> STARTING STAGE 2: REGULARIZATION & DROPOUT SWEEP (Best LR = %.1e) <<<", best_lr)
    logger.info("=" * 80)

    # Test weight decay: 1e-3, 5e-2 (1e-2 already tested in Stage 1)
    # Test dropout: 0.15, 0.35 (0.25 already tested in Stage 1)
    stage2_cfgs = [
        TuningConfig(run_name=f"stage2_lr{best_lr:.0e}_wd1e-3_drop0.25", stage=2, lr=best_lr, weight_decay=1e-3, cnn_dropout=0.25, trans_dropout=0.10, provenance="freshly_trained_stage2"),
        TuningConfig(run_name=f"stage2_lr{best_lr:.0e}_wd5e-2_drop0.25", stage=2, lr=best_lr, weight_decay=5e-2, cnn_dropout=0.25, trans_dropout=0.10, provenance="freshly_trained_stage2"),
        TuningConfig(run_name=f"stage2_lr{best_lr:.0e}_wd1e-2_drop0.15", stage=2, lr=best_lr, weight_decay=1e-2, cnn_dropout=0.15, trans_dropout=0.06, provenance="freshly_trained_stage2"),
        TuningConfig(run_name=f"stage2_lr{best_lr:.0e}_wd1e-2_drop0.35", stage=2, lr=best_lr, weight_decay=1e-2, cnn_dropout=0.35, trans_dropout=0.15, provenance="freshly_trained_stage2"),
    ]

    for cfg in stage2_cfgs:
        res = execute_or_load_run(cfg, train_loader, val_loader, device)
        all_stage_results[cfg.run_name] = res

    # Combine Stage 1 best with Stage 2 to find overall best regularization
    s2_pool = {k: v for k, v in all_stage_results.items() if v["stage"] in (1, 2) and v["lr"] == best_lr}
    best_s2_name, best_s2_res = pick_best_candidate(s2_pool)
    best_wd = best_s2_res["weight_decay"]
    best_drop = best_s2_res["cnn_dropout"]
    logger.info("\n>>> STAGE 2 BEST CANDIDATE: %s | WD: %.1e, Drop: %.2f (Val Acc: %.4f, Macro-F1: %.4f) <<<",
                best_s2_name, best_wd, best_drop, best_s2_res["validation_accuracy"], best_s2_res["validation_macro_f1"])

    # =========================================================================
    # STAGE 3: FINE-GRAINED LR ADJUSTMENT (0.5x, 1.5x around best Stage 2 config)
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info(">>> STARTING STAGE 3: FINE-GRAINED LR ADJUSTMENT (0.5x, 1.5x) <<<")
    logger.info("=" * 80)

    lr_half = round(best_lr * 0.5, 6)
    lr_one_half = round(best_lr * 1.5, 6)

    stage3_cfgs = [
        TuningConfig(run_name=f"stage3_lr{lr_half:.1e}_wd{best_wd:.0e}_drop{best_drop:.2f}", stage=3, lr=lr_half, weight_decay=best_wd, cnn_dropout=best_drop, provenance="freshly_trained_stage3"),
        TuningConfig(run_name=f"stage3_lr{lr_one_half:.1e}_wd{best_wd:.0e}_drop{best_drop:.2f}", stage=3, lr=lr_one_half, weight_decay=best_wd, cnn_dropout=best_drop, provenance="freshly_trained_stage3"),
    ]

    for cfg in stage3_cfgs:
        res = execute_or_load_run(cfg, train_loader, val_loader, device)
        all_stage_results[cfg.run_name] = res

    # Overall best selection across all stages
    overall_best_name, overall_best_res = pick_best_candidate(all_stage_results)
    best_acc = overall_best_res["validation_accuracy"]
    best_bal = overall_best_res["validation_balanced_accuracy"]
    best_f1 = overall_best_res["validation_macro_f1"]

    # =========================================================================
    # CONSOLIDATED SUMMARY & REPORTING
    # =========================================================================
    summary_rows = []
    for name, r in all_stage_results.items():
        summary_rows.append({
            "run_name": name,
            "stage": r.get("stage", 1),
            "lr": r.get("lr"),
            "weight_decay": r.get("weight_decay"),
            "cnn_dropout": r.get("cnn_dropout"),
            "provenance": r.get("provenance", "unknown"),
            "training_completed": r.get("training_completed", False),
            "selected_epoch": r.get("selected_epoch"),
            "validation_accuracy": r.get("validation_accuracy"),
            "validation_balanced_accuracy": r.get("validation_balanced_accuracy"),
            "validation_macro_f1": r.get("validation_macro_f1"),
            "validation_weighted_f1": r.get("validation_weighted_f1"),
            "checkpoint_path": r.get("checkpoint_path", ""),
            "runtime_seconds": r.get("runtime_seconds", 0.0),
            "alzheimer_healthy_recall": r["healthy_recall_by_dataset"].get("alzheimer", {}).get("recall", 0.0),
            "parkinson_healthy_recall": r["healthy_recall_by_dataset"].get("parkinson", {}).get("recall", 0.0),
            "modma_healthy_recall": r["healthy_recall_by_dataset"].get("modma", {}).get("recall", 0.0),
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.sort_values(by=["validation_accuracy", "validation_macro_f1"], ascending=[False, False], inplace=True)

    csv_path = EXPERIMENT_DIR / "all_stages_summary.csv"
    json_path = EXPERIMENT_DIR / "all_stages_summary.json"

    df_summary.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "exp3_hyperparameter_optimization",
            "baseline_val_reference": baseline_val,
            "overall_best_run": overall_best_name,
            "overall_best_metrics": overall_best_res,
            "all_runs": all_stage_results,
        }, f, indent=2)

    logger.info("Consolidated results exported to:")
    logger.info("  CSV:  %s", csv_path)
    logger.info("  JSON: %s", json_path)

    # Print summary table
    print("\n" + "=" * 115)
    print("ALL STAGES HYPERPARAMETER OPTIMIZATION SUMMARY (VALIDATION SPLIT ONLY)")
    print("=" * 115)
    print(f"{'Run Name':<38} {'Stage':>5} {'Val Acc':>10} {'Bal Acc':>10} {'Macro F1':>10} {'Status':>10} {'Provenance':<30}")
    print("-" * 115)
    print(f"{'Canonical Baseline Reference':<38} {'-':>5} "
          f"{baseline_val['val_accuracy']*100:>9.2f}% {baseline_val['val_balanced_acc']*100:>9.2f}% "
          f"{baseline_val['val_macro_f1']:>10.4f} {'Baseline':>10} {'canonical best_model.pt':<30}")

    for _, r in df_summary.iterrows():
        status_str = "COMPLETE" if r["training_completed"] else "INCOMPLETE"
        prov_short = str(r["provenance"])[:30]
        print(f"{r['run_name']:<38} {int(r['stage']):>5d} "
              f"{r['validation_accuracy']*100:>9.2f}% {r['validation_balanced_accuracy']*100:>9.2f}% "
              f"{r['validation_macro_f1']:>10.4f} {status_str:>10} {prov_short:<30}")

    print("=" * 115)
    delta = best_acc - baseline_val["val_accuracy"]
    print(f"Overall Best Run: {overall_best_name}")
    print(f"Validation Accuracy: {best_acc*100:.2f}% (Baseline: 76.62% | Delta: {delta*100:+.2f}%)")
    print(f"Balanced Accuracy:   {best_bal*100:.2f}% (Baseline: 83.17%)")
    print(f"Macro-F1:            {best_f1:.4f} (Baseline: 0.7996)")
    print(f"Provenance:          {overall_best_res.get('provenance')}")
    print(f"Checkpoint:          {overall_best_res.get('checkpoint_path')}")
    print("=" * 115)

    return overall_best_res


if __name__ == "__main__":
    run_full_multi_stage_tuning()
