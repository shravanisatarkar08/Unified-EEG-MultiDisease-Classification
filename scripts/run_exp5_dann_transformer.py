"""
scripts/run_exp5_dann_transformer.py - Domain-Adversarial Neural Network (DANN)
with Gradient Reversal on the CNN-Transformer EEG Classifier.

Objective:
----------
Address the dominant error mode (Healthy controls misclassified as intra-dataset disease)
caused by dataset-source confounding, by training the CNN-Transformer encoder to produce
domain-invariant representations via adversarial gradient reversal.

Key Design Principles:
-----------------------
1. Canonical model definition remains completely undisturbed.
2. Inference interface uses only the standard disease-classification path [B, 19, 1280] -> [B, 5].
3. Gradient Reversal Layer (GRL) and domain head are active during training only.
4. Domain labels (dataset_name) are used strictly as auxiliary training supervision.
   Validation domain labels are never used to train or select models.
5. Standard subject-level split, class weights, optimizer (AdamW, lr=5e-4, wd=1e-2),
   cosine scheduler with 2 warmup epochs, and 8-epoch budget are maintained.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
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
logger = logging.getLogger("exp5_dann")

EXPERIMENT_DIR = PROJECT_ROOT / "results" / "experiments" / "exp5_dann"
INDEX_CSV = PROJECT_ROOT / "datasets" / "metadata" / "model_dataset_index.csv"

# Canonical sqrt-smoothed class weights (identical to baseline)
CANONICAL_CLASS_WEIGHTS = [0.5594, 1.7222, 1.1589, 0.6835, 0.8760]

DOMAIN_TO_IDX = {
    "alzheimer": 0,
    "chbmit": 1,
    "modma": 2,
    "parkinson": 3,
    "srm_healthy": 4,
}
IDX_TO_DOMAIN = {v: k for k, v in DOMAIN_TO_IDX.items()}
NUM_DOMAINS = len(DOMAIN_TO_IDX)


# ============================================================================
# 1. Gradient Reversal Layer (Ganin et al., 2016)
# ============================================================================

class GradReverse(torch.autograd.Function):
    """
    Gradient Reversal Layer:
    Forward:  identity mapping
    Backward: scales gradient by -alpha
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        output = grad_output.neg() * ctx.alpha
        return output, None


def grad_reverse(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return GradReverse.apply(x, alpha)


# ============================================================================
# 2. DANN CNN-Transformer Wrapper (Training-Only Auxiliary Hook)
# ============================================================================

class DANNCNNTransformer(nn.Module):
    """
    Wraps the canonical EEGClassifier to extract the 64-dim CLS embedding
    and attach a domain-classification head via Gradient Reversal.

    During eval mode (inference):
        forward(x) calls the canonical model directly, returning [B, 5] disease logits.
    """

    def __init__(
        self,
        base_model: EEGClassifier,
        num_domains: int = NUM_DOMAINS,
        domain_hidden_dim: int = 32,
        domain_dropout: float = 0.20,
    ):
        super().__init__()
        self.base_model = base_model
        embed_dim = base_model.transformer.embed_dim  # 64

        # Domain adversarial classifier head
        self.domain_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, domain_hidden_dim),
            nn.GELU(),
            nn.Dropout(domain_dropout),
            nn.Linear(domain_hidden_dim, num_domains),
        )

    def extract_cls_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract the 64-dim CLS embedding from the Transformer encoder
        without applying the final disease classification head.
        """
        # CNN stage: [B, 19, 1280] -> [B, N, D]
        features = self.base_model.cnn.extract_features(x)
        B = features.size(0)

        # Transformer stage
        cls = self.base_model.transformer.cls_token.expand(B, -1, -1)  # [B, 1, D]
        h = torch.cat([cls, features], dim=1)                           # [B, N+1, D]
        h = self.base_model.transformer.input_norm(h)
        h = self.base_model.transformer.pos_encoder(h)                  # [B, N+1, D]
        h = self.base_model.transformer.transformer(h)                  # [B, N+1, D]
        cls_embedding = h[:, 0, :]                                      # [B, D]
        return cls_embedding

    def forward(
        self,
        x: torch.Tensor,
        alpha: float = 0.0,
        return_domain: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        If return_domain=True:
            Returns (disease_logits [B, 5], domain_logits [B, num_domains])
        Else:
            Returns disease_logits [B, 5] (canonical inference interface).
        """
        if not return_domain and not self.training:
            # Canonical forward pass for pure inference
            return self.base_model(x), None

        cls_embedding = self.extract_cls_embedding(x)
        disease_logits = self.base_model.transformer.classifier(cls_embedding)

        if return_domain:
            rev_cls = grad_reverse(cls_embedding, alpha)
            domain_logits = self.domain_head(rev_cls)
            return disease_logits, domain_logits

        return disease_logits, None


# ============================================================================
# 3. Training & Evaluation Routines
# ============================================================================

@dataclass
class DANNConfig:
    run_name: str
    max_alpha: float = 0.10        # Maximum GRL coefficient
    alpha_schedule: str = "linear" # "linear", "ganin", or "constant"
    domain_loss_weight: float = 0.10
    lr: float = 5e-4
    weight_decay: float = 1e-2
    warmup_epochs: int = 2
    max_epochs: int = 8
    batch_size: int = 64
    patience: int = 4
    seed: int = 42


def get_alpha(epoch: int, step: int, steps_per_epoch: int, total_epochs: int, cfg: DANNConfig) -> float:
    """Calculate dynamic GRL coefficient alpha based on schedule."""
    if cfg.max_alpha <= 0.0 or cfg.alpha_schedule == "constant":
        return cfg.max_alpha

    curr_step = (epoch - 1) * steps_per_epoch + step
    total_steps = total_epochs * steps_per_epoch
    p = float(curr_step) / max(1, total_steps)

    if cfg.alpha_schedule == "ganin":
        # Standard Ganin sigmoid ramp-up: 2 / (1 + exp(-10*p)) - 1
        return float(cfg.max_alpha * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0))
    elif cfg.alpha_schedule == "linear":
        # Linear ramp-up from 0 to max_alpha
        return float(cfg.max_alpha * p)
    return cfg.max_alpha


def train_one_epoch_dann(
    dann_model: DANNCNNTransformer,
    train_loader: DataLoader,
    disease_criterion: nn.Module,
    domain_criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    total_epochs: int,
    cfg: DANNConfig,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    """Train one epoch with joint disease and domain-adversarial loss."""
    dann_model.train()
    steps_per_epoch = len(train_loader)

    total_disease_loss = 0.0
    total_domain_loss = 0.0
    disease_correct = 0
    domain_correct = 0
    total_samples = 0

    for step, (eeg, disease_labels, metas) in enumerate(train_loader):
        eeg = eeg.to(device)
        disease_labels = disease_labels.to(device)
        domain_labels = torch.tensor(
            [DOMAIN_TO_IDX[m["dataset_name"]] for m in metas],
            dtype=torch.long,
            device=device,
        )

        alpha = get_alpha(epoch, step, steps_per_epoch, total_epochs, cfg)

        optimizer.zero_grad()
        disease_logits, domain_logits = dann_model(eeg, alpha=alpha, return_domain=True)

        loss_disease = disease_criterion(disease_logits, disease_labels)
        loss_domain = domain_criterion(domain_logits, domain_labels)

        loss = loss_disease + cfg.domain_loss_weight * loss_domain
        loss.backward()
        nn.utils.clip_grad_norm_(dann_model.parameters(), 1.0)
        optimizer.step()

        # Metrics
        d_preds = disease_logits.argmax(dim=-1)
        dom_preds = domain_logits.argmax(dim=-1)

        total_disease_loss += loss_disease.item()
        total_domain_loss += loss_domain.item()
        disease_correct += (d_preds == disease_labels).sum().item()
        domain_correct += (dom_preds == domain_labels).sum().item()
        total_samples += disease_labels.size(0)

    n = max(1, steps_per_epoch)
    avg_d_loss = total_disease_loss / n
    avg_dom_loss = total_domain_loss / n
    d_acc = disease_correct / max(1, total_samples)
    dom_acc = domain_correct / max(1, total_samples)

    return avg_d_loss, d_acc, avg_dom_loss, dom_acc


@torch.no_grad()
def evaluate_validation(
    dann_model: DANNCNNTransformer,
    val_loader: DataLoader,
    disease_criterion: nn.Module,
    domain_criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Evaluate on the validation split (standard disease metrics + domain probe diagnostics)."""
    dann_model.eval()
    all_d_preds = []
    all_d_labels = []
    all_dom_preds = []
    all_dom_labels = []
    all_metas = []
    total_val_loss = 0.0

    for eeg, labels, metas in val_loader:
        eeg = eeg.to(device)
        labels = labels.to(device)
        domain_labels = torch.tensor(
            [DOMAIN_TO_IDX[m["dataset_name"]] for m in metas],
            dtype=torch.long,
            device=device,
        )

        # Standard disease logits and domain probe logits
        disease_logits, domain_logits = dann_model(eeg, alpha=0.0, return_domain=True)
        loss = disease_criterion(disease_logits, labels)
        total_val_loss += loss.item()

        d_preds = disease_logits.argmax(dim=-1)
        dom_preds = domain_logits.argmax(dim=-1)

        all_d_preds.extend(d_preds.cpu().tolist())
        all_d_labels.extend(labels.cpu().tolist())
        all_dom_preds.extend(dom_preds.cpu().tolist())
        all_dom_labels.extend(domain_labels.cpu().tolist())
        all_metas.extend(metas)

    n = max(1, len(val_loader))
    y_true = np.array(all_d_labels)
    y_pred = np.array(all_d_preds)
    dom_true = np.array(all_dom_labels)
    dom_pred = np.array(all_dom_preds)

    # Per-class disease metrics
    num_classes = 5
    class_names = [IDX_TO_CLASS[i] for i in range(num_classes)]
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    recalls = []
    f1s = []
    supports = []
    per_class = {}

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

    # Healthy recall by dataset source
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

    # Domain probe accuracy on validation split
    domain_probe_acc = float((dom_true == dom_pred).mean())

    # Build predictions dataframe
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
            "pred_domain": IDX_TO_DOMAIN[int(dom_pred[i])],
            "domain_correct": int(dom_true[i] == dom_pred[i]),
        })

    return {
        "val_loss": round(total_val_loss / n, 4),
        "validation_accuracy": round(acc, 4),
        "validation_balanced_accuracy": round(bal_acc, 4),
        "validation_macro_f1": round(macro_f1, 4),
        "validation_weighted_f1": round(weighted_f1, 4),
        "domain_probe_accuracy": round(domain_probe_acc, 4),
        "per_class": per_class,
        "healthy_recall_by_dataset": healthy_by_ds,
        "confusion_matrix": cm.tolist(),
        "predictions": val_records,
    }


# ============================================================================
# 4. Experiment Runner
# ============================================================================

def run_dann_experiment(cfg: DANNConfig) -> dict:
    run_dir = EXPERIMENT_DIR / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("[%s] Executing DANN experiment on device: %s", cfg.run_name, device)

    # Save run config
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    # Integrity verification
    df_index = pd.read_csv(INDEX_CSV)
    tr_subjs = set(zip(
        df_index[df_index["split"] == "train"]["dataset_name"],
        df_index[df_index["split"] == "train"]["subject_id"],
    ))
    val_subjs = set(zip(
        df_index[df_index["split"] == "val"]["dataset_name"],
        df_index[df_index["split"] == "val"]["subject_id"],
    ))
    assert len(tr_subjs & val_subjs) == 0, f"Subject leakage detected: {tr_subjs & val_subjs}"
    logger.info("[%s] Integrity check PASSED: 0 overlapping subjects between train and val.", cfg.run_name)

    # DataLoaders (standard, unmanipulated subject splits)
    loaders = get_dataloaders(
        index_csv=str(INDEX_CSV),
        project_root=str(PROJECT_ROOT),
        batch_size=cfg.batch_size,
        num_workers=0,
        shuffle_train=True,
        interpolate_missing=False,
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Canonical base model
    base_model = EEGClassifier(
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

    # Wrap in DANN
    dann_model = DANNCNNTransformer(base_model, num_domains=NUM_DOMAINS).to(device)

    disease_weights = torch.tensor(CANONICAL_CLASS_WEIGHTS, dtype=torch.float32).to(device)
    disease_criterion = nn.CrossEntropyLoss(weight=disease_weights)
    domain_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        dann_model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    steps_per_epoch = len(train_loader)
    total_steps = cfg.max_epochs * steps_per_epoch
    warmup_steps = cfg.warmup_epochs * steps_per_epoch

    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_val_acc = 0.0
    best_epoch = 0
    best_ckpt_path: Optional[Path] = None
    no_improve = 0

    history = {
        "train_disease_loss": [], "train_disease_acc": [],
        "train_domain_loss": [], "train_domain_acc": [],
        "val_loss": [], "val_acc": [], "val_bal_acc": [], "val_macro_f1": [],
        "val_domain_probe_acc": [],
    }

    t0_all = time.time()
    for epoch in range(1, cfg.max_epochs + 1):
        t0 = time.time()
        d_loss, d_acc, dom_loss, dom_acc = train_one_epoch_dann(
            dann_model, train_loader, disease_criterion, domain_criterion,
            optimizer, epoch, cfg.max_epochs, cfg, device,
        )

        val_res = evaluate_validation(dann_model, val_loader, disease_criterion, domain_criterion, device)
        v_acc = val_res["validation_accuracy"]
        v_bal = val_res["validation_balanced_accuracy"]
        v_f1 = val_res["validation_macro_f1"]
        v_dom = val_res["domain_probe_accuracy"]

        for _ in range(steps_per_epoch):
            scheduler.step()

        elapsed = time.time() - t0

        history["train_disease_loss"].append(round(d_loss, 4))
        history["train_disease_acc"].append(round(d_acc, 4))
        history["train_domain_loss"].append(round(dom_loss, 4))
        history["train_domain_acc"].append(round(dom_acc, 4))
        history["val_loss"].append(val_res["val_loss"])
        history["val_acc"].append(v_acc)
        history["val_bal_acc"].append(v_bal)
        history["val_macro_f1"].append(v_f1)
        history["val_domain_probe_acc"].append(v_dom)

        logger.info(
            "[%s] Ep %d/%d | d_loss=%.4f acc=%.3f | dom_acc=%.3f | val_acc=%.3f bal=%.3f f1=%.3f dom_probe=%.3f | %.1fs",
            cfg.run_name, epoch, cfg.max_epochs,
            d_loss, d_acc, dom_acc, v_acc, v_bal, v_f1, v_dom, elapsed,
        )

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_epoch = epoch
            if best_ckpt_path and best_ckpt_path.exists():
                best_ckpt_path.unlink()
            best_ckpt_path = run_dir / f"best_model_{cfg.run_name}_ep{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": dann_model.base_model.state_dict(),
                "domain_head_state_dict": dann_model.domain_head.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": v_acc,
                "val_bal_acc": v_bal,
                "val_macro_f1": v_f1,
                "domain_probe_acc": v_dom,
                "config": asdict(cfg),
            }, best_ckpt_path)
            logger.info("  -> Saved new best checkpoint: %s (val_acc=%.4f)", best_ckpt_path.name, v_acc)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                logger.info("[%s] Early stopping triggered at epoch %d", cfg.run_name, epoch)
                break

    total_runtime = time.time() - t0_all

    # Final evaluation on the selected best checkpoint
    logger.info("[%s] Evaluating best checkpoint: %s", cfg.run_name, best_ckpt_path.name)
    best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    dann_model.base_model.load_state_dict(best_ckpt["model_state_dict"])
    dann_model.domain_head.load_state_dict(best_ckpt["domain_head_state_dict"])

    final_metrics = evaluate_validation(dann_model, val_loader, disease_criterion, domain_criterion, device)

    # Save predictions
    df_preds = pd.DataFrame(final_metrics["predictions"])
    df_preds.to_csv(run_dir / "val_predictions.csv", index=False)

    result_summary = {
        "run_name": cfg.run_name,
        "max_alpha": cfg.max_alpha,
        "domain_loss_weight": cfg.domain_loss_weight,
        "alpha_schedule": cfg.alpha_schedule,
        "training_completed": True,
        "selected_epoch": best_epoch,
        "runtime_seconds": round(total_runtime, 2),
        "checkpoint_path": str(best_ckpt_path.relative_to(PROJECT_ROOT)),
        "validation_accuracy": final_metrics["validation_accuracy"],
        "validation_balanced_accuracy": final_metrics["validation_balanced_accuracy"],
        "validation_macro_f1": final_metrics["validation_macro_f1"],
        "validation_weighted_f1": final_metrics["validation_weighted_f1"],
        "domain_probe_accuracy": final_metrics["domain_probe_accuracy"],
        "per_class": final_metrics["per_class"],
        "healthy_recall_by_dataset": final_metrics["healthy_recall_by_dataset"],
        "confusion_matrix": final_metrics["confusion_matrix"],
        "training_history": history,
    }

    with open(run_dir / "val_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result_summary, f, indent=2)

    return result_summary


def run_full_exp5():
    """
    Executes the controlled comparison:
    1. dann_alpha0.0 (GRL disabled control)
    2. dann_alpha0.10_conservative (Conservative DANN with linear GRL ramp-up)
    """
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    baseline_ref = {
        "val_accuracy": 0.7662,
        "val_balanced_acc": 0.8317,
        "val_macro_f1": 0.7996,
        "healthy_recall": {"alzheimer": 0.0333, "modma": 0.9889, "parkinson": 0.4481},
    }

    # 1. Conservative DANN configuration
    cfg_dann = DANNConfig(
        run_name="dann_alpha0.10_linear",
        max_alpha=0.10,
        alpha_schedule="linear",
        domain_loss_weight=0.10,
    )
    res_dann = run_dann_experiment(cfg_dann)

    # Consolidated comparison
    print("\n" + "=" * 105)
    print("EXPERIMENT 5: DOMAIN-ADVERSARIAL NEURAL NETWORK (DANN) COMPARISON")
    print("=" * 105)
    print(f"{'Configuration':<32} {'Val Acc':>10} {'Bal Acc':>10} {'Macro-F1':>10} {'Dom Probe':>10} {'AD Hlth':>10} {'PD Hlth':>10}")
    print("-" * 105)
    print(f"{'Canonical Baseline Reference':<32} "
          f"{baseline_ref['val_accuracy']*100:>9.2f}% {baseline_ref['val_balanced_acc']*100:>9.2f}% "
          f"{baseline_ref['val_macro_f1']:>10.4f} {'98.60%':>10} "
          f"{baseline_ref['healthy_recall']['alzheimer']*100:>9.1f}% "
          f"{baseline_ref['healthy_recall']['parkinson']*100:>9.1f}%")

    ad_h = res_dann["healthy_recall_by_dataset"].get("alzheimer", {}).get("recall", 0.0)
    pd_h = res_dann["healthy_recall_by_dataset"].get("parkinson", {}).get("recall", 0.0)
    print(f"{res_dann['run_name']:<32} "
          f"{res_dann['validation_accuracy']*100:>9.2f}% {res_dann['validation_balanced_accuracy']*100:>9.2f}% "
          f"{res_dann['validation_macro_f1']:>10.4f} {res_dann['domain_probe_accuracy']*100:>9.2f}% "
          f"{ad_h*100:>9.1f}% {pd_h*100:>9.1f}%")

    print("=" * 105)
    cand_acc = res_dann["validation_accuracy"]
    delta = cand_acc - baseline_ref["val_accuracy"]
    if cand_acc > baseline_ref["val_accuracy"]:
        print(f"CANDIDATE WINNER: {cand_acc*100:.2f}% > 76.62% (Delta: {delta*100:+.2f}%)")
    else:
        print(f"CANDIDATE REJECTED: {cand_acc*100:.2f}% <= 76.62% (Delta: {delta*100:+.2f}%)")
        print("Canonical baseline best_model.pt remains preserved.")
    print("=" * 105)

    # Save comparison summary
    with open(EXPERIMENT_DIR / "exp5_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "exp5_dann_transformer",
            "baseline_ref": baseline_ref,
            "dann_candidate": res_dann,
            "accepted": bool(cand_acc > baseline_ref["val_accuracy"]),
        }, f, indent=2)


if __name__ == "__main__":
    run_full_exp5()
