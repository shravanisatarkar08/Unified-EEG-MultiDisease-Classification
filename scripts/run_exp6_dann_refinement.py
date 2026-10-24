"""
scripts/run_exp6_dann_refinement.py - Controlled DANN Refinement Sweep (max_alpha = 0.05 and 0.15).

Compares:
- Exp 5 Champion: max_alpha = 0.10 (Val Acc = 77.02%)
- Candidate A:    max_alpha = 0.05
- Candidate B:    max_alpha = 0.15

All other settings (architecture, seed=42, lr=5e-4, wd=1e-2, epochs=8, batch_size=64,
domain_loss_weight=0.10, linear schedule) remain identical.
Artifacts are saved under results/experiments/exp6_dann_refinement/.
"""

from __future__ import annotations

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
from scripts.run_exp5_dann_transformer import (
    GradReverse,
    grad_reverse,
    DANNCNNTransformer,
    DANNConfig,
    train_one_epoch_dann,
    evaluate_validation,
    DOMAIN_TO_IDX,
    IDX_TO_DOMAIN,
    NUM_DOMAINS,
    CANONICAL_CLASS_WEIGHTS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("exp6_dann_refinement")

EXPERIMENT_DIR = PROJECT_ROOT / "results" / "experiments" / "exp6_dann_refinement"
INDEX_CSV = PROJECT_ROOT / "datasets" / "metadata" / "model_dataset_index.csv"


def run_single_candidate(cfg: DANNConfig, train_loader: DataLoader, val_loader: DataLoader, device: torch.device) -> dict:
    run_dir = EXPERIMENT_DIR / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("[%s] Starting training with max_alpha=%.2f, domain_weight=%.2f",
                cfg.run_name, cfg.max_alpha, cfg.domain_loss_weight)

    # Save run config
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

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
                logger.info("[%s] Early stopping at epoch %d", cfg.run_name, epoch)
                break

    total_runtime = time.time() - t0_all

    # Evaluate best checkpoint
    logger.info("[%s] Final evaluation on best checkpoint: %s", cfg.run_name, best_ckpt_path.name)
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


def run_exp6():
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Executing Exp 6 DANN Refinement on device: %s", device)

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
    logger.info("Integrity check PASSED: 0 overlapping subjects between train and val.")

    # DataLoaders (standard, unmanipulated subject splits)
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

    # Load Exp 5 champion reference
    exp5_metrics_path = PROJECT_ROOT / "results" / "experiments" / "exp5_dann" / "dann_alpha0.10_linear" / "val_metrics.json"
    with open(exp5_metrics_path, "r", encoding="utf-8") as f:
        exp5_ref = json.load(f)

    # 1. Candidate A: max_alpha = 0.05
    cfg_a = DANNConfig(
        run_name="dann_alpha0.05_linear",
        max_alpha=0.05,
        alpha_schedule="linear",
        domain_loss_weight=0.10,
    )
    res_a = run_single_candidate(cfg_a, train_loader, val_loader, device)

    # 2. Candidate B: max_alpha = 0.15
    cfg_b = DANNConfig(
        run_name="dann_alpha0.15_linear",
        max_alpha=0.15,
        alpha_schedule="linear",
        domain_loss_weight=0.10,
    )
    res_b = run_single_candidate(cfg_b, train_loader, val_loader, device)

    # Consolidated comparison
    candidates = {
        "exp5_champion_alpha0.10": exp5_ref,
        res_a["run_name"]: res_a,
        res_b["run_name"]: res_b,
    }

    # Summary table
    summary_rows = []
    for name, r in candidates.items():
        summary_rows.append({
            "configuration": name,
            "max_alpha": r.get("max_alpha", 0.10),
            "selected_epoch": r.get("selected_epoch"),
            "val_accuracy": r.get("validation_accuracy"),
            "val_balanced_acc": r.get("validation_balanced_accuracy"),
            "val_macro_f1": r.get("validation_macro_f1"),
            "domain_probe_acc": r.get("domain_probe_accuracy"),
            "ad_healthy_recall": r.get("healthy_recall_by_dataset", {}).get("alzheimer", {}).get("recall", 0.0),
            "pd_healthy_recall": r.get("healthy_recall_by_dataset", {}).get("parkinson", {}).get("recall", 0.0),
            "modma_healthy_recall": r.get("healthy_recall_by_dataset", {}).get("modma", {}).get("recall", 0.0),
            "checkpoint_path": r.get("checkpoint_path", ""),
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.sort_values(by=["val_accuracy", "val_macro_f1"], ascending=[False, False], inplace=True)
    df_summary.to_csv(EXPERIMENT_DIR / "exp6_summary.csv", index=False)

    # Determine winner between Candidate A and Candidate B
    new_candidates = {res_a["run_name"]: res_a, res_b["run_name"]: res_b}
    best_new_name = max(new_candidates.keys(), key=lambda k: (
        new_candidates[k]["validation_accuracy"],
        new_candidates[k]["validation_macro_f1"],
    ))
    best_new_res = new_candidates[best_new_name]
    best_new_acc = best_new_res["validation_accuracy"]
    exp5_acc = exp5_ref["validation_accuracy"]

    accepted = bool(best_new_acc > exp5_acc)
    champion_name = best_new_name if accepted else "exp5_champion_alpha0.10"
    champion_res = best_new_res if accepted else exp5_ref

    print("\n" + "=" * 105)
    print("EXPERIMENT 6: DANN REFINEMENT (max_alpha = 0.05 vs 0.10 vs 0.15)")
    print("=" * 105)
    print(f"{'Configuration':<30} {'Val Acc':>10} {'Bal Acc':>10} {'Macro-F1':>10} {'Dom Probe':>10} {'AD Hlth':>9} {'PD Hlth':>9}")
    print("-" * 105)
    for _, row in df_summary.iterrows():
        print(f"{row['configuration']:<30} "
              f"{row['val_accuracy']*100:>9.2f}% {row['val_balanced_acc']*100:>9.2f}% "
              f"{row['val_macro_f1']:>10.4f} {row['domain_probe_acc']*100:>9.2f}% "
              f"{row['ad_healthy_recall']*100:>8.1f}% {row['pd_healthy_recall']*100:>8.1f}%")
    print("=" * 105)

    delta = best_new_acc - exp5_acc
    if accepted:
        print(f"CANDIDATE ACCEPTED: {best_new_name} ({best_new_acc*100:.2f}%) > Exp 5 ({exp5_acc*100:.2f}%) | Delta: {delta*100:+.2f}%")
    else:
        print(f"CANDIDATES REJECTED: Best new ({best_new_name}: {best_new_acc*100:.2f}%) <= Exp 5 ({exp5_acc*100:.2f}%)")
        print(f"CHAMPION RETAINED: Exp 5 (dann_alpha0.10_linear: {exp5_acc*100:.2f}%)")
    print("=" * 105)

    with open(EXPERIMENT_DIR / "exp6_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "exp6_dann_refinement",
            "exp5_champion_reference": exp5_ref,
            "candidate_a": res_a,
            "candidate_b": res_b,
            "best_new_candidate": best_new_name,
            "best_new_accuracy": best_new_acc,
            "accepted_new_candidate": accepted,
            "champion_configuration": champion_name,
            "champion_metrics": champion_res,
        }, f, indent=2)

    return champion_res


if __name__ == "__main__":
    run_exp6()
