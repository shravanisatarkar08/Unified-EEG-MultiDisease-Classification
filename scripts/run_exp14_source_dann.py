"""Exp14: source-balanced DANN on the established train/validation split only."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_exp4_source_balanced import build_source_balanced_loader
from scripts.run_exp5_dann_transformer import (
    CANONICAL_CLASS_WEIGHTS,
    DANNConfig,
    DANNCNNTransformer,
    evaluate_validation,
    train_one_epoch_dann,
)
from models.eeg_classifier import EEGClassifier
from training.dataset import EEGDataset, _eeg_collate_fn
from training.trainer import get_cosine_schedule_with_warmup
import torch.nn as nn

OUT = ROOT / "results" / "experiments" / "exp14_source_dann"
INDEX = ROOT / "datasets" / "metadata" / "model_dataset_index.csv"
EXPECTED_MODEL = "e8cbd5bd9ec79b6e46524fbc2cad7b84a59c22d83c07a5fe2d335f3021441111"
EXPECTED_METRICS = "5f0a10003b0fe06b383614c097ab5287ae526a073a90dd420dda2bec1862ff3a"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def verify_canonical() -> None:
    assert sha256(ROOT / "checkpoints" / "best_model.pt") == EXPECTED_MODEL
    assert sha256(ROOT / "results" / "metrics.json") == EXPECTED_METRICS


def main() -> None:
    verify_canonical()
    OUT.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(INDEX)
    train_df = df[df.split == "train"].copy()
    val_df = df[df.split == "val"].copy()
    train_subjects = set(zip(train_df.dataset_name, train_df.subject_id))
    val_subjects = set(zip(val_df.dataset_name, val_df.subject_id))
    assert not train_subjects & val_subjects

    train_loader, source_weights = build_source_balanced_loader(train_df, ROOT, batch_size=64)
    val_dataset = EEGDataset(val_df.reset_index(drop=True), ROOT, cache_size=2, interpolate_missing=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0, collate_fn=_eeg_collate_fn)

    cfg = DANNConfig(run_name="exp14_source_dann", max_alpha=0.15, alpha_schedule="linear", domain_loss_weight=0.10)
    with (OUT / "config.json").open("w") as f:
        json.dump({**asdict(cfg), "source_sampler": "inverse dataset frequency", "source_weights": source_weights,
                   "n_train": len(train_df), "n_val": len(val_df), "test_accessed": False}, f, indent=2)

    base = EEGClassifier(n_channels=19, n_samples=1280, num_classes=5, cnn_dropout=0.25,
                         embed_dim=64, num_heads=4, num_layers=2, ff_dim=128, trans_dropout=0.10).to(device)
    model = DANNCNNTransformer(base).to(device)
    disease_criterion = nn.CrossEntropyLoss(weight=torch.tensor(CANONICAL_CLASS_WEIGHTS, dtype=torch.float32, device=device))
    domain_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg.warmup_epochs * steps, cfg.max_epochs * steps)
    history = {k: [] for k in ("train_disease_loss", "train_disease_acc", "train_domain_loss", "train_domain_acc", "val_loss", "val_acc", "val_bal_acc", "val_macro_f1", "val_domain_probe_acc")}
    best_acc, best_epoch, best_path, stalled = -1.0, 0, None, 0

    for epoch in range(1, cfg.max_epochs + 1):
        dl, da, ol, oa = train_one_epoch_dann(model, train_loader, disease_criterion, domain_criterion, optimizer, epoch, cfg.max_epochs, cfg, device)
        val = evaluate_validation(model, val_loader, disease_criterion, domain_criterion, device)
        for _ in range(steps): scheduler.step()
        values = (dl, da, ol, oa, val["val_loss"], val["validation_accuracy"], val["validation_balanced_accuracy"], val["validation_macro_f1"], val["domain_probe_accuracy"])
        for key, value in zip(history, values): history[key].append(round(float(value), 4))
        if val["validation_accuracy"] > best_acc:
            if best_path and best_path.exists(): best_path.unlink()
            best_acc, best_epoch, stalled = val["validation_accuracy"], epoch, 0
            best_path = OUT / f"best_model_exp14_ep{epoch:03d}.pt"
            torch.save({"epoch": epoch, "model_state_dict": model.base_model.state_dict(), "domain_head_state_dict": model.domain_head.state_dict(), "config": asdict(cfg)}, best_path)
        else:
            stalled += 1
            if stalled >= cfg.patience: break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.base_model.load_state_dict(checkpoint["model_state_dict"])
    model.domain_head.load_state_dict(checkpoint["domain_head_state_dict"])
    final = evaluate_validation(model, val_loader, disease_criterion, domain_criterion, device)
    pd.DataFrame(final["predictions"]).to_csv(OUT / "val_predictions.csv", index=False)
    result = {"experiment": "exp14_source_dann", "training_completed": True, "test_accessed": False,
              "selected_epoch": best_epoch, "checkpoint_path": str(best_path.relative_to(ROOT)),
              "validation_accuracy": final["validation_accuracy"], "validation_balanced_accuracy": final["validation_balanced_accuracy"],
              "validation_macro_f1": final["validation_macro_f1"], "validation_weighted_f1": final["validation_weighted_f1"],
              "domain_probe_accuracy": final["domain_probe_accuracy"], "per_class": final["per_class"],
              "healthy_recall_by_dataset": final["healthy_recall_by_dataset"], "confusion_matrix": final["confusion_matrix"],
              "training_history": history, "exp12_reference": {"accuracy": 0.7742, "balanced_accuracy": 0.8372, "macro_f1": 0.8085}}
    with (OUT / "val_metrics.json").open("w") as f: json.dump(result, f, indent=2)
    verify_canonical()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()