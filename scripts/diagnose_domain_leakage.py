"""
scripts/diagnose_domain_leakage.py - Frozen-Feature Dataset-Domain Diagnostic Probe.

Purpose:
--------
Measures whether the canonical baseline model's latent intermediate representation
encodes the recording site (dataset_name) rather than only disease-related physiology.

Method:
-------
1. Loads canonical checkpoint `checkpoints/best_model.pt` in evaluation mode with frozen weights.
2. Extracts one feature vector per EEG window for:
   - Primary: Transformer CLS token output (64-dim, directly feeding the classification head)
   - Secondary: CNN average-pooled tokens (64-dim, feature extractor output before Transformer)
3. Trains a linear probe on the training split features to predict `dataset_name`.
4. Evaluates probe on the validation split features only.
5. Strictly avoids any evaluation or fitting on the held-out test split.
6. Compares performance against a majority-class baseline.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Path bootstrap
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.eeg_classifier import EEGClassifier
from training.dataset import get_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("domain_probe")

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pt"
INDEX_CSV = PROJECT_ROOT / "datasets" / "metadata" / "model_dataset_index.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "experiments" / "diagnostics"
OUTPUT_FILE = OUTPUT_DIR / "domain_probe_metrics.json"


def load_canonical_frozen_model(device: torch.device) -> Tuple[EEGClassifier, dict]:
    """Load canonical baseline model and freeze all parameters."""
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Canonical checkpoint not found: {CHECKPOINT_PATH}")

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    if isinstance(cfg, dict):
        model = EEGClassifier(
            n_channels=cfg.get("n_channels", 19),
            n_samples=cfg.get("n_samples", 1280),
            num_classes=cfg.get("num_classes", 5),
            embed_dim=cfg.get("embed_dim", 64),
            num_heads=cfg.get("num_heads", 4),
            num_layers=cfg.get("num_layers", 2),
            ff_dim=cfg.get("ff_dim", 128),
            cnn_dropout=cfg.get("cnn_dropout", 0.25),
            trans_dropout=cfg.get("trans_dropout", 0.1),
        )
    else:
        model = EEGClassifier(
            n_channels=getattr(cfg, "n_channels", 19),
            n_samples=getattr(cfg, "n_samples", 1280),
            num_classes=getattr(cfg, "num_classes", 5),
            embed_dim=getattr(cfg, "embed_dim", 64),
            num_heads=getattr(cfg, "num_heads", 4),
            num_layers=getattr(cfg, "num_layers", 2),
            ff_dim=getattr(cfg, "ff_dim", 128),
            cnn_dropout=getattr(cfg, "cnn_dropout", 0.25),
            trans_dropout=getattr(cfg, "trans_dropout", 0.1),
        )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Strictly freeze all weights
    for param in model.parameters():
        param.requires_grad = False

    logger.info("Loaded canonical model from %s (epoch %d, val_acc %.4f). All weights frozen.",
                CHECKPOINT_PATH, ckpt.get("epoch", -1), ckpt.get("val_acc", 0.0))
    return model, ckpt


def extract_split_features(
    model: EEGClassifier,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    split_name: str,
) -> Dict[str, np.ndarray]:
    """
    Extract intermediate representations from the frozen model for a given split.
    Uses identical preprocessing (interpolate_missing=False as in baseline).
    """
    cls_features: List[torch.Tensor] = []
    cnn_features: List[torch.Tensor] = []
    domain_labels: List[str] = []
    subject_ids: List[str] = []
    disease_classes: List[str] = []

    # Register forward hook to capture input to the classifier head (CLS token representation)
    captured_cls: List[torch.Tensor] = []

    def hook_fn(module, inp, outp):
        captured_cls.append(inp[0].detach().cpu())

    hook_handle = model.transformer.classifier.register_forward_hook(hook_fn)

    t0 = time.time()
    with torch.no_grad():
        for batch_idx, (eeg, labels, metas) in enumerate(loader):
            eeg = eeg.to(device)
            captured_cls.clear()

            # Forward pass through CNN feature extractor
            cnn_tokens = model.cnn.extract_features(eeg)  # [B, N, D]
            cnn_pooled = cnn_tokens.mean(dim=1).detach().cpu()  # [B, D]

            # Forward pass through Transformer (triggers hook at classifier input)
            _ = model.transformer(cnn_tokens)
            cls_token = captured_cls[0]  # [B, D]

            cls_features.append(cls_token)
            cnn_features.append(cnn_pooled)

            for meta in metas:
                domain_labels.append(meta["dataset_name"])
                subject_ids.append(meta["subject_id"])
                disease_classes.append(meta["class_category"])

    hook_handle.remove()
    elapsed = time.time() - t0

    cls_all = torch.cat(cls_features, dim=0).numpy()
    cnn_all = torch.cat(cnn_features, dim=0).numpy()

    logger.info("Split '%s': Extracted %d windows in %.2fs (cls_shape=%s, cnn_shape=%s)",
                split_name, len(domain_labels), elapsed, cls_all.shape, cnn_all.shape)

    return {
        "cls_features": cls_all,
        "cnn_features": cnn_all,
        "domain_labels": np.array(domain_labels),
        "subject_ids": np.array(subject_ids),
        "disease_classes": np.array(disease_classes),
    }


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> dict:
    """Compute accuracy, balanced accuracy, macro F1, and per-class metrics."""
    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    per_class = {}
    recalls_active = []
    f1s_active = []
    f1s_all = []

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
        f1s_all.append(f1)
        if sup > 0:
            recalls_active.append(rec)
            f1s_active.append(f1)

    acc = float((y_true == y_pred).mean())
    bal_acc = float(np.mean(recalls_active)) if recalls_active else 0.0
    macro_f1 = float(np.mean(f1s_active)) if f1s_active else 0.0

    return {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "macro_f1_all_classes": round(float(np.mean(f1s_all)), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "n_samples": len(y_true),
    }


def train_and_eval_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    domain_names: List[str],
    representation_name: str,
    device: torch.device,
    epochs: int = 150,
    lr: float = 0.01,
    weight_decay: float = 1e-3,
    seed: int = 42,
) -> dict:
    """Train a linear classifier probe on X_train to predict y_train; evaluate on X_val."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Standardize features using training statistics
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-7

    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std

    x_tr_tensor = torch.tensor(X_train_norm, dtype=torch.float32).to(device)
    y_tr_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

    x_val_tensor = torch.tensor(X_val_norm, dtype=torch.float32).to(device)

    in_dim = X_train.shape[1]
    num_classes = len(domain_names)

    probe = nn.Linear(in_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Train linear probe on training features only
    probe.train()
    for ep in range(epochs):
        optimizer.zero_grad()
        logits = probe(x_tr_tensor)
        loss = criterion(logits, y_tr_tensor)
        loss.backward()
        optimizer.step()

    # Evaluate on training and validation features
    probe.eval()
    with torch.no_grad():
        tr_logits = probe(x_tr_tensor)
        tr_preds = tr_logits.argmax(dim=-1).cpu().numpy()

        val_logits = probe(x_val_tensor)
        val_preds = val_logits.argmax(dim=-1).cpu().numpy()

    tr_metrics = compute_multiclass_metrics(y_train, tr_preds, domain_names)
    val_metrics = compute_multiclass_metrics(y_val, val_preds, domain_names)

    # Majority-class baseline: predict the most frequent training domain for all samples
    maj_domain_idx = int(pd.Series(y_train).mode()[0])
    maj_domain_name = domain_names[maj_domain_idx]
    val_maj_preds = np.full_like(y_val, maj_domain_idx)
    maj_baseline_metrics = compute_multiclass_metrics(y_val, val_maj_preds, domain_names)

    return {
        "representation": representation_name,
        "input_dimension": in_dim,
        "probe_architecture": f"Linear({in_dim} -> {num_classes})",
        "probe_parameters": sum(p.numel() for p in probe.parameters()),
        "majority_class_baseline": {
            "majority_training_domain": maj_domain_name,
            "majority_domain_index": maj_domain_idx,
            "accuracy": maj_baseline_metrics["accuracy"],
            "balanced_accuracy": maj_baseline_metrics["balanced_accuracy"],
            "macro_f1": maj_baseline_metrics["macro_f1"],
        },
        "train_metrics": {
            "accuracy": tr_metrics["accuracy"],
            "balanced_accuracy": tr_metrics["balanced_accuracy"],
            "macro_f1": tr_metrics["macro_f1"],
            "per_class": tr_metrics["per_class"],
        },
        "validation_metrics": {
            "accuracy": val_metrics["accuracy"],
            "balanced_accuracy": val_metrics["balanced_accuracy"],
            "macro_f1": val_metrics["macro_f1"],
            "macro_f1_all_classes": val_metrics["macro_f1_all_classes"],
            "per_class": val_metrics["per_class"],
            "confusion_matrix": val_metrics["confusion_matrix"],
        },
        "comparison_delta_vs_majority": {
            "accuracy_delta": round(val_metrics["accuracy"] - maj_baseline_metrics["accuracy"], 4),
            "balanced_accuracy_delta": round(val_metrics["balanced_accuracy"] - maj_baseline_metrics["balanced_accuracy"], 4),
            "macro_f1_delta": round(val_metrics["macro_f1"] - maj_baseline_metrics["macro_f1"], 4),
        },
    }


def run_diagnostic():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    logger.info("=" * 80)
    logger.info("STEP 1: FROZEN-FEATURE DATASET-DOMAIN DIAGNOSTIC PROBE")
    logger.info("=" * 80)

    # 1. Load canonical model and freeze
    model, ckpt = load_canonical_frozen_model(device)

    # 2. Build DataLoaders (only train and val; test is never touched)
    logger.info("Loading train and val DataLoaders (test partition untouched)...")
    loaders = get_dataloaders(
        index_csv=str(INDEX_CSV),
        project_root=str(PROJECT_ROOT),
        batch_size=64,
        num_workers=0,
        shuffle_train=False,
        interpolate_missing=False,  # Baseline identical preprocessing
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    # Verify split row counts against CSV
    df = pd.read_csv(INDEX_CSV)
    expected_train = len(df[df["split"] == "train"])
    expected_val = len(df[df["split"] == "val"])
    assert len(train_loader.dataset) == expected_train, f"Train count mismatch: {len(train_loader.dataset)} != {expected_train}"
    assert len(val_loader.dataset) == expected_val, f"Val count mismatch: {len(val_loader.dataset)} != {expected_val}"
    logger.info("Verified split sizes: Train=%d, Val=%d", expected_train, expected_val)

    # 3. Extract features for train and val
    train_data = extract_split_features(model, train_loader, device, "train")
    val_data = extract_split_features(model, val_loader, device, "val")

    # 4. Domain mapping (sorted unique training domains)
    domain_names = sorted(list(set(train_data["domain_labels"])))
    domain_to_idx = {name: i for i, name in enumerate(domain_names)}
    logger.info("Domain categories (%d): %s", len(domain_names), domain_names)

    # Class counts by domain
    train_counts = pd.Series(train_data["domain_labels"]).value_counts().to_dict()
    val_counts = pd.Series(val_data["domain_labels"]).value_counts().to_dict()
    logger.info("Train domain counts: %s", train_counts)
    logger.info("Val domain counts:   %s", val_counts)

    y_train = np.array([domain_to_idx[d] for d in train_data["domain_labels"]])
    y_val = np.array([domain_to_idx[d] for d in val_data["domain_labels"]])

    # 5. Fit probe on Primary representation: Transformer CLS token (64-dim)
    logger.info("\n--- Training Probe on Primary Representation: Transformer CLS Token [64-dim] ---")
    cls_probe_results = train_and_eval_linear_probe(
        X_train=train_data["cls_features"],
        y_train=y_train,
        X_val=val_data["cls_features"],
        y_val=y_val,
        domain_names=domain_names,
        representation_name="Transformer CLS Token (input to classification head)",
        device=device,
    )

    # 6. Fit probe on Secondary representation: CNN pooled tokens (64-dim)
    logger.info("\n--- Training Probe on Secondary Representation: CNN Average-Pooled Tokens [64-dim] ---")
    cnn_probe_results = train_and_eval_linear_probe(
        X_train=train_data["cnn_features"],
        y_train=y_train,
        X_val=val_data["cnn_features"],
        y_val=y_val,
        domain_names=domain_names,
        representation_name="CNN Average-Pooled Features (input to Transformer)",
        device=device,
    )

    # 7. Compile report and save JSON
    output_data = {
        "diagnostic": "Dataset-Domain Confounding Probe (Step 1)",
        "model_checkpoint": str(CHECKPOINT_PATH),
        "model_architecture": "EEGClassifier (CNN-Transformer, 81,994 params)",
        "model_weights_status": "FROZEN (eval mode, zero parameter updates)",
        "domain_classes": domain_names,
        "split_counts": {
            "train_total": len(y_train),
            "val_total": len(y_val),
            "train_by_domain": train_counts,
            "val_by_domain": val_counts,
        },
        "primary_probe_transformer_cls": cls_probe_results,
        "secondary_probe_cnn_pooled": cnn_probe_results,
        "scientific_interpretation": {
            "high_domain_score_meaning": (
                "A high domain probe score indicates that dataset acquisition identity is strongly "
                "linearly separable in the model's frozen latent space. This confirms that intermediate "
                "layers retain significant domain signatures."
            ),
            "causality_caveat": (
                "This linear probe demonstrates domain separability but does not alone prove causal reliance "
                "on recording artifacts, nor does it prove disease biomarkers are absent."
            ),
        },
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    logger.info("Results successfully saved to: %s", OUTPUT_FILE)

    # Print summary table
    print("\n" + "=" * 75)
    print("DOMAIN PROBE DIAGNOSTIC SUMMARY (VALIDATION SPLIT ONLY)")
    print("=" * 75)
    print(f"Majority-class baseline ({cls_probe_results['majority_class_baseline']['majority_training_domain']}):")
    print(f"  Accuracy:          {cls_probe_results['majority_class_baseline']['accuracy'] * 100:.2f}%")
    print(f"  Balanced Accuracy: {cls_probe_results['majority_class_baseline']['balanced_accuracy'] * 100:.2f}%")
    print(f"  Macro-F1:          {cls_probe_results['majority_class_baseline']['macro_f1']:.4f}")
    print("-" * 75)
    print("Linear Probe on Transformer CLS Token [64-dim] (Pre-Classification Latent):")
    val_cls = cls_probe_results["validation_metrics"]
    print(f"  Accuracy:          {val_cls['accuracy'] * 100:.2f}%  (Δ vs baseline: +{cls_probe_results['comparison_delta_vs_majority']['accuracy_delta'] * 100:.2f}%)")
    print(f"  Balanced Accuracy: {val_cls['balanced_accuracy'] * 100:.2f}%  (Δ vs baseline: +{cls_probe_results['comparison_delta_vs_majority']['balanced_accuracy_delta'] * 100:.2f}%)")
    print(f"  Macro-F1:          {val_cls['macro_f1']:.4f}  (Δ vs baseline: +{cls_probe_results['comparison_delta_vs_majority']['macro_f1_delta']:.4f})")
    print("  Per-Domain Recall in Validation:")
    for dom, m in val_cls["per_class"].items():
        if m["support"] > 0:
            print(f"    {dom:<12}: Recall={m['recall'] * 100:5.1f}%  (Prec={m['precision'] * 100:5.1f}%, F1={m['f1_score']:.4f}, n={m['support']})")
    print("-" * 75)
    print("Linear Probe on CNN Average-Pooled Features [64-dim] (Pre-Transformer):")
    val_cnn = cnn_probe_results["validation_metrics"]
    print(f"  Accuracy:          {val_cnn['accuracy'] * 100:.2f}%  (Δ vs baseline: +{cnn_probe_results['comparison_delta_vs_majority']['accuracy_delta'] * 100:.2f}%)")
    print(f"  Balanced Accuracy: {val_cnn['balanced_accuracy'] * 100:.2f}%  (Δ vs baseline: +{cnn_probe_results['comparison_delta_vs_majority']['balanced_accuracy_delta'] * 100:.2f}%)")
    print(f"  Macro-F1:          {val_cnn['macro_f1']:.4f}")
    print("=" * 75)

    return output_data


if __name__ == "__main__":
    run_diagnostic()
