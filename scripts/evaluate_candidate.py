"""
scripts/evaluate_candidate.py - Single controlled evaluation of validation-selected candidate model on held-out test set.

Compares:
  - Baseline model: checkpoints/best_model.pt (Weighted CrossEntropy)
  - Candidate model: results/experiments/run2_label_smoothing/best_model_ep007_acc0.7575.pt (Label-Smoothed CrossEntropy)

Outputs saved strictly to:
  - results/experiments/run2_label_smoothing/test_metrics.json
  - results/experiments/run2_label_smoothing/test_classification_report.csv
  - results/experiments/run2_label_smoothing/test_confusion_matrix.png
Preserves all existing baseline artifacts unchanged.
"""

import json
import logging
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.eeg_classifier import EEGClassifier
from training.dataset import get_dataloaders

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_candidate")

CLASS_NAMES = ["healthy", "epilepsy", "alzheimers", "parkinsons", "depression"]
DISPLAY_NAMES = ["Healthy", "Epilepsy", "Alzheimer's", "Parkinson's", "Depression"]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray, n_classes: int = 5):
    """Compute comprehensive multiclass metrics using pure NumPy."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1

    total_samples = len(y_true)
    acc = float((y_true == y_pred).mean()) if total_samples > 0 else 0.0

    per_class = {}
    recalls, f1s, supports = [], [], []

    for c in range(n_classes):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        support = int(cm[c, :].sum())
        prec = float(tp) / max(1, tp + fp) if (tp + fp) > 0 else 0.0
        rec = float(tp) / max(1, tp + fn) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / max(1e-9, prec + rec)) if (prec + rec) > 0 else 0.0

        per_class[DISPLAY_NAMES[c]] = {
            "class_id": c,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "support": support,
        }
        if support > 0:
            recalls.append(rec)
        f1s.append(f1)
        supports.append(support)

    bal_acc = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s))
    weighted_f1 = float(np.sum(np.array(f1s) * np.array(supports)) / max(1, total_samples))

    # Cohen's Kappa
    actual_marginals = cm.sum(axis=1) / max(1, total_samples)
    pred_marginals = cm.sum(axis=0) / max(1, total_samples)
    pe = float(np.sum(actual_marginals * pred_marginals))
    kappa = float((acc - pe) / max(1e-9, 1.0 - pe)) if pe < 1.0 else 0.0

    # Macro ROC-AUC (OVR rank sum)
    aucs = []
    for c in range(n_classes):
        y_bin = (y_true == c).astype(int)
        n_pos = int(y_bin.sum())
        n_neg = len(y_bin) - n_pos
        if n_pos > 0 and n_neg > 0:
            scores = y_probs[:, c]
            ranks = scores.argsort().argsort() + 1
            pos_rank_sum = np.sum(ranks[y_bin == 1])
            auc_c = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
            aucs.append(float(auc_c))
    macro_auc = float(np.mean(aucs)) if aucs else 0.0

    return {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "cohen_kappa": round(kappa, 4),
        "macro_roc_auc": round(macro_auc, 4),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def load_model_from_checkpoint(ckpt_path: Path, device: torch.device) -> EEGClassifier:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    if isinstance(cfg, dict):
        kwargs = {
            "n_channels": cfg.get("n_channels", 19),
            "n_samples": cfg.get("n_samples", 1280),
            "num_classes": cfg.get("num_classes", 5),
            "embed_dim": cfg.get("embed_dim", 64),
            "num_heads": cfg.get("num_heads", 4),
            "num_layers": cfg.get("num_layers", 2),
            "ff_dim": cfg.get("ff_dim", 128),
            "cnn_dropout": cfg.get("cnn_dropout", 0.25),
            "trans_dropout": cfg.get("trans_dropout", 0.1),
        }
    else:
        kwargs = {
            "n_channels": cfg.n_channels,
            "n_samples": cfg.n_samples,
            "num_classes": cfg.num_classes,
            "embed_dim": cfg.embed_dim,
            "num_heads": cfg.num_heads,
            "num_layers": cfg.num_layers,
            "ff_dim": cfg.ff_dim,
            "cnn_dropout": cfg.cnn_dropout,
            "trans_dropout": cfg.trans_dropout,
        }
    model = EEGClassifier(**kwargs).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def main():
    logger.info("=" * 80)
    logger.info("MILESTONE 4: FINAL CONTROLLED TEST EVALUATION")
    logger.info("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Evaluating on device: %s", device)

    # Paths
    candidate_ckpt = PROJECT_ROOT / "results" / "experiments" / "run2_label_smoothing" / "best_model_ep007_acc0.7575.pt"
    baseline_metrics_path = PROJECT_ROOT / "results" / "metrics.json"

    if not candidate_ckpt.exists():
        logger.error("Candidate checkpoint not found at: %s", candidate_ckpt)
        sys.exit(1)

    # 1. Load Test DataLoader
    loaders = get_dataloaders(
        index_csv=str(PROJECT_ROOT / "datasets" / "metadata" / "model_dataset_index.csv"),
        project_root=str(PROJECT_ROOT),
        batch_size=64,
        num_workers=0,
        shuffle_train=False,
    )
    test_loader = loaders["test"]
    logger.info("Loaded held-out test split: %d windows", len(test_loader.dataset))

    # 2. Evaluate Candidate
    candidate_model = load_model_from_checkpoint(candidate_ckpt, device)
    preds, targets, probs = [], [], []

    with torch.no_grad():
        for x, y, _ in test_loader:
            x = x.to(device)
            logits = candidate_model(x)
            pr = F.softmax(logits, dim=-1)
            preds.extend(logits.argmax(dim=-1).cpu().numpy())
            targets.extend(y.numpy())
            probs.extend(pr.cpu().numpy())

    cand_metrics = compute_metrics(np.array(targets), np.array(preds), np.array(probs), n_classes=5)

    # 3. Load Baseline Metrics for comparison
    with open(baseline_metrics_path, "r", encoding="utf-8") as f:
        base_json = json.load(f)
    base_metrics = base_json["test_metrics"]
    base_per_class = base_json["per_class_metrics"]

    # 4. Save Candidate Results
    out_dir = candidate_ckpt.parent
    test_metrics_file = out_dir / "test_metrics.json"
    with open(test_metrics_file, "w", encoding="utf-8") as f:
        json.dump({
            "model": "run2_label_smoothing",
            "checkpoint": str(candidate_ckpt),
            "test_metrics": cand_metrics,
            "provenance": "Evaluated strictly on held-out test split (1525 windows, zero subject overlap)",
            "known_limitation": "Epilepsy test partition contains 55 windows from single held-out subject chb01.",
        }, f, indent=2)

    # Save CSV classification report
    report_rows = []
    for c_name, m in cand_metrics["per_class"].items():
        report_rows.append({
            "class_id": m["class_id"],
            "class_name": c_name,
            "precision": m["precision"],
            "recall": m["recall"],
            "f1_score": m["f1_score"],
            "support": m["support"],
        })
    pd.DataFrame(report_rows).to_csv(out_dir / "test_classification_report.csv", index=False)

    # Plot Confusion Matrix
    cm = np.array(cand_metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(8, 7))
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm.astype(float) / row_sums

    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized Fraction", rotation=270, labelpad=15)

    ax.set(
        xticks=np.arange(len(DISPLAY_NAMES)),
        yticks=np.arange(len(DISPLAY_NAMES)),
        xticklabels=DISPLAY_NAMES,
        yticklabels=DISPLAY_NAMES,
        title=f"Candidate (run2_label_smoothing) Test Confusion Matrix (Acc: {cand_metrics['accuracy']*100:.2f}%)",
        ylabel="True Class (Ground Truth)",
        xlabel="Predicted Class",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = cm_norm[i, j] * 100
            txt = f"{val}\n({pct:.1f}%)" if val > 0 else f"{val}"
            ax.text(
                j, i, txt,
                ha="center", va="center",
                color="white" if cm_norm[i, j] > 0.5 else "black",
                fontweight="bold" if val > 0 else "normal",
                fontsize=9,
            )
    plt.tight_layout()
    fig.savefig(out_dir / "test_confusion_matrix.png", dpi=300)
    plt.close(fig)

    # 5. Print Side-by-Side Comparison
    print("\n" + "=" * 80)
    print("  FINAL HELD-OUT TEST EVALUATION: BASELINE vs CANDIDATE")
    print("=" * 80)
    print(f"{'Metric':<25} {'Baseline (run0)':>18} {'Candidate (run2)':>18} {'Difference':>15}")
    print("-" * 80)

    diff_acc = cand_metrics["accuracy"] - base_metrics["accuracy"]
    diff_bal = cand_metrics["balanced_accuracy"] - base_metrics["balanced_accuracy"]
    diff_macro = cand_metrics["macro_f1"] - base_metrics["macro_f1"]
    diff_weight = cand_metrics["weighted_f1"] - base_metrics["weighted_f1"]
    diff_kappa = cand_metrics["cohen_kappa"] - base_metrics["cohen_kappa"]
    diff_auc = cand_metrics["macro_roc_auc"] - base_metrics["macro_roc_auc"]

    print(f"{'Accuracy':<25} {base_metrics['accuracy']*100:>17.2f}% {cand_metrics['accuracy']*100:>17.2f}% {diff_acc*100:>+14.2f}%")
    print(f"{'Balanced Accuracy':<25} {base_metrics['balanced_accuracy']*100:>17.2f}% {cand_metrics['balanced_accuracy']*100:>17.2f}% {diff_bal*100:>+14.2f}%")
    print(f"{'Macro F1-Score':<25} {base_metrics['macro_f1']:>18.4f} {cand_metrics['macro_f1']:>18.4f} {diff_macro:>+15.4f}")
    print(f"{'Weighted F1-Score':<25} {base_metrics['weighted_f1']:>18.4f} {cand_metrics['weighted_f1']:>18.4f} {diff_weight:>+15.4f}")
    print(f"{'Cohen Kappa':<25} {base_metrics['cohen_kappa']:>18.4f} {cand_metrics['cohen_kappa']:>18.4f} {diff_kappa:>+15.4f}")
    print(f"{'Macro ROC-AUC':<25} {base_metrics['macro_roc_auc']:>18.4f} {cand_metrics['macro_roc_auc']:>18.4f} {diff_auc:>+15.4f}")

    print("\n" + "-" * 80)
    print("PER-CLASS TEST METRICS BREAKDOWN (CANDIDATE)")
    print("-" * 80)
    print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 80)
    for c_name, m in cand_metrics["per_class"].items():
        print(f"{c_name:<15} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1_score']:>10.4f} {m['support']:>10d}")
    print("=" * 80)


if __name__ == "__main__":
    main()
