"""
training/evaluator.py - Evaluation + Per-Class Metrics for EEG Classifier.

Computes:
  - Overall accuracy, macro F1, weighted F1
  - Per-class precision, recall, F1, support
  - Confusion matrix
  - Optional: classification report (sklearn)

Usage:
    from training.evaluator import evaluate_model
    report = evaluate_model(model, test_loader, device, class_names)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Path bootstrap
_MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _MODULE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

CLASS_NAMES = ["healthy", "epilepsy", "alzheimers", "parkinsons", "depression"]


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
    return_probas: bool = False,
) -> dict:
    """Evaluate model on a DataLoader and return a metrics dict.

    Parameters
    ----------
    model        : trained EEGClassifier
    loader       : DataLoader yielding (eeg, labels, meta) triples
    device       : torch device
    class_names  : list of class name strings (length = num_classes)
    return_probas: if True, include raw softmax probabilities in output

    Returns
    -------
    dict with keys:
      accuracy, macro_f1, weighted_f1,
      confusion_matrix (np.ndarray [C, C]),
      per_class (dict: class_name -> {precision, recall, f1, support}),
      all_preds, all_labels, [all_probas]
    """
    if class_names is None:
        class_names = CLASS_NAMES

    model.eval()
    all_preds:  List[int]         = []
    all_labels: List[int]         = []
    all_probas: List[np.ndarray]  = []

    with torch.no_grad():
        for batch in loader:
            eeg, labels, _ = batch
            eeg    = eeg.to(device)
            labels = labels.to(device)
            logits = model(eeg)
            probas = F.softmax(logits, dim=-1)
            preds  = probas.argmax(dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            if return_probas:
                all_probas.extend(probas.cpu().numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    n_classes = len(class_names)

    # Confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1

    # Per-class metrics
    per_class: Dict[str, dict] = {}
    for c, name in enumerate(class_names):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        support = int(cm[c, :].sum())
        precision = float(tp) / max(1, tp + fp)
        recall    = float(tp) / max(1, tp + fn)
        f1 = (2 * precision * recall / max(1e-9, precision + recall))
        per_class[name] = {
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
            "support":   support,
        }

    # Aggregate metrics
    supports = np.array([per_class[n]["support"] for n in class_names], dtype=float)
    f1s      = np.array([per_class[n]["f1"]      for n in class_names], dtype=float)

    macro_f1    = float(f1s.mean())
    weighted_f1 = float((f1s * supports).sum() / max(1.0, supports.sum()))
    accuracy    = float((y_true == y_pred).mean())

    result: dict = {
        "accuracy":         round(accuracy, 4),
        "macro_f1":         round(macro_f1, 4),
        "weighted_f1":      round(weighted_f1, 4),
        "confusion_matrix": cm,
        "per_class":        per_class,
        "all_preds":        y_pred,
        "all_labels":       y_true,
        "n_samples":        len(y_true),
    }
    if return_probas:
        result["all_probas"] = np.array(all_probas)

    return result


def print_report(metrics: dict, class_names: Optional[List[str]] = None) -> None:
    """Pretty-print an evaluation report."""
    if class_names is None:
        class_names = CLASS_NAMES
    print("=" * 60)
    print(f"  Overall Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Macro F1         : {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1      : {metrics['weighted_f1']:.4f}")
    print(f"  N samples        : {metrics['n_samples']}")
    print("-" * 60)
    print(f"  {'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 60)
    for name in class_names:
        m = metrics["per_class"].get(name, {})
        print(f"  {name:<15} {m.get('precision',0):>10.4f} {m.get('recall',0):>10.4f}"
              f" {m.get('f1',0):>10.4f} {m.get('support',0):>10d}")
    print("=" * 60)


if __name__ == "__main__":
    # Smoke test
    import torch
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from models.eeg_classifier import EEGClassifier
    from torch.utils.data import TensorDataset

    B, C, T, K = 16, 19, 1280, 5
    device = torch.device("cpu")
    model  = EEGClassifier(n_channels=C, n_samples=T, num_classes=K).eval()

    xs = torch.randn(B, C, T)
    ys = torch.randint(0, K, (B,))

    def _collate(batch):
        eeg = torch.stack([b[0] for b in batch])
        lbl = torch.stack([b[1] for b in batch])
        return eeg, lbl, [{}] * len(batch)

    from torch.utils.data import DataLoader
    loader = DataLoader(TensorDataset(xs, ys), batch_size=4, collate_fn=_collate)

    metrics = evaluate_model(model, loader, device, return_probas=False)
    print_report(metrics)
    print("\n[OK] Evaluator smoke test passed!")
