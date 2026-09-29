"""
Baseline real-data training run for the Unified EEG Multi-Disease CNN-Transformer.

SCIENTIFIC NOTE:
    Only 4 of 5 target classes have local data available:
        0 = healthy   (srm_healthy + controls)
        1 = epilepsy  (CHB-MIT chb01)
        2 = alzheimers (ds004504 AD subjects)
        3 = parkinsons (ds004584 PD subjects)
        4 = depression — UNAVAILABLE locally (MODMA archives unextracted)

    This run is therefore a **4-class development experiment**.
    5-class full experiment is BLOCKED pending MODMA extraction.
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from training.dataset import get_dataloaders, CLASS_TO_IDX
from training.trainer import EEGTrainer, TrainConfig
from training.evaluator import evaluate_model, print_report
from models.eeg_classifier import EEGClassifier

INDEX_CSV = ROOT / "datasets" / "metadata" / "model_dataset_index.csv"
CHECKPOINT_DIR = ROOT / "checkpoints"


def _split_info(loader, name, logger):
    if loader is None:
        logger.info("  %s split : EMPTY (no samples)", name)
        return set(), 0
    cats = set(loader.dataset.index_df["class_category"].unique())
    n = len(loader.dataset)
    logger.info("  %s split : %d windows | classes present: %s", name, n, sorted(cats))
    return cats, n


def run_baseline_training():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(ROOT / "baseline_training.log", mode="w"),
        ],
    )
    logger = logging.getLogger(__name__)

    CHECKPOINT_DIR.mkdir(exist_ok=True)

    logger.info("=" * 70)
    logger.info("BASELINE REAL-DATA TRAINING — CNN-Transformer EEG Classifier")
    logger.info("=" * 70)
    logger.info("Loading dataset index: %s", INDEX_CSV)

    loaders = get_dataloaders(
        index_csv=str(INDEX_CSV),
        project_root=str(ROOT),
        batch_size=4,       # Small: only 15 train + 5 val windows in dev index
        num_workers=0,
        shuffle_train=True,
    )

    train_loader = loaders.get("train")
    val_loader   = loaders.get("val")
    test_loader  = loaders.get("test")

    train_cats, n_train = _split_info(train_loader, "train", logger)
    val_cats,   n_val   = _split_info(val_loader,   "val  ", logger)
    test_cats,  n_test  = _split_info(test_loader,  "test ", logger)

    all_local_cats = train_cats | val_cats | test_cats
    local_class_indices = sorted([CLASS_TO_IDX[c] for c in all_local_cats])
    logger.info("Local class indices in use : %s", local_class_indices)
    logger.info("Depression (class 4) in data: %s", "depression" in all_local_cats)

    if train_loader is None:
        logger.error("Train split is empty — cannot run training.")
        sys.exit(1)
    if val_loader is None:
        logger.warning("Val split is empty — using train loader for validation (overfitted estimate).")
        val_loader = train_loader

    # Training configuration — CPU-feasible, small data
    cfg = TrainConfig(
        n_channels=19,
        n_samples=1280,
        num_classes=5,          # 5-class head (depression slot will have 0 support)
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        ff_dim=128,
        cnn_dropout=0.25,
        trans_dropout=0.1,
        lr=1e-3,
        weight_decay=1e-2,
        warmup_epochs=1,
        max_epochs=20,          # Keep short for CPU
        batch_size=4,
        patience=20,            # Disable early stopping for this short run
        checkpoint_dir=str(CHECKPOINT_DIR),
        num_workers=0,
        class_weights=None,
    )

    logger.info("-" * 70)
    logger.info("TrainConfig: num_classes=%d, epochs=%d, lr=%.4f, batch_size=%d",
                cfg.num_classes, cfg.max_epochs, cfg.lr, cfg.batch_size)

    # Train
    t_start = time.time()
    trainer = EEGTrainer(cfg)
    history = trainer.train(train_loader, val_loader)
    elapsed = time.time() - t_start

    logger.info("-" * 70)
    logger.info("Training complete in %.1fs", elapsed)
    logger.info("Best val_acc achieved : %.4f", trainer.best_val_acc)
    logger.info("Checkpoint saved at   : %s", trainer.best_ckpt_path)

    # Evaluate best model on val split
    device = trainer.device
    model  = trainer.model

    if trainer.best_ckpt_path and trainer.best_ckpt_path.exists():
        ckpt = torch.load(
            trainer.best_ckpt_path,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Best checkpoint loaded for evaluation.")

    CLASS_NAMES = ["healthy", "epilepsy", "alzheimers", "parkinsons", "depression"]

    logger.info("\n=== VALIDATION EVALUATION ===")
    val_metrics = evaluate_model(model, val_loader, device, class_names=CLASS_NAMES)
    print_report(val_metrics, class_names=CLASS_NAMES)

    logger.info("Confusion matrix (val):\n%s", val_metrics["confusion_matrix"])

    # Test split (likely empty in current dev index)
    if test_loader is not None:
        logger.info("\n=== TEST EVALUATION ===")
        test_metrics = evaluate_model(model, test_loader, device, class_names=CLASS_NAMES)
        print_report(test_metrics, class_names=CLASS_NAMES)
        logger.info("Confusion matrix (test):\n%s", test_metrics["confusion_matrix"])
    else:
        logger.info("\n=== TEST EVALUATION ===")
        logger.info("SKIPPED — test split is empty in current development index.")
        test_metrics = None

    # Final summary
    print("\n" + "=" * 70)
    print("  BASELINE TRAINING RUN — FINAL SUMMARY")
    print("=" * 70)
    print(f"  Training windows      : {n_train}")
    print(f"  Validation windows    : {n_val}")
    print(f"  Test windows          : {n_test}")
    print(f"  Classes in data       : {sorted(all_local_cats)}")
    print(f"  Class indices used    : {local_class_indices}")
    print(f"  Depression available  : NO — MODMA archives unextracted")
    print(f"  Experiment type       : 4-CLASS DEVELOPMENT RUN (NOT 5-class)")
    print(f"  Epochs trained        : {len(history['train_loss'])}")
    print(f"  Final train loss      : {history['train_loss'][-1]:.4f}")
    print(f"  Final train acc       : {history['train_acc'][-1]:.4f}")
    print(f"  Final val loss        : {history['val_loss'][-1]:.4f}")
    print(f"  Final val acc         : {history['val_acc'][-1]:.4f}")
    print(f"  Best val acc          : {trainer.best_val_acc:.4f}")
    print(f"  Val accuracy (eval)   : {val_metrics['accuracy']:.4f}")
    print(f"  Val macro F1          : {val_metrics['macro_f1']:.4f}")
    print(f"  Val weighted F1       : {val_metrics['weighted_f1']:.4f}")
    if test_metrics:
        print(f"  Test accuracy (eval)  : {test_metrics['accuracy']:.4f}")
    else:
        print(f"  Test accuracy (eval)  : N/A — empty test split")
    print(f"  Training time         : {elapsed:.1f}s")
    print(f"  Checkpoint            : {trainer.best_ckpt_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_baseline_training()
