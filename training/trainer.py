"""
training/trainer.py - Training Loop for Unified EEG Multi-Disease Classifier.

Implements a complete training loop with:
  - AdamW optimizer + cosine LR schedule with warm-up
  - CrossEntropyLoss with optional class weighting
  - Per-epoch train/val loss and accuracy metrics
  - Best-model checkpoint saving (by val accuracy)
  - Early stopping (patience-based)
  - TensorBoard summary writing (optional, gracefully skipped if unavailable)

Usage (synthetic demo, no real data required):
    python -m training.trainer --demo
"""

from __future__ import annotations

import logging
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Path bootstrap
_MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _MODULE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.eeg_classifier import EEGClassifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Hyperparameters and paths for one training run."""
    # Model
    n_channels:    int   = 19
    n_samples:     int   = 1280
    num_classes:   int   = 5
    embed_dim:     int   = 64
    num_heads:     int   = 4
    num_layers:    int   = 2
    ff_dim:        int   = 128
    cnn_dropout:   float = 0.25
    trans_dropout: float = 0.1

    # Optimisation
    lr:            float = 1e-3
    weight_decay:  float = 1e-2
    warmup_epochs: int   = 2
    max_epochs:    int   = 50
    batch_size:    int   = 32

    # Early stopping
    patience:      int   = 10

    # I/O
    index_csv:     str   = ""
    project_root:  str   = str(PROJECT_ROOT)
    checkpoint_dir: str  = str(PROJECT_ROOT / "checkpoints")
    num_workers:   int   = 0   # 0 = safe on Windows

    # Class weights (None = uniform)
    class_weights: Optional[List[float]] = None


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine LR schedule with linear warm-up."""
    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

@dataclass
class EpochMetrics:
    loss: float = 0.0
    correct: int = 0
    total:   int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / max(1, self.total)

    def update(self, loss: float, preds: torch.Tensor, labels: torch.Tensor) -> None:
        self.loss    += loss
        self.correct += (preds == labels).sum().item()
        self.total   += labels.size(0)

    def avg_loss(self, n_batches: int) -> float:
        return self.loss / max(1, n_batches)


# ---------------------------------------------------------------------------
# Core Trainer
# ---------------------------------------------------------------------------

class EEGTrainer:
    """End-to-end training manager for EEGClassifier."""

    def __init__(self, config: TrainConfig):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("EEGTrainer initialised on %s", self.device)

        # Build model
        self.model = EEGClassifier(
            n_channels=config.n_channels,
            n_samples=config.n_samples,
            num_classes=config.num_classes,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            ff_dim=config.ff_dim,
            cnn_dropout=config.cnn_dropout,
            trans_dropout=config.trans_dropout,
        ).to(self.device)

        # Loss
        weights = (torch.tensor(config.class_weights, dtype=torch.float32).to(self.device)
                   if config.class_weights else None)
        self.criterion = nn.CrossEntropyLoss(weight=weights)

        # Optimiser
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # Checkpointing
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.best_val_acc  = 0.0
        self.best_ckpt_path: Optional[Path] = None

        # History
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   [],
        }

        # TensorBoard (optional)
        self._tb_writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = Path(config.checkpoint_dir) / "tensorboard"
            self._tb_writer = SummaryWriter(str(tb_dir))
            logger.info("TensorBoard logging -> %s", tb_dir)
        except ImportError:
            logger.info("TensorBoard not available; skipping TB logging.")

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def _train_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        m = EpochMetrics()
        for eeg, labels, _ in loader:
            eeg    = eeg.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(eeg)
            loss   = self.criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            preds = logits.argmax(dim=-1)
            m.update(loss.item(), preds, labels)

        n = max(1, len(loader))
        return m.avg_loss(n), m.accuracy

    # ------------------------------------------------------------------
    # Validation step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        m = EpochMetrics()
        for eeg, labels, _ in loader:
            eeg    = eeg.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(eeg)
            loss   = self.criterion(logits, labels)
            preds  = logits.argmax(dim=-1)
            m.update(loss.item(), preds, labels)
        n = max(1, len(loader))
        return m.avg_loss(n), m.accuracy

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch: int, val_acc: float) -> Path:
        ckpt_path = Path(self.cfg.checkpoint_dir) / f"best_model_ep{epoch:03d}_acc{val_acc:.4f}.pt"
        torch.save({
            "epoch":            epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state":  self.optimizer.state_dict(),
            "val_acc":          val_acc,
            "config":           self.cfg,
        }, ckpt_path)
        return ckpt_path

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        logger.info("Loaded checkpoint from %s (epoch %d, val_acc %.4f)",
                    path, ckpt["epoch"], ckpt["val_acc"])
        return ckpt["epoch"]

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, List[float]]:
        """Run full training loop.

        Returns
        -------
        history dict with train_loss, train_acc, val_loss, val_acc per epoch.
        """
        steps_per_epoch = len(train_loader)
        total_steps     = self.cfg.max_epochs * steps_per_epoch
        warmup_steps    = self.cfg.warmup_epochs * steps_per_epoch

        scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps
        )

        no_improve = 0

        for epoch in range(1, self.cfg.max_epochs + 1):
            t0 = time.time()

            train_loss, train_acc = self._train_epoch(train_loader)
            val_loss,   val_acc   = self._val_epoch(val_loader)

            # Step scheduler once per batch (already stepped inside _train_epoch
            # via closure, so step at epoch level here)
            for _ in range(steps_per_epoch):
                scheduler.step()

            elapsed = time.time() - t0

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            logger.info(
                "Epoch %3d/%d | train_loss=%.4f acc=%.3f | val_loss=%.4f acc=%.3f | %.1fs",
                epoch, self.cfg.max_epochs,
                train_loss, train_acc, val_loss, val_acc, elapsed,
            )

            if self._tb_writer:
                self._tb_writer.add_scalar("Loss/train", train_loss, epoch)
                self._tb_writer.add_scalar("Loss/val",   val_loss,   epoch)
                self._tb_writer.add_scalar("Acc/train",  train_acc,  epoch)
                self._tb_writer.add_scalar("Acc/val",    val_acc,    epoch)

            # Save best checkpoint
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                if self.best_ckpt_path and self.best_ckpt_path.exists():
                    self.best_ckpt_path.unlink()
                self.best_ckpt_path = self.save_checkpoint(epoch, val_acc)
                logger.info("  New best model saved -> %s", self.best_ckpt_path)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.cfg.patience:
                    logger.info("Early stopping at epoch %d (patience=%d)",
                                epoch, self.cfg.patience)
                    break

        if self._tb_writer:
            self._tb_writer.close()

        return self.history


# ---------------------------------------------------------------------------
# Convenience entry-point (synthetic demo)
# ---------------------------------------------------------------------------

def run_demo(num_classes: int = 5, n_epochs: int = 5):
    """Smoke-test the training loop with synthetic EEG data."""
    from torch.utils.data import TensorDataset

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    B, C, T = 32, 19, 1280
    torch.manual_seed(42)

    # Synthetic datasets
    def _make_loader(n_batches: int) -> DataLoader:
        xs = torch.randn(n_batches * B, C, T)
        ys = torch.randint(0, num_classes, (n_batches * B,))
        ds = TensorDataset(xs, ys)

        def _collate(batch):
            eeg = torch.stack([b[0] for b in batch])
            lbl = torch.stack([b[1] for b in batch])
            meta = [{}] * len(batch)
            return eeg, lbl, meta

        return DataLoader(ds, batch_size=B, collate_fn=_collate)

    train_loader = _make_loader(8)
    val_loader   = _make_loader(2)

    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg = TrainConfig(
            num_classes=num_classes,
            max_epochs=n_epochs,
            patience=n_epochs + 1,
            checkpoint_dir=tmp_dir,
            lr=1e-3,
        )
        trainer = EEGTrainer(cfg)
        history = trainer.train(train_loader, val_loader)

    print("\n[OK] Training demo completed.")
    print(f"  Final train_acc: {history['train_acc'][-1]:.3f}")
    print(f"  Final val_acc:   {history['val_acc'][-1]:.3f}")
    return history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EEG Classifier Training Loop")
    parser.add_argument("--demo",        action="store_true", help="Run synthetic demo")
    parser.add_argument("--index-csv",   default="", help="Path to model_dataset_index.csv")
    parser.add_argument("--checkpoint-dir", default=str(PROJECT_ROOT / "checkpoints"))
    parser.add_argument("--epochs",      type=int, default=50)
    parser.add_argument("--batch-size",  type=int, default=32)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--num-classes", type=int, default=5)
    args = parser.parse_args()

    if args.demo or not args.index_csv:
        run_demo(num_classes=args.num_classes, n_epochs=min(args.epochs, 5))
    else:
        from training.dataset import get_dataloaders
        loaders = get_dataloaders(
            index_csv=args.index_csv,
            project_root=str(PROJECT_ROOT),
            batch_size=args.batch_size,
        )
        cfg = TrainConfig(
            num_classes=args.num_classes,
            max_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            checkpoint_dir=args.checkpoint_dir,
        )
        trainer = EEGTrainer(cfg)
        trainer.train(loaders["train"], loaders.get("val", loaders["train"]))
