"""
run_model.py - Run and evaluate the Unified EEG Multi-Disease Classification Model.

Usage:
    python run_model.py              # Run forward pass inference demo
    python run_model.py --train-demo # Run quick end-to-end training demonstration
"""

import sys
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure root directory is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.eeg_classifier import EEGClassifier
from preprocessing.config import N_CHANNELS, WINDOW_SAMPLES, TARGET_SRATE, WINDOW_SEC

# Supported multi-disease classes in the framework
DISEASE_CLASSES = [
    "Healthy Control",
    "Epilepsy (Seizure/Non-Seizure)",
    "Alzheimer's Disease",
    "Parkinson's Disease"
]


def run_inference_demo(device: torch.device):
    print("=" * 70)
    print(" UNIFIED EEG MULTI-DISEASE CLASSIFIER - FORWARD INFERENCE DEMO")
    print("=" * 70)

    # 1. Configuration
    batch_size = 4
    n_channels = N_CHANNELS          # 19 standard 10-20 channels
    n_samples = WINDOW_SAMPLES       # 1280 samples (5s @ 256Hz)
    num_classes = len(DISEASE_CLASSES)

    print(f"[*] Device:             {device}")
    print(f"[*] Batch size:         {batch_size}")
    print(f"[*] EEG Montage:        {n_channels} channels (10-20 system)")
    print(f"[*] Window duration:    {WINDOW_SEC} seconds ({n_samples} samples @ {TARGET_SRATE} Hz)")
    print(f"[*] Target Classes ({num_classes}):")
    for idx, cname in enumerate(DISEASE_CLASSES):
        print(f"    Class {idx}: {cname}")
    print("-" * 70)

    # 2. Instantiate Model
    print("[*] Initializing EEGClassifier (CNN Feature Extractor + Transformer)...")
    model = EEGClassifier(
        n_channels=n_channels,
        n_samples=n_samples,
        num_classes=num_classes,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        ff_dim=128,
        cnn_dropout=0.25,
        trans_dropout=0.1
    ).to(device)
    model.eval()

    # Parameter Breakdown
    cnn_params = sum(p.numel() for p in model.cnn.parameters())
    trans_params = sum(p.numel() for p in model.transformer.parameters())
    total_params = sum(p.numel() for p in model.parameters())

    print(f"    - CNN Extractor Params:     {cnn_params:,}")
    print(f"    - Transformer Params:       {trans_params:,}")
    print(f"    - Total Model Parameters:   {total_params:,}")
    print("-" * 70)

    # 3. Simulate EEG Input Batch
    # Shape: [Batch, Channels, Time]
    torch.manual_seed(42)
    sample_eeg = torch.randn(batch_size, n_channels, n_samples, device=device)
    print(f"[*] Input Tensor Shape: [B, C, T] = {list(sample_eeg.shape)}")

    # 4. Step-by-Step Forward Pass
    start_time = time.time()
    with torch.no_grad():
        # Stage 1: Feature Extraction
        feature_tokens = model.get_features(sample_eeg)  # [B, N, D]
        # Stage 2: Transformer Classification
        logits = model(sample_eeg)                      # [B, num_classes]
        # Probabilities via Softmax
        probabilities = F.softmax(logits, dim=-1)
        predicted_classes = torch.argmax(probabilities, dim=-1)
    latency_ms = (time.time() - start_time) * 1000

    print(f"[*] Stage 1 (CNN): Feature Tokens Shape: [B, N, D] = {list(feature_tokens.shape)}")
    print(f"    (40 time-step tokens with 64 embedding dimensions per window)")
    print(f"[*] Stage 2 (Transformer): Logits Shape: [B, num_classes] = {list(logits.shape)}")
    print(f"[*] Forward pass latency: {latency_ms:.2f} ms ({latency_ms / batch_size:.2f} ms/sample)")
    print("-" * 70)

    # 5. Display Predictions
    print("[*] Sample Predictions across the batch:")
    for b in range(batch_size):
        pred_idx = predicted_classes[b].item()
        pred_label = DISEASE_CLASSES[pred_idx]
        conf = probabilities[b, pred_idx].item() * 100
        print(f"    Sample #{b + 1}: Predicted -> [{pred_label}] (Confidence: {conf:.1f}%)")
        probs_str = ", ".join([f"C{i}: {p*100:.1f}%" for i, p in enumerate(probabilities[b])])
        print(f"              Distribution: [{probs_str}]")

    print("=" * 70)
    print("[SUCCESS] Model executed successfully!")
    print("=" * 70)


def run_training_demo(device: torch.device):
    print("=" * 70)
    print(" UNIFIED EEG MULTI-DISEASE CLASSIFIER - TRAINING DEMO (SYNTHETIC)")
    print("=" * 70)

    n_channels = N_CHANNELS
    n_samples = WINDOW_SAMPLES
    num_classes = len(DISEASE_CLASSES)
    batch_size = 8

    model = EEGClassifier(
        n_channels=n_channels,
        n_samples=n_samples,
        num_classes=num_classes
    ).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    print(f"[*] Training on {device} for 5 demonstration iterations...")
    torch.manual_seed(42)

    for step in range(1, 6):
        # Generate synthetic batch
        x = torch.randn(batch_size, n_channels, n_samples, device=device)
        y = torch.randint(0, num_classes, (batch_size,), device=device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean().item() * 100
        print(f"    Step {step}/5: Loss = {loss.item():.4f}, Batch Accuracy = {acc:.1f}%")

    print("-" * 70)
    print("[SUCCESS] Backpropagation and optimizer step completed without issues!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Run Unified EEG Model")
    parser.add_argument("--train-demo", action="store_true", help="Run a quick synthetic training demonstration")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Compute device")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.train_demo:
        run_training_demo(device)
    else:
        run_inference_demo(device)


if __name__ == "__main__":
    main()
