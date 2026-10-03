"""
Generate presentation-quality Grad-CAM and saliency visualizations for all 5 classes.
Saves individual figures into results/gradcam/ and composite results/gradcam_analysis.png.
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.dataset import get_dataloaders
from training.explainability import EEGExplainer
from models.eeg_classifier import EEGClassifier

INDEX_CSV = ROOT / "datasets" / "metadata" / "model_dataset_index.csv"
CHECKPOINT_PATH = ROOT / "checkpoints" / "best_model.pt"
RESULTS_DIR = ROOT / "results"
GRADCAM_DIR = RESULTS_DIR / "gradcam"
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["healthy", "epilepsy", "alzheimers", "parkinsons", "depression"]
DISPLAY_NAMES = ["Healthy Control", "Epilepsy (CHB-MIT)", "Alzheimer's Disease", "Parkinson's Disease", "Major Depression"]
CHANNELS = ["Fp1","Fp2","F7","F3","Fz","F4","F8","T3","C3","Cz","C4","T4","T5","P3","Pz","P4","T6","O1","O2"]
COLORS = ["#1f77b4", "#d62728", "#9467bd", "#ff7f0e", "#2ca02c"]

def generate_all():
    print("Loading model and dataloaders...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    embed_dim = cfg.get("embed_dim", 64) if isinstance(cfg, dict) else getattr(cfg, "embed_dim", 64)
    num_heads = cfg.get("num_heads", 4) if isinstance(cfg, dict) else getattr(cfg, "num_heads", 4)
    num_layers = cfg.get("num_layers", 2) if isinstance(cfg, dict) else getattr(cfg, "num_layers", 2)
    ff_dim = cfg.get("ff_dim", 128) if isinstance(cfg, dict) else getattr(cfg, "ff_dim", 128)

    model = EEGClassifier(19, 1280, 5, embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, ff_dim=ff_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    device = torch.device("cpu")
    explainer = EEGExplainer(model, device)
    loaders = get_dataloaders(str(INDEX_CSV), str(ROOT), batch_size=32, num_workers=0)

    # Collect representative samples for each class
    samples = {}
    for split_name in ["test", "val", "train"]:
        loader = loaders[split_name]
        for x_b, y_b, meta_b in loader:
            for i in range(len(y_b)):
                lbl = int(y_b[i].item())
                if lbl not in samples:
                    if isinstance(meta_b, list):
                        s_id = meta_b[i].get("subject_id", "unknown") if isinstance(meta_b[i], dict) else "unknown"
                        s_ds = meta_b[i].get("dataset", "unknown") if isinstance(meta_b[i], dict) else "unknown"
                    elif isinstance(meta_b, dict):
                        s_id = meta_b["subject_id"][i] if isinstance(meta_b["subject_id"], (list, tuple)) else str(meta_b["subject_id"])
                        s_ds = meta_b["dataset"][i] if isinstance(meta_b["dataset"], (list, tuple)) else str(meta_b["dataset"])
                    else:
                        s_id, s_ds = "unknown", "unknown"
                    
                    with torch.no_grad():
                        logits = model(x_b[i:i+1])
                        probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
                        pred_class = int(np.argmax(probs))
                        conf = float(probs[pred_class])
                    
                    samples[lbl] = {
                        "tensor": x_b[i:i+1],
                        "subject": s_id,
                        "dataset": s_ds,
                        "split": split_name,
                        "pred_class": pred_class,
                        "confidence": conf,
                        "probs": probs
                    }
            if len(samples) >= 5:
                break
        if len(samples) >= 5:
            break

    time_sec = np.linspace(0, 5, 1280)

    # 1. Generate individual class figures in results/gradcam/
    for c_idx, c_name in enumerate(CLASS_NAMES):
        d_name = DISPLAY_NAMES[c_idx]
        samp = samples[c_idx]
        x_samp = samp["tensor"]
        
        cam = explainer.grad_cam(x_samp, target_class=c_idx)
        sal = explainer.input_gradients(x_samp, target_class=c_idx)
        eeg_np = x_samp.squeeze().detach().cpu().numpy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
        fig.suptitle(
            f"Explainability Analysis: {d_name} (True: {CLASS_NAMES[c_idx]}, Pred: {CLASS_NAMES[samp['pred_class']]}, Conf: {samp['confidence']*100:.1f}%)",
            fontsize=12, fontweight="bold"
        )

        # Waveform + Grad-CAM
        ch_idx = 9 if eeg_np.shape[0] > 9 else 0
        ax1.plot(time_sec, eeg_np[ch_idx], color="#333333", lw=1.2, label=f"EEG Cz Waveform")
        ax1_t = ax1.twinx()
        ax1_t.plot(time_sec, cam, color=COLORS[c_idx], lw=2.2, linestyle="--", label="Grad-CAM Activation")
        ax1_t.fill_between(time_sec, 0, cam, color=COLORS[c_idx], alpha=0.25)
        ax1_t.set_ylabel("Grad-CAM Weight", color=COLORS[c_idx], fontweight="bold")
        ax1_t.set_ylim(-0.05, 1.05)
        ax1.set_title(f"Temporal Importance [{samp['dataset']} / {samp['subject']} ({samp['split']})]", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Time (seconds)")
        ax1.set_ylabel("Standardized Amplitude (z-score)")
        ax1.grid(True, alpha=0.3)

        # Spatial attribution
        chan_sal = sal.mean(axis=1)
        ax2.bar(CHANNELS, chan_sal, color=COLORS[c_idx], edgecolor="black", alpha=0.85)
        ax2.set_title("19-Channel Electrode Spatial Gradient Saliency", fontsize=11, fontweight="bold")
        ax2.set_xlabel("10-20 Electrode Montage")
        ax2.set_ylabel("Mean Gradient Attribution")
        ax2.tick_params(axis="x", rotation=45)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = GRADCAM_DIR / f"gradcam_{c_name}.png"
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"Saved: {save_path}")

    # 2. Composite multi-panel figure
    fig, axes = plt.subplots(5, 2, figsize=(16, 18))
    fig.suptitle("Unified Explainability: CNN Grad-CAM Temporal Activation & Spatial Electrode Saliency", fontsize=14, fontweight="bold", y=0.995)

    for c_idx, c_name in enumerate(CLASS_NAMES):
        d_name = DISPLAY_NAMES[c_idx]
        samp = samples[c_idx]
        x_samp = samp["tensor"]
        cam = explainer.grad_cam(x_samp, target_class=c_idx)
        sal = explainer.input_gradients(x_samp, target_class=c_idx)
        eeg_np = x_samp.squeeze().detach().cpu().numpy()

        ax_wave = axes[c_idx, 0]
        ax_bar = axes[c_idx, 1]

        ch_idx = 9 if eeg_np.shape[0] > 9 else 0
        ax_wave.plot(time_sec, eeg_np[ch_idx], color="#333333", lw=1.2, label=f"EEG Cz signal")
        ax_twin = ax_wave.twinx()
        ax_twin.plot(time_sec, cam, color=COLORS[c_idx], lw=2.2, linestyle="--", label=f"Grad-CAM ({d_name})")
        ax_twin.fill_between(time_sec, 0, cam, color=COLORS[c_idx], alpha=0.25)
        ax_twin.set_ylabel("Grad-CAM Activation", color=COLORS[c_idx], fontweight="bold")
        ax_twin.set_ylim(-0.05, 1.05)

        ax_wave.set_title(f"{d_name} [{samp['dataset']} / {samp['subject']} ({samp['split']})]: Waveform & Temporal Grad-CAM", fontweight="bold", fontsize=11)
        ax_wave.set_xlabel("Time (seconds)")
        ax_wave.set_ylabel("Amplitude (z-score)")
        ax_wave.grid(True, alpha=0.3)

        chan_sal = sal.mean(axis=1)
        ax_bar.bar(CHANNELS, chan_sal, color=COLORS[c_idx], edgecolor="black", alpha=0.85)
        ax_bar.set_title(f"{d_name}: Spatial Channel Saliency Attribution (19 Electrodes)", fontweight="bold", fontsize=11)
        ax_bar.set_xlabel("Electrode Channel")
        ax_bar.set_ylabel("Mean Gradient Saliency")
        ax_bar.tick_params(axis="x", rotation=45)
        ax_bar.grid(True, alpha=0.3)

    plt.tight_layout()
    composite_path = RESULTS_DIR / "gradcam_analysis.png"
    fig.savefig(composite_path, dpi=300)
    plt.close(fig)
    print(f"Saved composite: {composite_path}")

if __name__ == "__main__":
    generate_all()
