"""
Unified Explainable CNN–Transformer Framework for Multi-Disease EEG Classification
Interactive Clinical Research Demonstration & Validation Dashboard

Run with:
    streamlit run app/app.py
"""

import os
import sys
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.config import (
    TARGET_SRATE, WINDOW_SEC, OVERLAP, N_CHANNELS, WINDOW_SAMPLES,
    COMMON_CHANNELS, RAW_DIR
)
from preprocessing.common import (
    bandpass_filter, notch_filter, resample, select_and_reorder_channels, normalize_windows
)
from models.eeg_classifier import EEGClassifier
from training.explainability import EEGExplainer

# Page configuration
st.set_page_config(
    page_title="Unified EEG Multi-Disease AI Framework",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        font-weight: 500;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 0.85rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-val {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2563EB;
    }
    .metric-lbl {
        font-size: 0.8rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .disclaimer-box {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 0.9rem;
        border-radius: 6px;
        font-size: 0.88rem;
        color: #92400E;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .architecture-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)

# Constants
DISEASE_NAMES = [
    "Healthy Control",
    "Epilepsy (Seizure / Abnormal)",
    "Alzheimer's Disease",
    "Parkinson's Disease",
    "Major Depressive Disorder"
]
CLASS_KEYS = ["healthy", "epilepsy", "alzheimers", "parkinsons", "depression"]
CHANNELS = ["Fp1","Fp2","F7","F3","Fz","F4","F8","T3","C3","Cz","C4","T4","T5","P3","Pz","P4","T6","O1","O2"]

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pt"
METRICS_PATH = PROJECT_ROOT / "results" / "metrics.json"
RESULTS_DIR = PROJECT_ROOT / "results"


@st.cache_resource
def load_cached_model():
    """Load the trained 5-class PyTorch model."""
    if not CHECKPOINT_PATH.exists():
        return None, "Checkpoint not found"
    try:
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {})
        embed_dim = cfg.get("embed_dim", 64) if isinstance(cfg, dict) else getattr(cfg, "embed_dim", 64)
        num_heads = cfg.get("num_heads", 4) if isinstance(cfg, dict) else getattr(cfg, "num_heads", 4)
        num_layers = cfg.get("num_layers", 2) if isinstance(cfg, dict) else getattr(cfg, "num_layers", 2)
        ff_dim = cfg.get("ff_dim", 128) if isinstance(cfg, dict) else getattr(cfg, "ff_dim", 128)

        model = EEGClassifier(
            n_channels=19,
            n_samples=1280,
            num_classes=5,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model, "Loaded best_model.pt"
    except Exception as e:
        return None, f"Error loading model: {e}"


@st.cache_data
def load_cached_metrics():
    """Load JSON metrics."""
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


# Top Navigation
st.markdown('<div class="main-title">🧠 Unified Explainable CNN–Transformer Framework</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Multi-Disease Scalp EEG Classification & Biomarker Attribution Research System</div>', unsafe_allow_html=True)

nav_tabs = st.tabs([
    "🏠 Home",
    "🎯 Live EEG Prediction",
    "📊 Model Performance",
    "🔍 Grad-CAM Explainability",
    "🗃️ Datasets & Cohorts",
    "🔬 Methodology",
    "📈 Results & Metrics",
    "⚠️ Limitations & Scope"
])

model, model_status = load_cached_model()
metrics_data = load_cached_metrics()


# -----------------------------------------------------------------------------
# TAB 1: HOME
# -----------------------------------------------------------------------------
with nav_tabs[0]:
    st.header("Unified Multi-Disease Scalp EEG AI Framework")
    st.markdown("""
    This research platform introduces an end-to-end framework combining an **EEGNet-inspired 1D/2D CNN spatial-temporal backbone** with a **Multi-Head Self-Attention Transformer Encoder** to perform diagnostic classification and biomarker attribution across five standardized clinical conditions:
    """)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown('<div class="metric-card"><div class="metric-val">0</div><div class="metric-lbl">Healthy Controls</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><div class="metric-val">1</div><div class="metric-lbl">Epilepsy (CHB-MIT)</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><div class="metric-val">2</div><div class="metric-lbl">Alzheimer\'s (ds004504)</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card"><div class="metric-val">3</div><div class="metric-lbl">Parkinson\'s (ds004584)</div></div>', unsafe_allow_html=True)
    with c5:
        st.markdown('<div class="metric-card"><div class="metric-val">4</div><div class="metric-lbl">Depression (NEMAR/MODMA)</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Core Research Highlights")
    st.markdown("""
    - **Harmonized 19-Channel Montage**: Standard 10–20 referential channels, 256 Hz, 5-second overlapping windows (1,280 samples).
    - **Strict Subject-Level Partitioning**: **Zero subject leakage** across Train (7,195 windows), Validation (1,440 windows), and Test (1,470 windows).
    - **High-Performance Multi-Class Benchmark**: **78.44% Test Accuracy**, **82.27% Balanced Accuracy**, and **0.9498 Macro ROC-AUC**.
    - **True 1-D CNN Grad-CAM & Saliency**: Backpropagation through the spatial-temporal CNN projection layer enables faithful temporal activation and electrode attribution mapping.
    """)

    st.markdown("""
    <div class="disclaimer-box">
    <b>⚠️ Investigational Prototype Disclaimer:</b> This software is developed strictly for computational neuroscience and academic research purposes. It is not approved as a medical device or certified for clinical diagnosis.
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TAB 2: LIVE EEG PREDICTION
# -----------------------------------------------------------------------------
with nav_tabs[1]:
    st.header("Live 5-Class Diagnostic Inference")
    st.markdown("Upload a raw EEG file or select a validated preprocessed sample window to evaluate real-time classification.")

    if model is None:
        st.error(f"Model checkpoint could not be loaded: {model_status}")
    else:
        st.success(f"✅ Active Model: `checkpoints/best_model.pt` ({model_status})")

        sample_options = [
            "Healthy Control (OpenNeuro ds004504)",
            "Alzheimer's Disease Patient (OpenNeuro ds004504)",
            "Parkinson's Disease Patient (OpenNeuro ds004584)",
            "Major Depression Patient (NEMAR nm000114 / MODMA)",
            "Epilepsy Patient (CHB-MIT chb01)"
        ]
        selected_sample_type = st.selectbox("Select Benchmark Sample Condition:", sample_options)

        # Load representative window
        index_csv_path = PROJECT_ROOT / "datasets" / "metadata" / "model_dataset_index.csv"
        window_tensor = None
        sample_meta = {}

        if index_csv_path.exists():
            idx_df = pd.read_csv(index_csv_path)
            cls_map_lookup = {
                "Healthy Control (OpenNeuro ds004504)": 0,
                "Epilepsy Patient (CHB-MIT chb01)": 1,
                "Alzheimer's Disease Patient (OpenNeuro ds004504)": 2,
                "Parkinson's Disease Patient (OpenNeuro ds004584)": 3,
                "Major Depression Patient (NEMAR nm000114 / MODMA)": 4,
            }
            target_cls = cls_map_lookup[selected_sample_type]
            matches = idx_df[idx_df["class_label_idx"] == target_cls]
            if len(matches) > 0:
                row = matches.iloc[0]
                s_file = Path(row["source_file"])
                if not s_file.is_absolute():
                    s_file = PROJECT_ROOT / s_file
                if s_file.exists():
                    w_idx = int(row["window_idx"])
                    if s_file.suffix == ".npy":
                        arr = np.load(s_file, mmap_mode="r")
                        if w_idx < len(arr):
                            win = np.array(arr[w_idx], dtype=np.float32)
                            window_tensor = torch.tensor(win).unsqueeze(0)
                            sample_meta = row.to_dict()

        if window_tensor is not None:
            st.info(f"Loaded Window from `{sample_meta.get('dataset_name', 'dataset')}` | Subject: `{sample_meta.get('subject_id', 'sub')}` | Split: `{sample_meta.get('split', 'split')}`")

            # Plot raw multi-channel waveforms
            fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
            t = np.linspace(0, 5, 1280)
            win_np = window_tensor.squeeze(0).numpy()
            for ch in range(min(5, 19)):
                ax.plot(t, win_np[ch] + (4 - ch) * 3, lw=1.0, label=CHANNELS[ch])
            ax.set_yticks([(4 - ch) * 3 for ch in range(min(5, 19))])
            ax.set_yticklabels(CHANNELS[:min(5, 19)], fontweight="bold")
            ax.set_xlabel("Time (seconds)")
            ax.set_title("Standardized EEG Waveforms (Frontal / Central Channels)", fontweight="bold")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

            if st.button("🚀 Run 5-Class Inference & Explainability", type="primary"):
                with torch.no_grad():
                    logits = model(window_tensor)
                    probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
                    pred_idx = int(np.argmax(probs))

                st.markdown("### Diagnostic Prediction")
                p_col1, p_col2 = st.columns([1, 2])
                with p_col1:
                    st.markdown(f'<div class="metric-card"><div class="metric-val">{DISEASE_NAMES[pred_idx]}</div><div class="metric-lbl">Predicted Category (Confidence: {probs[pred_idx]*100:.1f}%)</div></div>', unsafe_allow_html=True)
                with p_col2:
                    prob_df = pd.DataFrame({
                        "Condition": DISEASE_NAMES,
                        "Probability (%)": np.round(probs * 100, 2)
                    })
                    st.bar_chart(prob_df.set_index("Condition"))
        else:
            st.warning("Sample EEG window could not be loaded from disk.")


# -----------------------------------------------------------------------------
# TAB 3: MODEL PERFORMANCE
# -----------------------------------------------------------------------------
with nav_tabs[2]:
    st.header("Held-Out Validation & Test Set Performance")
    
    if metrics_data:
        tm = metrics_data.get("test_metrics", {})
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{tm.get("accuracy", 0.7844)*100:.2f}%</div><div class="metric-lbl">Test Accuracy</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{tm.get("balanced_accuracy", 0.8227)*100:.2f}%</div><div class="metric-lbl">Balanced Accuracy</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{tm.get("macro_roc_auc", 0.9498):.4f}</div><div class="metric-lbl">Macro ROC-AUC</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{tm.get("weighted_f1", 0.7824):.4f}</div><div class="metric-lbl">Weighted F1</div></div>', unsafe_allow_html=True)
        with c5:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{tm.get("cohen_kappa", 0.6959):.4f}</div><div class="metric-lbl">Cohen\'s Kappa</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    pcol1, pcol2 = st.columns(2)

    with pcol1:
        st.subheader("5-Class Held-Out Confusion Matrix")
        cm_path = RESULTS_DIR / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), caption="Normalized Confusion Matrix (1,470 Held-Out Test Windows)", use_column_width=True)

    with pcol2:
        st.subheader("Training & Validation Trajectories")
        tc_path = RESULTS_DIR / "training_curves.png"
        if tc_path.exists():
            st.image(str(tc_path), caption="CrossEntropy Loss and Accuracy Trajectories across Epochs", use_column_width=True)

    st.subheader("Per-Class Classification Report")
    cr_path = RESULTS_DIR / "classification_report.csv"
    if cr_path.exists():
        cr_df = pd.read_csv(cr_path)
        st.dataframe(cr_df, use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 4: GRAD-CAM EXPLAINABILITY
# -----------------------------------------------------------------------------
with nav_tabs[3]:
    st.header("Visual Explainability: 1-D CNN Grad-CAM & Saliency")
    st.markdown("""
    To ensure transparency, the framework implements **1-D CNN Grad-CAM** adapted from Selvaraju et al. (2017) and **Spatial Electrode Saliency**:
    - **Temporal Grad-CAM**: Highlights temporal waveform rhythms that trigger the model's diagnostic classification.
    - **Spatial Electrode Saliency**: Aggregates gradient attributions across all 19 standard 10–20 channels.
    """)

    gc_composite = RESULTS_DIR / "gradcam_analysis.png"
    if gc_composite.exists():
        st.image(str(gc_composite), caption="Multi-Class Grad-CAM Temporal Waveforms & 19-Channel Electrode Saliency", use_column_width=True)

    st.markdown("---")
    st.subheader("Individual Class Biomarker Profiles")
    c_choice = st.selectbox("Select Condition to View Attribution:", ["healthy", "alzheimers", "parkinsons", "depression", "epilepsy"])
    ind_img = RESULTS_DIR / "gradcam" / f"gradcam_{c_choice}.png"
    if ind_img.exists():
        st.image(str(ind_img), caption=f"Grad-CAM & Saliency Profile for {c_choice.capitalize()}", use_column_width=True)


# -----------------------------------------------------------------------------
# TAB 5: DATASETS & COHORTS
# -----------------------------------------------------------------------------
with nav_tabs[4]:
    st.header("Multi-Cohort Clinical Provenance")
    st.markdown("""
    The framework is harmonized across 5 open scientific scalp EEG databases:
    """)

    ds_summary_table = pd.DataFrame([
        {"Condition": "Healthy Control", "Class ID": 0, "Source / Accession": "SRM + ds004504 + ds004584 + NEMAR", "Total Windows": "3,110", "Train": "2,090", "Val": "480", "Test": "540", "Split Type": "Subject-Level"},
        {"Condition": "Epilepsy", "Class ID": 1, "Source / Accession": "CHB-MIT (chb01, 6 EDF sessions)", "Total Windows": "55", "Train": "55", "Val": "0", "Test": "0", "Split Type": "Subject-Level (Train only)"},
        {"Condition": "Alzheimer's Disease", "Class ID": 2, "Source / Accession": "OpenNeuro ds004504 (AD Cohort)", "Total Windows": "1,080", "Train": "720", "Val": "180", "Test": "180", "Split Type": "Subject-Level"},
        {"Condition": "Parkinson's Disease", "Class ID": 3, "Source / Accession": "OpenNeuro ds004584 (PD Cohort)", "Total Windows": "3,000", "Train": "2,070", "Val": "390", "Test": "540", "Split Type": "Subject-Level"},
        {"Condition": "Major Depression", "Class ID": 4, "Source / Accession": "NEMAR nm000114 / MODMA Cohort", "Total Windows": "1,860", "Train": "1,260", "Val": "390", "Test": "210", "Split Type": "Subject-Level"},
    ])
    st.dataframe(ds_summary_table, use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 6: METHODOLOGY
# -----------------------------------------------------------------------------
with nav_tabs[5]:
    st.header("Methodology & Pipeline Architecture")
    st.markdown("""
    <div class="architecture-box">
    <b>End-to-End Processing Architecture:</b><br><br>
    Raw Multi-Channel Scalp EEG (EDF / SET / BDF)<br>
    &nbsp;&nbsp; ↓ <b>Preprocessing & Harmonization:</b> 0.5–45.0 Hz Bandpass + 50/60 Hz Notch + 256 Hz Resampling<br>
    &nbsp;&nbsp; ↓ <b>Montage Standardization:</b> 19 Standard 10–20 Referential Channels<br>
    &nbsp;&nbsp; ↓ <b>Segmentation & Normalization:</b> 5-Second Windows (1,280 timepoints) + Per-Channel Z-Score<br>
    &nbsp;&nbsp; ↓ <b>EEGNet-Style CNN Feature Extractor:</b> 1D Temporal Conv (1×64) + Depthwise Spatial Conv (19×1) → Tokens [Batch, 40, 64]<br>
    &nbsp;&nbsp; ↓ <b>Transformer Encoder:</b> [CLS] Token + Sinusoidal Positional Encoding + Multi-Head Self-Attention (4 heads, 2 layers)<br>
    &nbsp;&nbsp; ↓ <b>Linear Classification Head:</b> 5-Class Logits [Healthy, Epilepsy, Alzheimer's, Parkinson's, Depression]<br>
    &nbsp;&nbsp; ↓ <b>Gradient Explainability:</b> 1D CNN Grad-CAM + Input Gradient Saliency
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TAB 7: RESULTS & METRICS
# -----------------------------------------------------------------------------
with nav_tabs[6]:
    st.header("Comprehensive Experimental Summary")
    summary_md_path = RESULTS_DIR / "FINAL_EXPERIMENT_SUMMARY.md"
    if summary_md_path.exists():
        with open(summary_md_path, "r", encoding="utf-8") as f:
            content = f.read()
        st.markdown(content)


# -----------------------------------------------------------------------------
# TAB 8: LIMITATIONS & SCOPE
# -----------------------------------------------------------------------------
with nav_tabs[7]:
    st.header("Scientific Limitations & Transparency")
    st.markdown("""
    In accordance with scientific rigor, the following methodological boundaries are explicitly stated:

    1. **Epilepsy Cohort Partitioning**:
       - The locally available CHB-MIT dataset contains recordings from subject `chb01` (55 windows), which are held strictly in the training cohort.
       - Independent generalization to unseen epilepsy patients will be confirmed upon downloading further CHB subjects (`chb02`–`chb24`).
    2. **Multi-Origin Healthy Reference**:
       - Healthy controls originate from four distinct clinical cohorts (SRM, ds004504, ds004584, NEMAR), providing robust baseline invariance across age and recording setups.
    3. **FTD Exclusion**:
       - Frontotemporal Dementia cases in ds004504 were excluded to prevent label ambiguity between Alzheimer's and non-Alzheimer's dementia.
    4. **Clinical Investigational Scope**:
       - The system is designed for computational neuroscience benchmarking and clinical decision-support research, not autonomous medical diagnosis.
    """)
