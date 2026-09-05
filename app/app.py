"""
Transformer-Based Deep Learning Framework for Robust EEG Signal Classification
Faculty Seminar Prototype Demonstration UI

Run with:
    streamlit run app/app.py
"""

import os
import sys
import tempfile
import glob
import re
import numpy as np
import pandas as pd
import torch
import mne
import matplotlib.pyplot as plt
import streamlit as st

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from preprocessing.config import (
    TARGET_SRATE, WINDOW_SEC, OVERLAP, N_CHANNELS, WINDOW_SAMPLES,
    COMMON_CHANNELS, DATASET_PATHS, RAW_DIR
)
from preprocessing.common import (
    bandpass_filter, notch_filter, resample, select_and_reorder_channels, normalize_windows
)
from preprocessing.harmonize import harmonize_raw
from preprocessing.segment import segment_continuous, segment_labeled
from preprocessing.datasets.chbmit import (
    parse_summary, select_chbmit_channels, CHBMIT_CHANNELS
)
from models.eeg_cnn import EEGFeatureExtractor
from models.transformer import EEGTransformerEncoder
from models.eeg_classifier import EEGClassifier

# Page configuration
st.set_page_config(
    page_title="EEG Classification Prototype",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for academic, clean styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        font-weight: 500;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .status-badge {
        background-color: #E2E8F0;
        color: #0F172A;
        padding: 0.35rem 0.75rem;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 1rem;
    }
    .section-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1.5rem;
    }
    .architecture-box {
        background-color: #F1F5F9;
        border-left: 4px solid #2563EB;
        padding: 1rem;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 6px;
        padding: 0.75rem;
        text-align: center;
    }
    .metric-val {
        font-size: 1.4rem;
        font-weight: 700;
        color: #2563EB;
    }
    .metric-lbl {
        font-size: 0.8rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state variables
if 'raw_eeg' not in st.session_state:
    st.session_state['raw_eeg'] = None
if 'file_info' not in st.session_state:
    st.session_state['file_info'] = None
if 'processed_data' not in st.session_state:
    st.session_state['processed_data'] = None
if 'ch_names' not in st.session_state:
    st.session_state['ch_names'] = None
if 'windows' not in st.session_state:
    st.session_state['windows'] = None
if 'window_metadata' not in st.session_state:
    st.session_state['window_metadata'] = None
if 'cnn_features' not in st.session_state:
    st.session_state['cnn_features'] = None
if 'transformer_logits' not in st.session_state:
    st.session_state['transformer_logits'] = None


# Helper functions
def load_sample_edf(edf_path):
    """Load sample EDF using MNE."""
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        return raw
    except Exception as e:
        st.error(f"Error loading EDF file: {e}")
        return None

def plot_eeg_signals(data, ch_names, srate, n_channels_to_plot=5, title="EEG Signal Waveforms"):
    """Plot multi-channel EEG signals using Matplotlib."""
    fig, ax = plt.subplots(figsize=(10, 1.2 * n_channels_to_plot), dpi=100)
    
    n_channels = min(n_channels_to_plot, data.shape[0])
    n_samples = data.shape[1]
    time = np.arange(n_samples) / srate

    # Calculate scale offset for stacking channels cleanly
    scale = np.std(data[:n_channels, :]) * 3
    if scale == 0 or np.isnan(scale):
        scale = 1.0

    for i in range(n_channels):
        channel_data = data[i, :]
        offset = (n_channels - 1 - i) * scale
        ax.plot(time, channel_data + offset, label=ch_names[i], linewidth=0.8, color='#1E40AF')

    ax.set_yticks([(n_channels - 1 - i) * scale for i in range(n_channels)])
    ax.set_yticklabels(ch_names[:n_channels], fontsize=9, fontweight='bold')
    ax.set_xlabel("Time (seconds)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/brain.png", width=60)
st.sidebar.title("Pipeline Controls")

dataset_choice = st.sidebar.selectbox(
    "Select Dataset",
    [
        "CHB-MIT (Epilepsy / Seizure EEG)",
        "Alzheimer's Disease (ds004504)",
        "Parkinson's Disease (ds004584)",
        "MODMA (Depression)",
        "TUAB (Abnormal/Normal EEG)"
    ],
    index=0
)

# Status badges per dataset
dataset_status_map = {
    "CHB-MIT (Epilepsy / Seizure EEG)": "✅ Fully Supported (Local Raw Data Available)",
    "Alzheimer's Disease (ds004504)": "✅ Supported (BIDS .set Loader)",
    "Parkinson's Disease (ds004584)": "✅ Supported (BIDS .set Loader)",
    "MODMA (Depression)": "⚠️ Blocked (Multi-part 7z Archive Compressed)",
    "TUAB (Abnormal/Normal EEG)": "⚠️ Restricted (Institutional Access Required)"
}

st.sidebar.caption(f"**Dataset Status:** {dataset_status_map[dataset_choice]}")

# Input Method
input_method = st.sidebar.radio(
    "EEG Input Method",
    ["Select Sample EDF from Dataset", "Upload Custom EDF File"]
)

selected_file_path = None
uploaded_file = None

if input_method == "Select Sample EDF from Dataset":
    if dataset_choice.startswith("CHB-MIT"):
        epilepsy_dir = os.path.join(RAW_DIR, 'epilepsy')
        edf_files = sorted(glob.glob(os.path.join(epilepsy_dir, '*.edf')))
        if edf_files:
            file_names = [os.path.basename(f) for f in edf_files]
            selected_file_name = st.sidebar.selectbox("Choose CHB-MIT File", file_names)
            selected_file_path = os.path.join(epilepsy_dir, selected_file_name)
        else:
            st.sidebar.warning("No sample EDF files found in `datasets/raw/epilepsy`")
    elif dataset_choice.startswith("Alzheimer"):
        alz_dir = os.path.join(RAW_DIR, 'alzheimers')
        set_files = sorted(glob.glob(os.path.join(alz_dir, '**', '*.set'), recursive=True))
        if set_files:
            file_names = [os.path.basename(f) for f in set_files]
            selected_file_name = st.sidebar.selectbox("Choose Alzheimer File", file_names)
            selected_file_path = set_files[file_names.index(selected_file_name)]
        else:
            st.sidebar.warning("No sample .set files found in `datasets/raw/alzheimers`")
    elif dataset_choice.startswith("Parkinson"):
        pd_dir = os.path.join(RAW_DIR, 'parkinsons')
        set_files = sorted(glob.glob(os.path.join(pd_dir, '**', '*.set'), recursive=True))
        if set_files:
            file_names = [os.path.basename(f) for f in set_files]
            selected_file_name = st.sidebar.selectbox("Choose Parkinson File", file_names)
            selected_file_path = set_files[file_names.index(selected_file_name)]
        else:
            st.sidebar.warning("No sample .set files found in `datasets/raw/parkinsons`")
    else:
        st.sidebar.info("Dataset files are compressed or restricted. Upload custom EDF file instead.")

else:
    uploaded_file = st.sidebar.file_uploader("Upload EEG EDF File", type=["edf"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            selected_file_path = tmp_file.name

# Visualization settings
n_plot_channels = st.sidebar.slider("Channels to Display in Waveform Plot", min_value=1, max_value=19, value=5)


# -----------------------------------------------------------------------------
# MAIN APP HEADER
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">Transformer-Based EEG Classification Framework</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Unified Multi-Disease Architecture: CNN Feature Extraction + Transformer Self-Attention</div>', unsafe_allow_html=True)
st.markdown('<div class="status-badge">RESEARCH PROTOTYPE DEMONSTRATION — FACULTY SEMINAR EDITION</div>', unsafe_allow_html=True)

st.markdown("---")

# Auto-load file when selected
if selected_file_path and (st.session_state['file_info'] is None or st.session_state['file_info'].get('path') != selected_file_path):
    if selected_file_path.endswith('.set'):
        raw = mne.io.read_raw_eeglab(selected_file_path, preload=True, verbose=False)
    else:
        raw = load_sample_edf(selected_file_path)

    if raw is not None:
        st.session_state['raw_eeg'] = raw
        st.session_state['file_info'] = {
            'name': os.path.basename(selected_file_path),
            'path': selected_file_path,
            'n_channels': len(raw.ch_names),
            'srate': float(raw.info['sfreq']),
            'duration': float(raw.times[-1]),
            'n_samples': int(len(raw.times))
        }
        # Reset downstream stages
        st.session_state['processed_data'] = None
        st.session_state['windows'] = None
        st.session_state['cnn_features'] = None
        st.session_state['transformer_logits'] = None


# -----------------------------------------------------------------------------
# SECTION 1 — DATASET / EEG INPUT INFORMATION
# -----------------------------------------------------------------------------
st.header("1. Dataset & EEG File Information")

if st.session_state['raw_eeg'] is not None:
    info = st.session_state['file_info']

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{info["name"]}</div><div class="metric-lbl">File Name</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{info["n_channels"]}</div><div class="metric-lbl">Channels</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{info["srate"]:.1f} Hz</div><div class="metric-lbl">Sampling Rate</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{info["duration"]:.1f} s</div><div class="metric-lbl">Duration</div></div>', unsafe_allow_html=True)
    with col5:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{info["n_samples"]:,}</div><div class="metric-lbl">Total Samples</div></div>', unsafe_allow_html=True)

    # Check for seizure annotations in CHB-MIT summary file
    if "CHB-MIT" in dataset_choice and info["name"].startswith("chb"):
        epilepsy_dir = os.path.dirname(info['path'])
        subj_prefix = info["name"].split("_")[0]
        summary_file = os.path.join(epilepsy_dir, f"{subj_prefix}-summary.txt")
        if os.path.exists(summary_file):
            seizures_dict = parse_summary(summary_file)
            file_seizures = seizures_dict.get(info["name"], [])
            if file_seizures:
                st.warning(f"⚡ **Seizure Annotations Found in {info['name']}:** {len(file_seizures)} Seizure Event(s)")
                for idx, (s_start, s_end) in enumerate(file_seizures, 1):
                    st.write(f"- **Event {idx}:** Seizure Start = `{s_start}s`, Seizure End = `{s_end}s` (Duration: `{s_end - s_start}s`)")
            else:
                st.info(f"ℹ️ **Annotations:** No seizure events recorded in `{info['name']}` (Non-seizure recording segment).")
else:
    st.info("👈 Please select or upload an EEG EDF file from the sidebar to begin analysis.")

st.markdown("---")

# -----------------------------------------------------------------------------
# SECTION 2 — RAW EEG VISUALIZATION
# -----------------------------------------------------------------------------
st.header("2. Raw EEG Signal Visualization")

if st.session_state['raw_eeg'] is not None:
    raw = st.session_state['raw_eeg']
    # Extract first 5 seconds or full duration for display
    plot_sec = min(10.0, raw.times[-1])
    sample_limit = int(plot_sec * raw.info['sfreq'])
    raw_data = raw.get_data()[:, :sample_limit]
    
    fig = plot_eeg_signals(
        raw_data,
        raw.ch_names,
        raw.info['sfreq'],
        n_channels_to_plot=n_plot_channels,
        title=f"Raw EEG Waveform — First {plot_sec:.1f} Seconds ({info['name']})"
    )
    st.pyplot(fig)
    plt.close(fig)
else:
    st.warning("No raw EEG data loaded yet.")

st.markdown("---")

# -----------------------------------------------------------------------------
# SECTION 3 — PREPROCESSING & HARMONIZATION
# -----------------------------------------------------------------------------
st.header("3. Preprocessing & Harmonization")

st.markdown("""
Preprocessing harmonizes heterogenous EEG datasets into a unified format:
- **Resampling:** Standardized to **256 Hz** target sampling rate
- **Bandpass Filtering:** FIR filter between **0.5 Hz – 45.0 Hz** to isolate physiological brain rhythms
- **Notch Filtering:** Eliminates powerline interference at **50 Hz / 60 Hz**
- **Montage Selection:** Standard 19-channel 10-20 system / bipolar montage harmonization
""")

col_prep_btn, col_prep_status = st.columns([1, 3])

with col_prep_btn:
    btn_preprocess = st.button("⚡ Preprocess EEG Signal", use_container_width=True)

if btn_preprocess and st.session_state['raw_eeg'] is not None:
    raw_copy = st.session_state['raw_eeg'].copy()
    
    # Run harmonization pipeline
    with st.spinner("Applying filtering, resampling, and channel harmonization..."):
        if "CHB-MIT" in dataset_choice or raw_copy.ch_names[0].startswith("FP1-"):
            raw_copy.load_data()
            notch_filter(raw_copy)
            bandpass_filter(raw_copy)
            resample(raw_copy)
            data, ch_found = select_chbmit_channels(raw_copy)
            ch_names = CHBMIT_CHANNELS[:N_CHANNELS]
        else:
            harmonize_raw(raw_copy, dataset_name=dataset_choice)
            data, ch_found = select_and_reorder_channels(raw_copy)
            ch_names = COMMON_CHANNELS

        st.session_state['processed_data'] = data
        st.session_state['ch_names'] = ch_names

with col_prep_status:
    if st.session_state['processed_data'] is not None:
        st.success("✅ **Preprocessing Pipeline Completed Successfully!**")
        st.markdown("""
        - [x] **EEG Loaded:** Raw signal loaded from EDF reader
        - [x] **Band-pass Filter:** Applied 0.5 – 45.0 Hz FIR filter
        - [x] **Notch Filter:** Removed 50/60 Hz powerline line noise
        - [x] **Channel Selection:** Harmonized to 19 standardized channels
        - [x] **Resampled:** Target sampling rate set to 256 Hz
        """)

if st.session_state['processed_data'] is not None:
    proc_data = st.session_state['processed_data']
    ch_names = st.session_state['ch_names']
    srate = TARGET_SRATE

    plot_sec = min(10.0, proc_data.shape[1] / srate)
    sample_limit = int(plot_sec * srate)

    fig_proc = plot_eeg_signals(
        proc_data[:, :sample_limit],
        ch_names,
        srate,
        n_channels_to_plot=n_plot_channels,
        title=f"Harmonized & Preprocessed EEG Waveform (256 Hz, 0.5-45 Hz Filtered)"
    )
    st.pyplot(fig_proc)
    plt.close(fig_proc)

st.markdown("---")

# -----------------------------------------------------------------------------
# SECTION 4 — SEGMENTATION & WINDOWING
# -----------------------------------------------------------------------------
st.header("4. Segmentation & Fixed-Size Windowing")

st.markdown("""
Continuous EEG recordings are sliced into fixed-duration overlapping segments:
- **Window Duration:** `5.0 seconds`
- **Overlap:** `50%` (2.5 second step)
- **Target Shape per Window:** `[19 channels × 1280 time samples]` (at 256 Hz)
""")

col_seg_btn, col_seg_status = st.columns([1, 3])

with col_seg_btn:
    btn_segment = st.button("🔪 Segment EEG into Windows", use_container_width=True)

if btn_segment and st.session_state['processed_data'] is not None:
    proc_data = st.session_state['processed_data']
    
    with st.spinner("Segmenting continuous EEG into 5-second overlapping windows..."):
        # Check if seizure annotations exist for this file
        info = st.session_state['file_info']
        file_seizures = []
        if "CHB-MIT" in dataset_choice and info["name"].startswith("chb"):
            epilepsy_dir = os.path.dirname(info['path'])
            subj_prefix = info["name"].split("_")[0]
            summary_file = os.path.join(epilepsy_dir, f"{subj_prefix}-summary.txt")
            if os.path.exists(summary_file):
                seizures_dict = parse_summary(summary_file)
                file_seizures = seizures_dict.get(info["name"], [])

        if file_seizures:
            # Segment labeled seizure & non-seizure windows
            s_wins, s_ranges = segment_labeled(proc_data, file_seizures, 'seizure', srate=TARGET_SRATE)
            
            # Non-seizure intervals
            total_duration = proc_data.shape[1] / TARGET_SRATE
            non_s_intervals = []
            prev_end = 0
            for ss, se in sorted(file_seizures):
                if ss > prev_end:
                    non_s_intervals.append((prev_end, ss))
                prev_end = se
            if prev_end < total_duration:
                non_s_intervals.append((prev_end, total_duration))
                
            ns_wins, ns_ranges = segment_labeled(proc_data, non_s_intervals, 'non_seizure', srate=TARGET_SRATE)
            
            if len(s_wins) > 0 and len(ns_wins) > 0:
                windows = np.concatenate([s_wins, ns_wins], axis=0)
            elif len(s_wins) > 0:
                windows = s_wins
            else:
                windows = ns_wins
                
            windows = normalize_windows(windows)
            st.session_state['windows'] = windows
            st.session_state['window_metadata'] = {
                'seizure_count': len(s_wins),
                'non_seizure_count': len(ns_wins),
                'total_count': len(windows)
            }
        else:
            windows, time_ranges = segment_continuous(proc_data, window_sec=WINDOW_SEC, overlap=OVERLAP, srate=TARGET_SRATE)
            if len(windows) > 0:
                windows = normalize_windows(windows)
            st.session_state['windows'] = windows
            st.session_state['window_metadata'] = {
                'total_count': len(windows),
                'seizure_count': 0,
                'non_seizure_count': len(windows)
            }

with col_seg_status:
    if st.session_state['windows'] is not None:
        wins = st.session_state['windows']
        meta = st.session_state['window_metadata']
        
        st.success(f"✅ **Generated {len(wins)} Harmonized EEG Windows!**")
        
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{wins.shape[0]}</div><div class="metric-lbl">Total Windows</div></div>', unsafe_allow_html=True)
        with mcol2:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{wins.shape[1]} × {wins.shape[2]}</div><div class="metric-lbl">Window Shape [C × T]</div></div>', unsafe_allow_html=True)
        with mcol3:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{meta.get("seizure_count", 0)}</div><div class="metric-lbl">Seizure Windows</div></div>', unsafe_allow_html=True)
        with mcol4:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{meta.get("non_seizure_count", 0)}</div><div class="metric-lbl">Non-Seizure Windows</div></div>', unsafe_allow_html=True)

st.markdown("---")

# -----------------------------------------------------------------------------
# SECTION 5 — CNN FEATURE EXTRACTION
# -----------------------------------------------------------------------------
st.header("5. CNN / EEGNet Feature Extraction")

st.markdown("""
An EEGNet-inspired 2D/1D CNN extracts spatio-temporal features from raw 5-second EEG windows:
1. **Temporal Convolution:** 1D Conv across time `(1 × 64)` to capture frequency rhythms
2. **Depthwise Spatial Convolution:** Conv across channels `(19 × 1)` to model inter-electrode spatial patterns
3. **Separable Convolution:** Depthwise-separable block `(1 × 16)` for feature compression
4. **Feature Sequence Projection:** Reshapes feature maps into Transformer token sequence `[B, N, D]`
""")

col_cnn_btn, col_cnn_status = st.columns([1, 3])

with col_cnn_btn:
    btn_cnn = st.button("🧠 Run CNN Feature Extractor", use_container_width=True)

if btn_cnn and st.session_state['windows'] is not None and len(st.session_state['windows']) > 0:
    windows = st.session_state['windows']
    
    # Take first window [1, 19, 1280] for demonstration
    single_win = torch.tensor(windows[0:1], dtype=torch.float32)  # [1, 19, 1280]
    
    cnn_model = EEGFeatureExtractor(
        n_channels=single_win.shape[1],
        n_samples=single_win.shape[2],
        embed_dim=64
    )
    cnn_model.eval()
    
    with torch.no_grad():
        feature_tokens = cnn_model.extract_features(single_win)  # [1, 40, 64]
        st.session_state['cnn_features'] = feature_tokens.numpy()

with col_cnn_status:
    if st.session_state['cnn_features'] is not None:
        feats = st.session_state['cnn_features']
        st.success("✅ **CNN Feature Extraction Forward Pass Complete!**")
        
        st.markdown(f"""
        <div class="architecture-box">
        <b>CNN Pipeline Dimension Transformation:</b><br>
        Input EEG Window: &nbsp;&nbsp;&nbsp; <b>[1, 19, 1280]</b> (Batch=1, Channels=19, Samples=1280)<br>
        &nbsp;&nbsp; ↓ (Temporal Conv + Depthwise Spatial Conv)<br>
        &nbsp;&nbsp; ↓ (Separable Conv + Pooling)<br>
        Feature Tokens Output: <b>[{feats.shape[0]}, {feats.shape[1]}, {feats.shape[2]}]</b> (Batch=1, Tokens N={feats.shape[1]}, Embedding Dim D={feats.shape[2]})
        </div>
        """, unsafe_allow_html=True)
        st.caption("ℹ️ *CNN Architecture Forward-Pass Verified — Model initialized with PyTorch architecture weights.*")

st.markdown("---")

# -----------------------------------------------------------------------------
# SECTION 6 — TRANSFORMER ENCODER & SELF-ATTENTION
# -----------------------------------------------------------------------------
st.header("6. Transformer Encoder & Self-Attention")

st.markdown("""
The Transformer Encoder processes temporal feature tokens from the CNN:
1. **CLS Token Prepending:** Adds learnable class token `[B, 1, D]` → `[B, N+1, D]`
2. **Sinusoidal Positional Encoding:** Adds positional context across temporal sequence
3. **Multi-Head Self-Attention:** 2 Transformer Encoder Layers (4 attention heads) capture long-range temporal dependencies
4. **GELU MLP & Logit Output:** Maps pooled CLS token representation to classification logits `[B, num_classes]`
""")

col_trans_btn, col_trans_status = st.columns([1, 3])

with col_trans_btn:
    btn_trans = st.button("⚡ Run Transformer Encoder", use_container_width=True)

if btn_trans and st.session_state['cnn_features'] is not None:
    feats = torch.tensor(st.session_state['cnn_features'], dtype=torch.float32)  # [1, 40, 64]
    
    transformer_model = EEGTransformerEncoder(
        embed_dim=feats.shape[2],
        num_heads=4,
        num_layers=2,
        num_classes=2
    )
    transformer_model.eval()
    
    with torch.no_grad():
        logits = transformer_model(feats)  # [1, 2]
        st.session_state['transformer_logits'] = logits.numpy()

with col_trans_status:
    if st.session_state['transformer_logits'] is not None:
        logits = st.session_state['transformer_logits']
        st.success("✅ **Transformer Self-Attention Forward Pass Complete!**")
        
        st.markdown(f"""
        <div class="architecture-box">
        <b>Transformer Pipeline Dimension Transformation:</b><br>
        CNN Feature Tokens Input: <b>[1, 40, 64]</b><br>
        &nbsp;&nbsp; ↓ (Prepend CLS Token: [1, 41, 64])<br>
        &nbsp;&nbsp; ↓ (Sinusoidal Positional Encoding)<br>
        &nbsp;&nbsp; ↓ (2 × Multi-Head Self-Attention Encoder Layers)<br>
        &nbsp;&nbsp; ↓ (Linear Classification Head)<br>
        Output Logits Tensor: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>[{logits.shape[0]}, {logits.shape[1]}]</b> (Batch=1, Classes={logits.shape[1]})
        </div>
        """, unsafe_allow_html=True)
        st.caption("ℹ️ *Transformer Forward-Pass Prototype Completed — Architecture successfully accepts CNN feature sequence and computes multi-head self-attention.*")

st.markdown("---")

# -----------------------------------------------------------------------------
# SECTION 7 — CLASSIFICATION STAGE
# -----------------------------------------------------------------------------
st.header("7. Classification Stage")

st.info("""
⚠️ **Architecture Verification Stage:** 
The CNN + Transformer forward-pass tensor pipeline has been fully integrated and verified in PyTorch `(EEG -> CNN -> Transformer -> Logits)`. 
Full multi-epoch backpropagation training and evaluation benchmarking on full multi-gigabyte datasets is scheduled for the next development phase.
""")

st.markdown("---")

# -----------------------------------------------------------------------------
# SECTION 8 — UNIFIED MULTI-DISEASE ARCHITECTURE
# -----------------------------------------------------------------------------
st.header("8. Unified Multi-Disease Framework Overview")

st.markdown("""
<div class="section-card">
<h4>Unified EEG Multi-Disease Classification Pipeline</h4>
<p>This framework standardizes heterogeneous EEG datasets across multiple neurological disorders into a shared spatial-temporal representation.</p>
</div>
""", unsafe_allow_html=True)

col_d1, col_d2, col_d3, col_d4, col_d5 = st.columns(5)

with col_d1:
    st.markdown("""
    <div class="metric-card">
        <b>CHB-MIT</b><br>
        <span style="color:#2563EB; font-size:0.85rem;">Epilepsy Seizures</span><br>
        <small>EDF Raw / Bipolar</small><br>
        <span style="color:green; font-weight:bold;">Available</span>
    </div>
    """, unsafe_allow_html=True)

with col_d2:
    st.markdown("""
    <div class="metric-card">
        <b>Alzheimer's</b><br>
        <span style="color:#2563EB; font-size:0.85rem;">ds004504 AD/HC</span><br>
        <small>BIDS .set Loader</small><br>
        <span style="color:green; font-weight:bold;">Available</span>
    </div>
    """, unsafe_allow_html=True)

with col_d3:
    st.markdown("""
    <div class="metric-card">
        <b>Parkinson's</b><br>
        <span style="color:#2563EB; font-size:0.85rem;">ds004584 PD/HC</span><br>
        <small>BIDS .set Loader</small><br>
        <span style="color:green; font-weight:bold;">Available</span>
    </div>
    """, unsafe_allow_html=True)

with col_d4:
    st.markdown("""
    <div class="metric-card">
        <b>MODMA</b><br>
        <span style="color:#2563EB; font-size:0.85rem;">Depression</span><br>
        <small>Split 7z Archive</small><br>
        <span style="color:orange; font-weight:bold;">Compressed</span>
    </div>
    """, unsafe_allow_html=True)

with col_d5:
    st.markdown("""
    <div class="metric-card">
        <b>TUAB</b><br>
        <span style="color:#2563EB; font-size:0.85rem;">Abnormal / Normal</span><br>
        <small>TUH EEG Corpus</small><br>
        <span style="color:red; font-weight:bold;">Restricted</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# -----------------------------------------------------------------------------
# SECTION 9 — EXPLAINABILITY & INTERPRETABILITY
# -----------------------------------------------------------------------------
st.header("9. Explainability & Interpretability (Grad-CAM)")

st.markdown("""
<div class="section-card">
<h4>🔮 Grad-CAM Explainability Module — Planned Development Stage</h4>
<p>Future stage will project Transformer self-attention gradients back onto raw EEG channel waveforms to highlight spatio-temporal biomarkers associated with neurological disorders.</p>
</div>
""", unsafe_allow_html=True)

st.caption("Developed for BTech Final Year Project — Unified EEG Multi-Disease Classification")
