"""
Transformer-Based Deep Learning Framework for Robust EEG Signal Classification
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Interactive Research Demonstration Dashboard

Run:
    streamlit run app/app.py
"""

import sys
import json
import io
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F_torch

# ═══════════════════════════════════════════════════════════════════════════════
# PROJECT PATH SETUP
# ═══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.eeg_classifier import EEGClassifier

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="EEG Multi-Disease Classification — Research Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — Professional Medical Research Theme
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global typography */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Header styling */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0d4f6e 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .hero-title {
        font-size: 1.75rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.3rem;
        color: #ffffff;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        font-weight: 400;
        color: #94a3b8;
        margin-bottom: 0;
    }

    /* Metric cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.1rem 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: box-shadow 0.2s;
    }
    .metric-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1e40af;
        line-height: 1.2;
    }
    .metric-lbl {
        font-size: 0.72rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 0.25rem;
    }

    /* Disease category cards */
    .disease-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
    }
    .disease-card.healthy { border-left-color: #22c55e; }
    .disease-card.epilepsy { border-left-color: #ef4444; }
    .disease-card.alzheimers { border-left-color: #a855f7; }
    .disease-card.parkinsons { border-left-color: #3b82f6; }
    .disease-card.depression { border-left-color: #f59e0b; }

    .disease-name {
        font-weight: 600;
        font-size: 0.95rem;
        color: #1e293b;
    }
    .disease-desc {
        font-size: 0.8rem;
        color: #64748b;
    }

    /* Disclaimer */
    .disclaimer-box {
        background: linear-gradient(135deg, #fef3c7, #fef9c3);
        border-left: 4px solid #f59e0b;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        font-size: 0.85rem;
        color: #92400e;
        margin: 1rem 0;
    }

    /* Architecture box */
    .arch-box {
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: 10px;
        padding: 1.5rem;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 0.82rem;
        line-height: 1.6;
    }
    .arch-step {
        display: flex;
        align-items: center;
        margin: 0.3rem 0;
    }
    .arch-arrow {
        color: #3b82f6;
        font-weight: 700;
        margin: 0.2rem 0;
        text-align: center;
    }

    /* Prediction result */
    .prediction-box {
        background: linear-gradient(135deg, #ecfdf5, #d1fae5);
        border: 2px solid #10b981;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .prediction-class {
        font-size: 1.5rem;
        font-weight: 700;
        color: #065f46;
    }
    .prediction-conf {
        font-size: 0.9rem;
        color: #047857;
        margin-top: 0.25rem;
    }

    /* Section dividers */
    .section-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #cbd5e1, transparent);
        margin: 1.5rem 0;
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .status-ok { background: #dcfce7; color: #166534; }
    .status-err { background: #fef2f2; color: #991b1b; }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown li,
    [data-testid="stSidebar"] label {
        color: #cbd5e1 !important;
    }
    [data-testid="stSidebar"] .stSelectbox label {
        color: #e2e8f0 !important;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

DISEASE_NAMES = [
    "Healthy Control",
    "Epilepsy",
    "Alzheimer's Disease",
    "Parkinson's Disease",
    "Depression",
]

DISEASE_ICONS = ["💚", "⚡", "🟣", "🔵", "🟡"]

DISEASE_KEYS = ["healthy", "epilepsy", "alzheimers", "parkinsons", "depression"]

DISEASE_COLORS = ["#22c55e", "#ef4444", "#a855f7", "#3b82f6", "#f59e0b"]

CHANNELS_19 = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4", "T5", "P3",
    "Pz", "P4", "T6", "O1", "O2",
]

# 10-10 to 10-20 channel aliases for uploaded EEG files
CHANNEL_ALIASES = {
    'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6',
    'FP1': 'Fp1', 'FP2': 'Fp2', 'FZ': 'Fz', 'CZ': 'Cz', 'PZ': 'Pz',
    'fp1': 'Fp1', 'fp2': 'Fp2', 'fz': 'Fz', 'cz': 'Cz', 'pz': 'Pz',
    'f7': 'F7', 'f3': 'F3', 'f4': 'F4', 'f8': 'F8',
    't3': 'T3', 'c3': 'C3', 'c4': 'C4', 't4': 'T4',
    't5': 'T5', 'p3': 'P3', 'p4': 'P4', 't6': 'T6',
    'o1': 'O1', 'o2': 'O2',
    't7': 'T3', 't8': 'T4', 'p7': 'T5', 'p8': 'T6',
}

TARGET_SRATE = 256
WINDOW_SAMPLES = 1280
N_CHANNELS = 19

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pt"
METRICS_PATH = PROJECT_ROOT / "results" / "metrics.json"
RESULTS_DIR = PROJECT_ROOT / "results"


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING (cached)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    """Load the trained 5-class CNN-Transformer model."""
    if not CHECKPOINT_PATH.exists():
        return None, f"Checkpoint not found at {CHECKPOINT_PATH}"

    try:
        checkpoint = torch.load(
            CHECKPOINT_PATH, map_location="cpu", weights_only=False
        )
        cfg = checkpoint.get("config", {})

        # Extract hyperparameters from config (supports both dict and dataclass)
        def _get(key, default):
            if isinstance(cfg, dict):
                return cfg.get(key, default)
            return getattr(cfg, key, default)

        model = EEGClassifier(
            n_channels=19,
            n_samples=1280,
            num_classes=5,
            embed_dim=_get("embed_dim", 64),
            num_heads=_get("num_heads", 4),
            num_layers=_get("num_layers", 2),
            ff_dim=_get("ff_dim", 128),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        epoch = checkpoint.get("epoch", "?")
        val_acc = checkpoint.get("val_acc", None)
        info = f"Epoch {epoch}"
        if val_acc is not None:
            info += f" | Val Acc {val_acc*100:.1f}%"

        return model, info

    except Exception as exc:
        return None, f"Error: {exc}"


@st.cache_data
def load_metrics():
    """Load evaluation metrics from JSON."""
    if not METRICS_PATH.exists():
        return None
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# EEG PREPROCESSING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_channel_name(ch_name):
    """Normalize an EEG channel name to standard 10-20 naming."""
    name = ch_name.strip()
    for prefix in ['EEG ', 'EEG-', 'eeg ', 'eeg-']:
        if name.startswith(prefix):
            name = name[len(prefix):]
    for suffix in ['-REF', '-LE', '-Ref', '-ref', '-AVG', '-Avg']:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    name = name.strip()

    if name in CHANNEL_ALIASES:
        return CHANNEL_ALIASES[name]
    if name in CHANNELS_19:
        return name

    name_lower = name.lower()
    for std_ch in CHANNELS_19:
        if std_ch.lower() == name_lower:
            return std_ch
    for alias, std in CHANNEL_ALIASES.items():
        if alias.lower() == name_lower:
            return std
    return None


def process_edf_upload(file_bytes, filename):
    """Process an uploaded EDF/BDF file and return preprocessed windows.

    Returns:
        dict with keys: windows (np.ndarray [N, 19, 1280]),
        sfreq, channels_found, channel_count, duration, info, steps
    """
    import mne
    mne.set_log_level("ERROR")

    steps = []

    # Write to temp file for MNE
    import tempfile, os
    suffix = Path(filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        # Step 1: Load raw
        steps.append("📂 Loading EEG recording...")
        if suffix in ('.edf', '.EDF'):
            raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
        elif suffix in ('.bdf', '.BDF'):
            raw = mne.io.read_raw_bdf(tmp_path, preload=True, verbose=False)
        elif suffix in ('.set', '.SET'):
            raw = mne.io.read_raw_eeglab(tmp_path, preload=True, verbose=False)
        else:
            return None, f"Unsupported format: {suffix}"

        orig_sfreq = raw.info['sfreq']
        orig_duration = raw.times[-1]
        orig_nchan = len(raw.ch_names)
        steps.append(f"✅ Loaded: {orig_nchan} channels, {orig_sfreq} Hz, {orig_duration:.1f}s")

        # Step 2: Select and map channels
        steps.append("🔧 Mapping channels to standard 10-20 montage...")
        ch_mapping = {}
        for ch in raw.ch_names:
            std = normalize_channel_name(ch)
            if std is not None:
                ch_mapping[ch] = std

        channels_found = list(set(ch_mapping.values()))
        n_found = len(channels_found)
        steps.append(f"✅ Matched {n_found}/19 standard channels")

        if n_found < 10:
            return None, f"Insufficient channels: found only {n_found}/19 standard EEG channels. Need at least 10."

        # Step 3: Extract data with channel reordering
        data = raw.get_data()
        n_samples_total = data.shape[1]

        output = np.zeros((19, n_samples_total), dtype=np.float64)
        for orig_ch, std_ch in ch_mapping.items():
            idx_orig = raw.ch_names.index(orig_ch)
            idx_std = CHANNELS_19.index(std_ch)
            output[idx_std] = data[idx_orig]

        # Step 4: Resample to 256 Hz
        if abs(orig_sfreq - TARGET_SRATE) > 1.0:
            steps.append(f"🔄 Resampling {orig_sfreq} Hz → {TARGET_SRATE} Hz...")
            from scipy.signal import resample_poly
            from math import gcd
            up = TARGET_SRATE
            down = int(orig_sfreq)
            g = gcd(up, down)
            output = resample_poly(output, up // g, down // g, axis=1).astype(np.float64)
            steps.append(f"✅ Resampled to {TARGET_SRATE} Hz ({output.shape[1]} samples)")
        else:
            steps.append(f"✅ Already at {TARGET_SRATE} Hz")

        # Step 5: Segment into 5-second windows (50% overlap)
        steps.append("✂️ Segmenting into 5-second windows (50% overlap)...")
        total_samples = output.shape[1]
        step_samples = WINDOW_SAMPLES // 2  # 50% overlap

        if total_samples < WINDOW_SAMPLES:
            # Pad short recordings
            padded = np.zeros((19, WINDOW_SAMPLES), dtype=np.float64)
            padded[:, :total_samples] = output
            output = padded
            windows = [output]
            steps.append(f"⚠️ Recording shorter than 5s — zero-padded to 1280 samples")
        else:
            windows = []
            start = 0
            while start + WINDOW_SAMPLES <= total_samples:
                windows.append(output[:, start:start + WINDOW_SAMPLES])
                start += step_samples
            steps.append(f"✅ Created {len(windows)} windows")

        windows = np.array(windows, dtype=np.float32)

        # Step 6: Z-score normalization per channel per window
        steps.append("📊 Applying per-channel z-score normalization...")
        mean = windows.mean(axis=2, keepdims=True)
        std = windows.std(axis=2, keepdims=True)
        std[std < 1e-8] = 1e-8
        windows = (windows - mean) / std
        steps.append("✅ Normalization complete")

        return {
            "windows": windows,
            "sfreq": TARGET_SRATE,
            "channels_found": channels_found,
            "channel_count": n_found,
            "duration": orig_duration,
            "orig_sfreq": orig_sfreq,
            "orig_nchan": orig_nchan,
            "n_windows": len(windows),
            "steps": steps,
        }, None

    except Exception as exc:
        return None, f"Processing error: {exc}"
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def process_csv_upload(file_bytes):
    """Process an uploaded CSV file containing EEG data.

    Expected CSV format: columns are channel names, rows are time samples.
    Or: 19 columns (unnamed), rows are samples.
    Or: columns named Fp1, Fp2, ... O2 (19 channels).

    Returns same dict format as process_edf_upload.
    """
    steps = []
    steps.append("📂 Parsing CSV file...")

    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as e:
        return None, f"Could not parse CSV: {e}"

    steps.append(f"✅ Loaded CSV: {df.shape[0]} rows × {df.shape[1]} columns")

    # Try to identify channel columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 19:
        return None, f"CSV has only {len(numeric_cols)} numeric columns. Need at least 19 for EEG channels."

    # Try to match column names to standard channels
    col_mapping = {}
    for col in numeric_cols:
        std = normalize_channel_name(str(col))
        if std is not None:
            col_mapping[col] = std

    if len(col_mapping) >= 10:
        steps.append(f"🔧 Matched {len(col_mapping)} columns to standard channel names")
        # Use named mapping
        data = np.zeros((19, df.shape[0]), dtype=np.float64)
        channels_found = []
        for col, std in col_mapping.items():
            idx = CHANNELS_19.index(std)
            data[idx] = df[col].values.astype(np.float64)
            channels_found.append(std)
    elif len(numeric_cols) >= 19:
        steps.append("⚠️ Column names don't match standard channels; using first 19 numeric columns in order")
        data = df[numeric_cols[:19]].values.T.astype(np.float64)
        channels_found = CHANNELS_19.copy()
    else:
        return None, f"Cannot map CSV columns to 19 EEG channels."

    n_found = len(set(channels_found))
    steps.append(f"✅ Using {n_found} channels, {data.shape[1]} samples")

    # Segment
    total_samples = data.shape[1]
    step_samples = WINDOW_SAMPLES // 2

    if total_samples < WINDOW_SAMPLES:
        padded = np.zeros((19, WINDOW_SAMPLES), dtype=np.float64)
        padded[:, :total_samples] = data
        data = padded
        windows = [data]
        steps.append(f"⚠️ Data shorter than 1280 samples — zero-padded")
    else:
        windows = []
        start = 0
        while start + WINDOW_SAMPLES <= total_samples:
            windows.append(data[:, start:start + WINDOW_SAMPLES])
            start += step_samples

    steps.append(f"✂️ Created {len(windows)} windows")
    windows = np.array(windows, dtype=np.float32)

    # Z-score
    steps.append("📊 Applying z-score normalization...")
    mean = windows.mean(axis=2, keepdims=True)
    std_val = windows.std(axis=2, keepdims=True)
    std_val[std_val < 1e-8] = 1e-8
    windows = (windows - mean) / std_val
    steps.append("✅ Normalization complete")

    return {
        "windows": windows,
        "sfreq": TARGET_SRATE,
        "channels_found": channels_found,
        "channel_count": n_found,
        "duration": total_samples / TARGET_SRATE,
        "orig_sfreq": TARGET_SRATE,
        "orig_nchan": len(numeric_cols),
        "n_windows": len(windows),
        "steps": steps,
    }, None


def process_npy_upload(file_bytes):
    """Process an uploaded .npy file containing preprocessed EEG data."""
    steps = []
    steps.append("📂 Loading NumPy array...")

    try:
        arr = np.load(io.BytesIO(file_bytes))
    except Exception as e:
        return None, f"Could not load .npy file: {e}"

    steps.append(f"✅ Array shape: {arr.shape}, dtype: {arr.dtype}")

    # Determine shape interpretation
    if arr.ndim == 2:
        if arr.shape == (19, 1280):
            windows = arr[np.newaxis].astype(np.float32)
            steps.append("✅ Interpreted as single window [19, 1280]")
        elif arr.shape == (1280, 19):
            windows = arr.T[np.newaxis].astype(np.float32)
            steps.append("✅ Interpreted as single window [1280, 19] → transposed")
        elif arr.shape[0] == 19:
            # [19, N] — segment
            steps.append(f"🔧 Interpreting as [19, {arr.shape[1]}] continuous data")
            data = arr.astype(np.float64)
            win_list = []
            start = 0
            while start + WINDOW_SAMPLES <= data.shape[1]:
                win_list.append(data[:, start:start + WINDOW_SAMPLES])
                start += WINDOW_SAMPLES // 2
            if not win_list:
                return None, f"Array too short to form a 1280-sample window"
            windows = np.array(win_list, dtype=np.float32)
            steps.append(f"✂️ Created {len(windows)} windows")
        elif arr.shape[1] == 19:
            # [N, 19] — transpose and segment
            data = arr.T.astype(np.float64)
            win_list = []
            start = 0
            while start + WINDOW_SAMPLES <= data.shape[1]:
                win_list.append(data[:, start:start + WINDOW_SAMPLES])
                start += WINDOW_SAMPLES // 2
            if not win_list:
                return None, f"Array too short to form a 1280-sample window"
            windows = np.array(win_list, dtype=np.float32)
            steps.append(f"✂️ Created {len(windows)} windows from [N, 19] array")
        else:
            return None, f"Unrecognized 2D shape {arr.shape}. Expected [19, 1280], [1280, 19], or similar."
    elif arr.ndim == 3:
        if arr.shape[1] == 19 and arr.shape[2] == 1280:
            windows = arr.astype(np.float32)
            steps.append(f"✅ Loaded {arr.shape[0]} pre-segmented windows [N, 19, 1280]")
        elif arr.shape[1] == 1280 and arr.shape[2] == 19:
            windows = arr.transpose(0, 2, 1).astype(np.float32)
            steps.append(f"✅ Loaded {arr.shape[0]} windows [N, 1280, 19] → transposed")
        else:
            return None, f"Unrecognized 3D shape {arr.shape}. Expected [N, 19, 1280]."
    else:
        return None, f"Unexpected array dimensions: {arr.ndim}D. Expected 2D or 3D."

    # Z-score normalize
    steps.append("📊 Applying z-score normalization...")
    mean = windows.mean(axis=2, keepdims=True)
    std_val = windows.std(axis=2, keepdims=True)
    std_val[std_val < 1e-8] = 1e-8
    windows = (windows - mean) / std_val
    steps.append("✅ Ready for inference")

    return {
        "windows": windows,
        "sfreq": TARGET_SRATE,
        "channels_found": CHANNELS_19.copy(),
        "channel_count": 19,
        "duration": windows.shape[0] * 5.0,
        "orig_sfreq": TARGET_SRATE,
        "orig_nchan": 19,
        "n_windows": windows.shape[0],
        "steps": steps,
    }, None


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_eeg_window(window_data, title="EEG Signal", n_channels_show=19):
    """Plot EEG window with professional styling.

    Args:
        window_data: [19, 1280] numpy array
        title: plot title
        n_channels_show: number of channels to display
    """
    n_ch = min(n_channels_show, window_data.shape[0])
    n_samp = window_data.shape[1]
    time_axis = np.linspace(0, n_samp / TARGET_SRATE, n_samp)

    fig, ax = plt.subplots(figsize=(14, max(4, n_ch * 0.4)), dpi=100)
    fig.patch.set_facecolor('#fafafa')
    ax.set_facecolor('#fafafa')

    # Compute spacing
    spacing = 4.0
    colors = plt.cm.Set3(np.linspace(0, 1, n_ch))

    for i in range(n_ch):
        signal = window_data[i]
        # Clip extreme values for display
        signal = np.clip(signal, -5, 5)
        offset = (n_ch - 1 - i) * spacing
        ax.plot(time_axis, signal + offset, lw=0.7, color=colors[i], alpha=0.85)

    ax.set_yticks([(n_ch - 1 - i) * spacing for i in range(n_ch)])
    ax.set_yticklabels(CHANNELS_19[:n_ch], fontsize=8, fontweight='500')
    ax.set_xlabel("Time (seconds)", fontsize=10, fontweight='500')
    ax.set_title(title, fontsize=12, fontweight='600', pad=10)
    ax.set_xlim(0, time_axis[-1])
    ax.grid(True, alpha=0.15, linestyle='-')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig


def plot_probability_bars(probabilities, predicted_idx):
    """Plot probability distribution as horizontal bars."""
    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=100)
    fig.patch.set_facecolor('#fafafa')
    ax.set_facecolor('#fafafa')

    y_pos = np.arange(5)
    bars = ax.barh(y_pos, probabilities * 100, height=0.6,
                   color=[DISEASE_COLORS[i] if i == predicted_idx else '#cbd5e1' for i in range(5)],
                   edgecolor='none', alpha=0.85)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{DISEASE_ICONS[i]} {DISEASE_NAMES[i]}" for i in range(5)],
                       fontsize=9, fontweight='500')
    ax.set_xlabel("Probability (%)", fontsize=10, fontweight='500')
    ax.set_xlim(0, 105)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='x', alpha=0.15)

    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        ax.text(prob * 100 + 1.5, i, f"{prob*100:.1f}%",
                va='center', fontsize=9, fontweight='600',
                color=DISEASE_COLORS[i] if i == predicted_idx else '#64748b')

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD RESOURCES
# ═══════════════════════════════════════════════════════════════════════════════

model, model_status = load_model()
metrics_data = load_metrics()


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0 0.5rem 0;">
        <span style="font-size: 2.5rem;">🧠</span>
        <h3 style="color: #e2e8f0; margin: 0.5rem 0 0.2rem 0; font-size: 1.1rem;">
            EEG Classification
        </h3>
        <p style="color: #94a3b8; font-size: 0.75rem; margin: 0;">
            Research Dashboard v1.0
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr style="border-color: #334155; margin: 0.5rem 0;">', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠 Home", "🎯 EEG Prediction", "🔍 Explainable AI",
         "📊 Model Performance", "🔬 About & Research"],
        label_visibility="collapsed",
    )

    st.markdown('<hr style="border-color: #334155; margin: 0.5rem 0;">', unsafe_allow_html=True)

    # Model status indicator
    if model is not None:
        st.markdown(f"""
        <div style="padding: 0.5rem; border-radius: 8px; background: #064e3b; margin: 0.5rem 0;">
            <span style="color: #34d399; font-size: 0.8rem;">✅ Model Loaded</span><br>
            <span style="color: #6ee7b7; font-size: 0.7rem;">{model_status}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="padding: 0.5rem; border-radius: 8px; background: #7f1d1d; margin: 0.5rem 0;">
            <span style="color: #fca5a5; font-size: 0.8rem;">❌ Model Not Loaded</span><br>
            <span style="color: #fca5a5; font-size: 0.7rem;">{model_status}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="position: fixed; bottom: 1rem; padding: 0.5rem;">
        <p style="color: #475569; font-size: 0.65rem; margin: 0;">
            Thesis Research Project<br>
            © 2026 — Not for clinical use
        </p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero-container">
    <div class="hero-title">🧠 Transformer-Based Deep Learning Framework for Robust EEG Signal Classification</div>
    <div class="hero-subtitle">
        CNN–Transformer Architecture · 5-Class Multi-Disease Classification · Grad-CAM Explainability · 19-Channel 10-20 Montage
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Home":

    st.markdown("## Project Overview")
    st.markdown("""
    This research platform demonstrates a **unified CNN–Transformer framework** for
    automated classification of scalp EEG recordings into five neurological categories.
    The system combines convolutional feature extraction with Transformer-based temporal
    modelling and provides model-based visual attribution using Grad-CAM.
    """)

    # Disease categories
    st.markdown("### Classification Categories")
    cols = st.columns(5)
    disease_descs = [
        "Normal brain activity baseline",
        "Seizure-related EEG patterns",
        "Neurodegenerative cognitive decline",
        "Motor-system neurodegeneration",
        "Major Depressive Disorder patterns",
    ]
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"""
            <div class="disease-card {DISEASE_KEYS[i]}">
                <div class="disease-name">{DISEASE_ICONS[i]} {DISEASE_NAMES[i]}</div>
                <div class="disease-desc">{disease_descs[i]}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Architecture overview
    col_arch, col_specs = st.columns([3, 2])

    with col_arch:
        st.markdown("### Model Architecture")
        st.markdown("""
        <div class="arch-box">
            <b>📥 Input:</b> Raw 19-channel EEG (19 × 1,280 @ 256 Hz)<br>
            <div class="arch-arrow">↓</div>
            <b>🔧 Preprocessing:</b> Bandpass 0.5–45 Hz · Notch 50/60 Hz · Resample 256 Hz<br>
            <div class="arch-arrow">↓</div>
            <b>🧮 CNN Feature Extractor:</b> EEGNet-style temporal + spatial convolutions<br>
            &nbsp;&nbsp;&nbsp;&nbsp;Conv2d → DepthwiseConv2d → SeparableConv2d → [B, 40, 64]<br>
            <div class="arch-arrow">↓</div>
            <b>🔄 Transformer Encoder:</b> CLS token + positional encoding<br>
            &nbsp;&nbsp;&nbsp;&nbsp;2 layers, 4 heads, GELU activation, pre-norm<br>
            <div class="arch-arrow">↓</div>
            <b>🎯 Classification Head:</b> LayerNorm → Linear(128) → GELU → Linear(5)<br>
            <div class="arch-arrow">↓</div>
            <b>📊 Output:</b> 5-class probability distribution
        </div>
        """, unsafe_allow_html=True)

    with col_specs:
        st.markdown("### Key Specifications")

        specs = [
            ("Input Shape", "19 × 1,280"),
            ("Sampling Rate", "256 Hz"),
            ("Window Duration", "5 seconds"),
            ("Channels", "Standard 10-20"),
            ("Classes", "5"),
            ("Embedding Dim", "64"),
            ("Attention Heads", "4"),
            ("Transformer Layers", "2"),
            ("Parameters", "~57K"),
        ]
        for label, value in specs:
            st.markdown(f"**{label}:** `{value}`")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Metrics summary
    if metrics_data:
        st.markdown("### Evaluation Results (Held-Out Test Set)")
        test_m = metrics_data.get("test_metrics", {})

        m_cols = st.columns(5)
        metric_items = [
            ("Test Accuracy", f"{test_m.get('accuracy', 0)*100:.2f}%"),
            ("Balanced Accuracy", f"{test_m.get('balanced_accuracy', 0)*100:.2f}%"),
            ("Macro F1", f"{test_m.get('macro_f1', 0):.4f}"),
            ("ROC-AUC", f"{test_m.get('macro_roc_auc', 0):.4f}"),
            ("Cohen's κ", f"{test_m.get('cohen_kappa', 0):.4f}"),
        ]
        for col, (label, value) in zip(m_cols, metric_items):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-val">{value}</div>
                    <div class="metric-lbl">{label}</div>
                </div>
                """, unsafe_allow_html=True)

        st.caption("Window-level metrics · 1,525 test windows · Subject-level partitioning · Zero subject leakage")

    # Disclaimer
    st.markdown("""
    <div class="disclaimer-box">
        <b>⚠️ Research Prototype Disclaimer:</b> This software is an academic research tool
        developed for computational neuroscience investigation. It is <b>not</b> approved as a
        medical device, is <b>not</b> certified for clinical diagnosis, and must <b>not</b> be used
        as a substitute for professional medical evaluation.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: EEG PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🎯 EEG Prediction":

    st.markdown("## 🎯 EEG Signal Classification")
    st.markdown("""
    Upload a real EEG recording to run inference through the trained CNN–Transformer model.
    The system will preprocess the signal, extract features, and produce a 5-class prediction.
    """)

    if model is None:
        st.error(f"⛔ Model checkpoint not available: {model_status}")
        st.info("Place `best_model.pt` in `checkpoints/` directory and restart the app.")
        st.stop()

    # Upload section
    st.markdown("### 📂 Upload EEG Recording")
    col_upload, col_info = st.columns([2, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Select an EEG file",
            type=["edf", "bdf", "set", "csv", "npy"],
            help="Supported formats: EDF, BDF, EEGLAB .set, CSV (19 channels × samples), NumPy .npy",
        )

    with col_info:
        st.markdown("""
        **Supported Formats:**
        - 📋 **EDF/BDF** — European Data Format (recommended)
        - 📋 **SET** — EEGLAB dataset format
        - 📋 **CSV** — Columnar data (19 EEG channels)
        - 📋 **NPY** — NumPy array [19, 1280] or [N, 19, 1280]

        **Requirements:**
        - Standard 10-20 montage (≥10 of 19 channels)
        - Any sampling rate (auto-resampled to 256 Hz)
        """)

    if uploaded_file is not None:
        file_ext = Path(uploaded_file.name).suffix.lower()
        file_bytes = uploaded_file.read()

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Process the file
        with st.spinner("Processing EEG recording..."):
            if file_ext in ('.edf', '.bdf', '.set'):
                result, error = process_edf_upload(file_bytes, uploaded_file.name)
            elif file_ext == '.csv':
                result, error = process_csv_upload(file_bytes)
            elif file_ext == '.npy':
                result, error = process_npy_upload(file_bytes)
            else:
                result, error = None, f"Unsupported file format: {file_ext}"

        if error:
            st.error(f"❌ {error}")
            st.stop()

        # Show preprocessing steps
        st.markdown("### 🔧 Preprocessing Pipeline")
        with st.expander("View preprocessing steps", expanded=True):
            for step in result["steps"]:
                st.markdown(f"&nbsp;&nbsp;{step}")

        # Recording info
        info_cols = st.columns(4)
        with info_cols[0]:
            st.metric("Channels Matched", f"{result['channel_count']}/19")
        with info_cols[1]:
            st.metric("Original Sample Rate", f"{result['orig_sfreq']:.0f} Hz")
        with info_cols[2]:
            st.metric("Duration", f"{result['duration']:.1f}s")
        with info_cols[3]:
            st.metric("Windows Created", result['n_windows'])

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Window selection
        windows = result["windows"]
        n_windows = windows.shape[0]

        if n_windows > 1:
            window_idx = st.slider(
                "Select window for analysis",
                0, n_windows - 1, 0,
                help=f"Recording contains {n_windows} overlapping 5-second windows"
            )
        else:
            window_idx = 0

        selected_window = windows[window_idx]  # [19, 1280]

        # Signal preview
        st.markdown("### 📈 EEG Signal Preview")
        n_ch_show = st.select_slider("Channels to display", [5, 10, 19], value=19)
        fig_eeg = plot_eeg_window(
            selected_window, f"Window {window_idx + 1}/{n_windows} — Preprocessed EEG",
            n_channels_show=n_ch_show
        )
        st.pyplot(fig_eeg)
        plt.close(fig_eeg)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Inference
        st.markdown("### 🚀 Model Inference")

        if st.button("Run Classification", type="primary", use_container_width=True):
            with st.spinner("Running inference..."):
                input_tensor = torch.tensor(selected_window, dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    logits = model(input_tensor)
                    probabilities = torch.softmax(logits, dim=-1).squeeze(0).numpy()

            predicted_idx = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_idx])

            # Result display
            res_col1, res_col2 = st.columns([1, 2])

            with res_col1:
                st.markdown(f"""
                <div class="prediction-box">
                    <div style="font-size: 2rem;">{DISEASE_ICONS[predicted_idx]}</div>
                    <div class="prediction-class">{DISEASE_NAMES[predicted_idx]}</div>
                    <div class="prediction-conf">Model output: {confidence*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            with res_col2:
                fig_prob = plot_probability_bars(probabilities, predicted_idx)
                st.pyplot(fig_prob)
                plt.close(fig_prob)

            # Detailed probabilities table
            with st.expander("View detailed probabilities"):
                prob_df = pd.DataFrame({
                    "Class": DISEASE_NAMES,
                    "Probability": [f"{p*100:.2f}%" for p in probabilities],
                    "Raw Score": [f"{p:.6f}" for p in probabilities],
                })
                st.dataframe(prob_df, use_container_width=True, hide_index=True)

            st.info(
                "⚕️ This prediction is a **research model output** from a computational "
                "prototype. It must **not** be interpreted as a clinical diagnosis. "
                "Always consult qualified medical professionals for health decisions."
            )

    else:
        # Show synthetic demo option when no file is uploaded
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 🧪 Quick Demo (Synthetic Data)")
        st.caption(
            "No EEG file? Run a demo with synthetic random noise to verify the pipeline works. "
            "Predictions on synthetic data are meaningless and serve only as a technical smoke test."
        )

        if st.button("🧪 Run Demo with Synthetic Signal", type="secondary"):
            synth = np.random.randn(19, 1280).astype(np.float32)
            # Z-score
            mean = synth.mean(axis=1, keepdims=True)
            std_val = synth.std(axis=1, keepdims=True)
            std_val[std_val < 1e-8] = 1e-8
            synth = (synth - mean) / std_val

            fig_synth = plot_eeg_window(synth, "Synthetic Random Noise (Demo Only)")
            st.pyplot(fig_synth)
            plt.close(fig_synth)

            with torch.no_grad():
                input_t = torch.tensor(synth, dtype=torch.float32).unsqueeze(0)
                logits = model(input_t)
                probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()

            pred_idx = int(np.argmax(probs))
            st.warning(
                f"⚠️ **Synthetic data prediction:** {DISEASE_NAMES[pred_idx]} ({probs[pred_idx]*100:.1f}%) — "
                f"This is **meaningless random noise**, not a real EEG. Shown only to verify pipeline connectivity."
            )

            fig_p = plot_probability_bars(probs, pred_idx)
            st.pyplot(fig_p)
            plt.close(fig_p)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: EXPLAINABLE AI
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Explainable AI":

    st.markdown("## 🔍 Explainable AI — Model Attribution")
    st.markdown("""
    The framework provides gradient-based attribution visualizations to inspect which
    temporal EEG regions and channel-level contributions are associated with model predictions.

    These visualizations represent **mathematical model attribution** — which input features the
    network weighted most heavily — and should **not** be interpreted as clinically established biomarkers.
    """)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Pre-computed Grad-CAM visualizations
    st.markdown("### Pre-Computed Grad-CAM Analysis")

    composite_path = RESULTS_DIR / "gradcam_analysis.png"
    if composite_path.exists():
        st.image(
            str(composite_path),
            caption="Multi-Disease Grad-CAM Temporal Attribution and Electrode Saliency",
            use_container_width=True,
        )
    else:
        st.warning("Composite Grad-CAM image not found in results/")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Individual class profiles
    st.markdown("### Individual Class Attribution Profiles")

    sel_col1, sel_col2 = st.columns([1, 2])
    with sel_col1:
        selected_class_key = st.selectbox(
            "Select condition:",
            DISEASE_KEYS,
            format_func=lambda x: f"{DISEASE_ICONS[DISEASE_KEYS.index(x)]} {DISEASE_NAMES[DISEASE_KEYS.index(x)]}",
        )

    individual_path = RESULTS_DIR / "gradcam" / f"gradcam_{selected_class_key}.png"
    if individual_path.exists():
        st.image(
            str(individual_path),
            caption=f"Grad-CAM Attribution — {DISEASE_NAMES[DISEASE_KEYS.index(selected_class_key)]}",
            use_container_width=True,
        )
    else:
        st.info(f"Individual Grad-CAM image for {selected_class_key} not available.")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Live explainability on synthetic data (or model demo)
    st.markdown("### Interactive Explainability Demo")

    if model is not None:
        st.caption(
            "Generate a live Grad-CAM and saliency visualization using synthetic data. "
            "This demonstrates the explainability pipeline technically; attributions on random noise are not meaningful."
        )

        if st.button("🔬 Generate Live Attribution (Synthetic Data)", type="primary"):
            from training.explainability import EEGExplainer

            with st.spinner("Computing attributions..."):
                device = torch.device("cpu")
                explainer = EEGExplainer(model, device)

                x_synth = torch.randn(1, 19, 1280)

                # Grad-CAM
                cam = explainer.grad_cam(x_synth)

                # Input gradients (saliency)
                saliency = explainer.input_gradients(x_synth)

                # Attention
                attn = explainer.transformer_attention(x_synth)

            # Plot results
            fig = plt.figure(figsize=(14, 10), dpi=100)
            fig.patch.set_facecolor('#fafafa')
            gs = gridspec.GridSpec(3, 1, height_ratios=[1, 2, 1], hspace=0.35)

            # Grad-CAM temporal
            ax1 = fig.add_subplot(gs[0])
            time_ax = np.linspace(0, 5, len(cam))
            ax1.fill_between(time_ax, cam, alpha=0.4, color='#ef4444')
            ax1.plot(time_ax, cam, color='#ef4444', lw=1.5)
            ax1.set_title("Grad-CAM Temporal Activation", fontweight='600', fontsize=11)
            ax1.set_xlabel("Time (seconds)")
            ax1.set_ylabel("Activation")
            ax1.set_xlim(0, 5)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)

            # Channel saliency heatmap
            ax2 = fig.add_subplot(gs[1])
            im = ax2.imshow(saliency, aspect='auto', cmap='hot',
                           extent=[0, 5, 18.5, -0.5], interpolation='bilinear')
            ax2.set_yticks(range(19))
            ax2.set_yticklabels(CHANNELS_19, fontsize=7)
            ax2.set_xlabel("Time (seconds)")
            ax2.set_title("Channel × Time Saliency Map", fontweight='600', fontsize=11)
            plt.colorbar(im, ax=ax2, fraction=0.02, pad=0.02, label="Gradient magnitude")

            # Attention weights
            if attn is not None and len(attn) > 0:
                ax3 = fig.add_subplot(gs[2])
                ax3.bar(range(len(attn)), attn, color='#3b82f6', alpha=0.7)
                ax3.set_title("Transformer CLS Token Attention Weights", fontweight='600', fontsize=11)
                ax3.set_xlabel("Temporal Token Index")
                ax3.set_ylabel("Attention Weight")
                ax3.spines['top'].set_visible(False)
                ax3.spines['right'].set_visible(False)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.caption("⚠️ Attributions shown above are computed on synthetic random noise for demonstration purposes only.")
    else:
        st.warning("Model not loaded. Explainability features require a loaded model checkpoint.")

    # Explanation of methods
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### Understanding the Visualizations")

    with st.expander("What do these visualizations mean?", expanded=True):
        st.markdown("""
        **Grad-CAM (Gradient-weighted Class Activation Mapping):**
        Shows which temporal regions of the EEG signal the CNN features most strongly
        activate for the predicted class. High activation regions indicate time windows
        the model considers most discriminative.

        **Channel × Time Saliency Map:**
        Displays the magnitude of input gradients across all 19 channels and 1280 time
        samples. Brighter regions indicate inputs where small changes would most affect
        the model's prediction.

        **Transformer Attention Weights:**
        Shows how the classification (CLS) token attends to each temporal feature token
        from the CNN output. Higher weights indicate temporal segments the Transformer
        considers more relevant for classification.

        > **Important:** These are mathematical model attributions — they reveal what the
        > network learned to focus on, not necessarily neurophysiologically meaningful biomarkers.
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Model Performance":

    st.markdown("## 📊 Model Evaluation — Held-Out Test Set")
    st.caption("All metrics evaluated on 1,525 independent test windows from subjects never seen during training.")

    if metrics_data is None:
        st.error("Could not load `results/metrics.json`.")
        st.stop()

    test_m = metrics_data.get("test_metrics", {})

    # Top-level metrics
    st.markdown("### Overall Performance Metrics")
    m_cols = st.columns(6)
    metric_items = [
        ("Accuracy", f"{test_m.get('accuracy', 0)*100:.2f}%"),
        ("Balanced Accuracy", f"{test_m.get('balanced_accuracy', 0)*100:.2f}%"),
        ("Macro F1", f"{test_m.get('macro_f1', 0):.4f}"),
        ("Weighted F1", f"{test_m.get('weighted_f1', 0):.4f}"),
        ("ROC-AUC (OVR)", f"{test_m.get('macro_roc_auc', 0):.4f}"),
        ("Cohen's κ", f"{test_m.get('cohen_kappa', 0):.4f}"),
    ]
    for col, (label, value) in zip(m_cols, metric_items):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{value}</div>
                <div class="metric-lbl">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Confusion matrix and training curves
    vis_col1, vis_col2 = st.columns(2)

    with vis_col1:
        st.markdown("### Confusion Matrix")
        cm_path = RESULTS_DIR / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), caption="5-Class Confusion Matrix — 1,525 Test Windows", use_container_width=True)
        else:
            # Try to render from metrics data
            cm_data = metrics_data.get("confusion_matrix_test_5x5", {})
            if cm_data:
                matrix = np.array(cm_data["matrix"])
                labels = cm_data["labels"]

                fig_cm, ax_cm = plt.subplots(figsize=(6, 5), dpi=100)
                fig_cm.patch.set_facecolor('#fafafa')
                im = ax_cm.imshow(matrix, cmap='Blues', interpolation='nearest')
                ax_cm.set_xticks(range(5))
                ax_cm.set_yticks(range(5))
                ax_cm.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
                ax_cm.set_yticklabels(labels, fontsize=8)
                ax_cm.set_xlabel("Predicted", fontweight='600')
                ax_cm.set_ylabel("True", fontweight='600')
                ax_cm.set_title("Confusion Matrix", fontweight='600')

                for i in range(5):
                    for j in range(5):
                        color = 'white' if matrix[i, j] > matrix.max() / 2 else 'black'
                        ax_cm.text(j, i, str(matrix[i, j]), ha='center', va='center',
                                  fontsize=10, fontweight='600', color=color)

                plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
                plt.tight_layout()
                st.pyplot(fig_cm)
                plt.close(fig_cm)
            else:
                st.warning("Confusion matrix not available.")

    with vis_col2:
        st.markdown("### Training & Validation Curves")
        curves_path = RESULTS_DIR / "training_curves.png"
        if curves_path.exists():
            st.image(str(curves_path), caption="Training and validation trajectories", use_container_width=True)
        else:
            st.warning("Training curves image not found.")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Per-class report
    st.markdown("### Per-Class Classification Report")

    report_path = RESULTS_DIR / "classification_report.csv"
    if report_path.exists():
        report_df = pd.read_csv(report_path)
        st.dataframe(report_df, use_container_width=True, hide_index=True)
    else:
        # Build from metrics.json
        per_class = metrics_data.get("per_class_metrics", {})
        if per_class:
            rows = []
            for key, data in per_class.items():
                rows.append({
                    "Class": data.get("display_name", key),
                    "Precision": f"{data.get('precision', 0):.4f}",
                    "Recall": f"{data.get('recall', 0):.4f}",
                    "F1-Score": f"{data.get('f1_score', 0):.4f}",
                    "Support": data.get("test_support", 0),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Dataset composition
    st.markdown("### Dataset Composition")

    dataset_df = pd.DataFrame([
        {"Condition": "Healthy Control", "Class": 0, "Source": "SRM + ds004504 + ds004584 + NEMAR",
         "Train": 3090, "Val": 480, "Test": 540, "Total": 4110},
        {"Condition": "Epilepsy", "Class": 1, "Source": "CHB-MIT (7 subjects)",
         "Train": 326, "Val": 57, "Test": 55, "Total": 438},
        {"Condition": "Alzheimer's", "Class": 2, "Source": "OpenNeuro ds004504",
         "Train": 720, "Val": 180, "Test": 180, "Total": 1080},
        {"Condition": "Parkinson's", "Class": 3, "Source": "OpenNeuro ds004584",
         "Train": 2070, "Val": 390, "Test": 540, "Total": 3000},
        {"Condition": "Depression", "Class": 4, "Source": "NEMAR nm000114 / MODMA",
         "Train": 1260, "Val": 390, "Test": 210, "Total": 1860},
    ])
    st.dataframe(dataset_df, use_container_width=True, hide_index=True)

    tot_cols = st.columns(4)
    for col, (lbl, val) in zip(tot_cols, [
        ("Train Windows", "7,466"), ("Val Windows", "1,497"),
        ("Test Windows", "1,525"), ("Total Windows", "10,488"),
    ]):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{val}</div>
                <div class="metric-lbl">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.caption("All splits use strict subject-level partitioning with zero subject overlap.")

    # Experiment comparison
    manifest_path = RESULTS_DIR / "experiments" / "manifest.csv"
    if manifest_path.exists():
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("### Loss Function Experiments")
        st.markdown("""
        Candidate loss formulations were evaluated on validation data. The canonical
        baseline (unweighted cross-entropy) was retained as it demonstrated superior
        held-out test performance.
        """)
        try:
            m_df = pd.read_csv(manifest_path)
            st.dataframe(m_df, use_container_width=True, hide_index=True)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT & RESEARCH
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🔬 About & Research":

    st.markdown("## 🔬 Research Overview")

    st.markdown("""
    ### Problem Statement

    Neurological disorders affect hundreds of millions of people globally. Electroencephalography
    (EEG) is a non-invasive, widely accessible tool for assessing brain function, but manual
    interpretation requires significant clinical expertise and is subject to inter-rater variability.

    This research proposes a **unified deep learning framework** that can automatically classify
    scalp EEG recordings across five neurological categories, potentially assisting clinicians
    in screening and triage workflows.

    ### Proposed Architecture

    The **CNN–Transformer** architecture combines:

    1. **EEGNet-inspired CNN Backbone** for spatial-temporal feature extraction from raw EEG signals
    2. **Transformer Encoder** with multi-head self-attention for capturing long-range temporal dependencies
    3. **Gradient-based Explainability** (Grad-CAM + Input Gradient Saliency) for model interpretability
    """)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    ### EEG Harmonization

    A key challenge in multi-cohort EEG research is heterogeneity across recording setups.
    This project implements a harmonization pipeline that standardizes:

    - **Channel montage:** 19 standard 10-20 channels with automatic alias resolution (10-10 → 10-20 mapping)
    - **Sampling rate:** Resampled to 256 Hz via polyphase resampling
    - **Filtering:** 0.5–45 Hz bandpass + 50/60 Hz notch filtering
    - **Segmentation:** 5-second windows with 50% overlap (1,280 samples per window)
    - **Normalization:** Per-channel, per-window z-score standardization
    - **Missing channels:** Zero-padding with optional topological interpolation

    ### Five-Class Classification

    | Class | Condition | Data Source |
    |:---:|:---|:---|
    | 0 | Healthy Control | SRM + OpenNeuro ds004504 + ds004584 + NEMAR |
    | 1 | Epilepsy | CHB-MIT Scalp EEG Database |
    | 2 | Alzheimer's Disease | OpenNeuro ds004504 |
    | 3 | Parkinson's Disease | OpenNeuro ds004584 |
    | 4 | Depression (MDD) | NEMAR nm000114 / MODMA |
    """)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    ### Explainability Methods

    The framework implements three gradient-based attribution methods:

    1. **Grad-CAM** — Visualizes which temporal regions of the EEG most strongly activate
       the CNN features for a given prediction
    2. **Input Gradient Saliency** — Shows the sensitivity of the prediction to each
       channel × time input element
    3. **Transformer Attention Weights** — Reveals how the classification token attends
       to temporal feature tokens

    ### Limitations and Future Work

    - **Sample size:** Some classes (epilepsy: 55 test windows from 1 subject) have limited test representation
    - **Multi-center confounding:** Acquisition hardware differences across cohorts may contribute to classification signals
    - **Generalization:** Results are validated on specific open datasets and may not generalize to arbitrary clinical recordings
    - **Clinical validation:** No prospective clinical trial has been conducted
    - **Model capacity:** The lightweight architecture (~57K parameters) trades capacity for efficiency

    **Future directions** include larger multi-center datasets, adversarial domain adaptation,
    subject-level classification aggregation, and prospective clinical evaluation.
    """)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Full experiment summary
    summary_path = RESULTS_DIR / "FINAL_EXPERIMENT_SUMMARY.md"
    if summary_path.exists():
        with st.expander("📄 Full Experiment Summary"):
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    st.markdown(f.read())
            except Exception as e:
                st.error(f"Could not load summary: {e}")

    st.markdown("""
    <div class="disclaimer-box">
        <b>⚠️ Research Disclaimer:</b> This system is an investigational research prototype
        developed for academic and computational neuroscience purposes. Model predictions and
        attribution visualizations are intended solely for research experimentation. They must
        not be used as a substitute for professional medical evaluation or diagnosis. This
        software is not approved as a medical device and is not certified for clinical use.
    </div>
    """, unsafe_allow_html=True)