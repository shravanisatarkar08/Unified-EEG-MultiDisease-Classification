"""
Unified Explainable CNN-Transformer Framework for Multi-Disease EEG Classification
Interactive Research Demonstration Dashboard

Run with:
    python -m streamlit run app/app.py
"""

import sys
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch


# =============================================================================
# PROJECT PATH
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# PROJECT IMPORTS
# =============================================================================

from models.eeg_classifier import EEGClassifier


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Unified EEG Multi-Disease AI Framework",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown(
    """
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
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# CONSTANTS
# =============================================================================

DISEASE_NAMES = [
    "Healthy Control",
    "Epilepsy",
    "Alzheimer's Disease",
    "Parkinson's Disease",
    "Depression",
]

CLASS_KEYS = [
    "healthy",
    "epilepsy",
    "alzheimers",
    "parkinsons",
    "depression",
]

CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4", "T5", "P3",
    "Pz", "P4", "T6", "O1", "O2"
]

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pt"
METRICS_PATH = PROJECT_ROOT / "results" / "metrics.json"
RESULTS_DIR = PROJECT_ROOT / "results"
INDEX_PATH = PROJECT_ROOT / "datasets" / "metadata" / "model_dataset_index.csv"


# =============================================================================
# MODEL LOADING
# =============================================================================

@st.cache_resource
def load_cached_model():
    """Load the final trained 5-class model."""

    if not CHECKPOINT_PATH.exists():
        return None, "Checkpoint not found"

    try:
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location="cpu",
            weights_only=False,
        )

        cfg = checkpoint.get("config", {})

        if isinstance(cfg, dict):
            embed_dim = cfg.get("embed_dim", 64)
            num_heads = cfg.get("num_heads", 4)
            num_layers = cfg.get("num_layers", 2)
            ff_dim = cfg.get("ff_dim", 128)
        else:
            embed_dim = getattr(cfg, "embed_dim", 64)
            num_heads = getattr(cfg, "num_heads", 4)
            num_layers = getattr(cfg, "num_layers", 2)
            ff_dim = getattr(cfg, "ff_dim", 128)

        model = EEGClassifier(
            n_channels=19,
            n_samples=1280,
            num_classes=5,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        return model, "Loaded best_model.pt"

    except Exception as exc:
        return None, f"Error loading model: {exc}"


# =============================================================================
# METRICS LOADING
# =============================================================================

@st.cache_data
def load_cached_metrics():
    """Load final experiment metrics."""

    if not METRICS_PATH.exists():
        return None

    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return None


model, model_status = load_cached_model()
metrics_data = load_cached_metrics()


# =============================================================================
# HEADER
# =============================================================================

st.markdown(
    '<div class="main-title">'
    '🧠 Unified Explainable CNN-Transformer Framework'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sub-title">'
    'Multi-Disease Scalp EEG Classification & Explainability Research System'
    '</div>',
    unsafe_allow_html=True,
)


# =============================================================================
# NAVIGATION
# =============================================================================

nav_tabs = st.tabs(
    [
        "🏠 Home",
        "🎯 Live EEG Prediction",
        "📊 Model Performance",
        "🔍 Grad-CAM Explainability",
        "🗃️ Datasets & Cohorts",
        "🔬 Methodology",
        "📈 Results & Metrics",
        "⚠️ Limitations & Scope",
    ]
)


# =============================================================================
# TAB 1 — HOME
# =============================================================================

with nav_tabs[0]:

    st.header("Unified Multi-Disease Scalp EEG AI Framework")

    st.markdown(
        """
        This research platform demonstrates a unified CNN-Transformer
        framework for classification of five EEG categories:

        **Healthy, Epilepsy, Alzheimer's Disease, Parkinson's Disease,
        and Depression.**

        The system combines convolutional feature extraction with
        Transformer-based temporal modelling and provides model-based
        visual attribution using Grad-CAM.
        """
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-val">0</div>'
            '<div class="metric-lbl">Healthy</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-val">1</div>'
            '<div class="metric-lbl">Epilepsy</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-val">2</div>'
            '<div class="metric-lbl">Alzheimer\'s</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-val">3</div>'
            '<div class="metric-lbl">Parkinson\'s</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with c5:
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-val">4</div>'
            '<div class="metric-lbl">Depression</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.subheader("Final Experiment Highlights")

    st.markdown(
        """
        - **5-class EEG classification**
        - **19-channel standardized EEG representation**
        - **256 Hz target sampling rate**
        - **5-second windows with 50% overlap**
        - **7 independent CHB-MIT epilepsy subjects across train, validation, and test partitions**
        - **Strict subject-level partitioning**
        - **Zero subject-level leakage**
        - **82.49% held-out test accuracy**
        - **86.15% balanced accuracy**
        - **0.8518 macro F1**
        - **0.9648 macro ROC-AUC**
        """
    )

    st.markdown(
        """
        <div class="disclaimer-box">
        <b>⚠️ Investigational Research Prototype:</b>
        This software is developed for computational neuroscience and
        academic research purposes. It is not approved as a medical
        device and is not certified for clinical diagnosis.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# TAB 2 — LIVE EEG PREDICTION
# =============================================================================

with nav_tabs[1]:

    st.header("5-Class EEG Inference Demonstration")

    st.markdown(
        """
        Select a representative preprocessed EEG window from the project
        dataset and run the trained model.
        """
    )

    if model is None:

        st.error(
            f"Model checkpoint could not be loaded: {model_status}"
        )

    else:

        st.success(
            f"✅ Active Model: checkpoints/best_model.pt — {model_status}"
        )

        sample_options = [
            "Healthy Control (OpenNeuro ds004504)",
            "Epilepsy Patient (CHB-MIT chb01)",
            "Alzheimer's Disease Patient (OpenNeuro ds004504)",
            "Parkinson's Disease Patient (OpenNeuro ds004584)",
            "Depression Patient (NEMAR nm000114 / MODMA)",
        ]

        selected_sample_type = st.selectbox(
            "Select Benchmark Sample Condition:",
            sample_options,
        )

        window_tensor = None
        sample_meta = {}

        if INDEX_PATH.exists():

            try:

                idx_df = pd.read_csv(INDEX_PATH)

                cls_map_lookup = {
                    "Healthy Control (OpenNeuro ds004504)": 0,
                    "Epilepsy Patient (CHB-MIT chb01)": 1,
                    "Alzheimer's Disease Patient (OpenNeuro ds004504)": 2,
                    "Parkinson's Disease Patient (OpenNeuro ds004584)": 3,
                    "Depression Patient (NEMAR nm000114 / MODMA)": 4,
                }

                target_cls = cls_map_lookup[selected_sample_type]

                matches = idx_df[
                    idx_df["class_label_idx"] == target_cls
                ]

                if len(matches) > 0:

                    row = matches.iloc[0]

                    source_file = Path(row["source_file"])

                    if not source_file.is_absolute():
                        source_file = PROJECT_ROOT / source_file

                    if source_file.exists():

                        window_idx = int(row["window_idx"])

                        if source_file.suffix.lower() == ".npy":

                            arr = np.load(
                                source_file,
                                mmap_mode="r",
                            )

                            if window_idx < len(arr):

                                win = np.array(
                                    arr[window_idx],
                                    dtype=np.float32,
                                )

                                window_tensor = torch.tensor(
                                    win
                                ).unsqueeze(0)

                                sample_meta = row.to_dict()

            except Exception as exc:

                st.warning(
                    f"Could not load sample index: {exc}"
                )

        if window_tensor is not None:

            st.info(
                f"Dataset: `{sample_meta.get('dataset_name', 'dataset')}` "
                f"| Subject: `{sample_meta.get('subject_id', 'unknown')}` "
                f"| Split: `{sample_meta.get('split', 'unknown')}`"
            )

            # -----------------------------------------------------------------
            # ROBUST EEG PLOT
            # -----------------------------------------------------------------

            st.subheader("Representative EEG Window")

            fig, ax = plt.subplots(
                figsize=(12, 4),
                dpi=100,
            )

            win_np = np.asarray(
                window_tensor.squeeze().numpy(),
                dtype=np.float32,
            )

            if win_np.ndim == 1:
                win_np = win_np.reshape(1, -1)

            win_np = np.squeeze(win_np)

            if win_np.ndim == 1:
                win_np = win_np.reshape(1, -1)

            # Convert [samples, channels] to [channels, samples]
            if (
                win_np.ndim == 2
                and win_np.shape[0] == 1280
                and win_np.shape[1] == 19
            ):
                win_np = win_np.T

            n_channels = min(5, win_np.shape[0])
            n_samples = win_np.shape[1]

            time_axis = np.linspace(
                0,
                5,
                n_samples,
            )

            for ch in range(n_channels):

                signal = np.asarray(
                    win_np[ch]
                ).reshape(-1)

                ax.plot(
                    time_axis[:len(signal)],
                    signal + (4 - ch) * 3,
                    lw=1.0,
                    label=CHANNELS[ch],
                )

            ax.set_yticks(
                [
                    (4 - ch) * 3
                    for ch in range(n_channels)
                ]
            )

            ax.set_yticklabels(
                CHANNELS[:n_channels],
                fontweight="bold",
            )

            ax.set_xlabel("Time (seconds)")
            ax.set_title(
                "Standardized EEG Waveforms",
                fontweight="bold",
            )

            ax.grid(
                True,
                alpha=0.3,
            )

            ax.legend(
                loc="upper right",
                ncol=2,
            )

            st.pyplot(fig)

            plt.close(fig)

            # -----------------------------------------------------------------
            # INFERENCE
            # -----------------------------------------------------------------

            if st.button(
                "🚀 Run 5-Class Inference",
                type="primary",
            ):

                with torch.no_grad():

                    # CNN expects 4D input: [batch, channels, height, samples]
                    if window_tensor.ndim == 3:
                        window_tensor = window_tensor.unsqueeze(2)

                    logits = model(window_tensor)

                    probabilities = torch.softmax(
                        logits,
                        dim=-1,
                    ).squeeze(0).numpy()

                prediction_index = int(
                    np.argmax(probabilities)
                )

                st.markdown(
                    "### Model Prediction"
                )

                p_col1, p_col2 = st.columns(
                    [1, 2]
                )

                with p_col1:

                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-val">
                                {DISEASE_NAMES[prediction_index]}
                            </div>
                            <div class="metric-lbl">
                                Model Prediction
                                ({probabilities[prediction_index] * 100:.1f}%)
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with p_col2:

                    probability_df = pd.DataFrame(
                        {
                            "Condition": DISEASE_NAMES,
                            "Probability (%)": np.round(
                                probabilities * 100,
                                2,
                            ),
                        }
                    )

                    st.bar_chart(
                        probability_df.set_index(
                            "Condition"
                        )
                    )

                st.info(
                    "This prediction is a research-model output and "
                    "must not be interpreted as a clinical diagnosis."
                )

        else:

            st.warning(
                "The selected representative EEG window "
                "could not be loaded from the dataset index."
            )


# =============================================================================
# TAB 3 — MODEL PERFORMANCE
# =============================================================================

with nav_tabs[2]:

    st.header(
        "Held-Out Test Set Performance"
    )

    if metrics_data:

        test_metrics = metrics_data.get(
            "test_metrics",
            {},
        )

        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-val">
                        {test_metrics.get("accuracy", 0) * 100:.2f}%
                    </div>
                    <div class="metric-lbl">
                        Test Accuracy
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c2:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-val">
                        {test_metrics.get("balanced_accuracy", 0) * 100:.2f}%
                    </div>
                    <div class="metric-lbl">
                        Balanced Accuracy
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c3:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-val">
                        {test_metrics.get("macro_f1", 0):.4f}
                    </div>
                    <div class="metric-lbl">
                        Macro F1
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c4:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-val">
                        {test_metrics.get("macro_roc_auc", 0):.4f}
                    </div>
                    <div class="metric-lbl">
                        Macro ROC-AUC
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c5:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-val">
                        {test_metrics.get("cohen_kappa", 0):.4f}
                    </div>
                    <div class="metric-lbl">
                        Cohen's Kappa
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        st.subheader(
            "Additional Test Metrics"
        )

        additional_metrics = pd.DataFrame(
            {
                "Metric": [
                    "Accuracy",
                    "Balanced Accuracy",
                    "Macro F1",
                    "Weighted F1",
                    "Macro ROC-AUC",
                    "Cohen's Kappa",
                ],
                "Value": [
                    test_metrics.get("accuracy", 0),
                    test_metrics.get("balanced_accuracy", 0),
                    test_metrics.get("macro_f1", 0),
                    test_metrics.get("weighted_f1", 0),
                    test_metrics.get("macro_roc_auc", 0),
                    test_metrics.get("cohen_kappa", 0),
                ],
            }
        )

        st.dataframe(
            additional_metrics,
            use_container_width=True,
        )

    else:

        st.error(
            "results/metrics.json could not be loaded."
        )

    st.markdown("---")

    pcol1, pcol2 = st.columns(2)

    with pcol1:

        st.subheader(
            "5-Class Confusion Matrix"
        )

        cm_path = RESULTS_DIR / "confusion_matrix.png"

        if cm_path.exists():

            st.image(
                str(cm_path),
                caption="Held-out Test Set Confusion Matrix — 1,525 Test Windows",
                use_container_width=True,
            )

        else:

            st.warning(
                "Confusion matrix image not found."
            )

    with pcol2:

        st.subheader(
            "Training & Validation Curves"
        )

        curves_path = (
            RESULTS_DIR / "training_curves.png"
        )

        if curves_path.exists():

            st.image(
                str(curves_path),
                caption="Training and validation trajectories",
                use_container_width=True,
            )

        else:

            st.warning(
                "Training curves image not found."
            )

    st.markdown("---")

    st.subheader(
        "Per-Class Classification Report"
    )

    report_path = (
        RESULTS_DIR / "classification_report.csv"
    )

    if report_path.exists():

        report_df = pd.read_csv(
            report_path
        )

        st.dataframe(
            report_df,
            use_container_width=True,
        )

    else:

        st.warning(
            "Classification report not found."
        )


# =============================================================================
# TAB 4 — GRAD-CAM
# =============================================================================

with nav_tabs[3]:

    st.header(
        "Grad-CAM Explainability"
    )

    st.markdown(
        """
        The framework provides model-based attribution visualizations
        to inspect temporal EEG regions and channel-level contributions
        associated with model predictions.

        These visualizations represent model attribution and should not
        be interpreted as clinically established biomarkers.
        """
    )

    composite_path = (
        RESULTS_DIR / "gradcam_analysis.png"
    )

    if composite_path.exists():

        st.image(
            str(composite_path),
            caption="Grad-CAM Temporal Attribution and Electrode Saliency",
            use_container_width=True,
        )

    else:

        st.warning(
            "Grad-CAM composite image not found."
        )

    st.markdown("---")

    st.subheader(
        "Individual Class Attribution Profiles"
    )

    selected_class = st.selectbox(
        "Select Condition:",
        CLASS_KEYS,
    )

    individual_image = (
        RESULTS_DIR
        / "gradcam"
        / f"gradcam_{selected_class}.png"
    )

    if individual_image.exists():

        st.image(
            str(individual_image),
            caption=(
                f"Grad-CAM attribution profile — "
                f"{selected_class.capitalize()}"
            ),
            use_container_width=True,
        )

    else:

        st.info(
            "An individual Grad-CAM image for this class "
            "is not available in the results directory."
        )


# =============================================================================
# TAB 5 — DATASETS & COHORTS
# =============================================================================

with nav_tabs[4]:

    st.header(
        "Multi-Cohort Dataset Provenance"
    )

    st.markdown(
        """
        The final experiment combines EEG data from multiple open
        scientific cohorts and harmonizes the recordings into a
        common 19-channel, 256 Hz representation.
        """
    )

    dataset_summary = pd.DataFrame(
        [
            {
                "Condition": "Healthy Control",
                "Class ID": 0,
                "Source / Accession": (
                    "SRM + ds004504 + ds004584 + NEMAR"
                ),
                "Total Windows": 4110,
                "Train": 3090,
                "Val": 480,
                "Test": 540,
                "Split": "Subject-Level",
            },
            {
                "Condition": "Epilepsy",
                "Class ID": 1,
                "Source / Accession": "CHB-MIT",
                "Total Windows": 438,
                "Train": 326,
                "Val": 57,
                "Test": 55,
                "Split": "Subject-Level",
            },
            {
                "Condition": "Alzheimer's Disease",
                "Class ID": 2,
                "Source / Accession": "OpenNeuro ds004504",
                "Total Windows": 1080,
                "Train": 720,
                "Val": 180,
                "Test": 180,
                "Split": "Subject-Level",
            },
            {
                "Condition": "Parkinson's Disease",
                "Class ID": 3,
                "Source / Accession": "OpenNeuro ds004584",
                "Total Windows": 3000,
                "Train": 2070,
                "Val": 390,
                "Test": 540,
                "Split": "Subject-Level",
            },
            {
                "Condition": "Depression",
                "Class ID": 4,
                "Source / Accession": (
                    "NEMAR nm000114 / MODMA"
                ),
                "Total Windows": 1860,
                "Train": 1260,
                "Val": 390,
                "Test": 210,
                "Split": "Subject-Level",
            },
        ]
    )

    st.dataframe(
        dataset_summary,
        use_container_width=True,
    )

    st.markdown("---")

    st.subheader(
        "Final Dataset Totals"
    )

    total_columns = st.columns(4)

    totals = [
        ("Train Windows", "7,466"),
        ("Validation Windows", "1,497"),
        ("Test Windows", "1,525"),
        ("Total Windows", "10,488"),
    ]

    for column, (label, value) in zip(
        total_columns,
        totals,
    ):

        with column:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-val">
                        {value}
                    </div>
                    <div class="metric-lbl">
                        {label}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# =============================================================================
# TAB 6 — METHODOLOGY
# =============================================================================

with nav_tabs[5]:

    st.header(
        "Methodology & Pipeline Architecture"
    )

    st.markdown(
        """
        <div class="architecture-box">

        <b>End-to-End Processing Architecture</b>
        <br><br>

        Raw Multi-Channel Scalp EEG
        <br>
        ↓
        <b>Preprocessing & Harmonization</b>
        <br>
        0.5–45 Hz Bandpass + 50/60 Hz Notch + 256 Hz Resampling
        <br><br>

        ↓
        <b>Montage Standardization</b>
        <br>
        19 Standard EEG Channels
        <br><br>

        ↓
        <b>Segmentation</b>
        <br>
        5-Second Windows + 50% Overlap
        <br>
        1,280 samples per channel
        <br><br>

        ↓
        <b>CNN Feature Extraction</b>
        <br>
        EEGNet-inspired temporal and spatial feature extraction
        <br><br>

        ↓
        <b>Transformer Encoder</b>
        <br>
        Multi-Head Self-Attention
        <br>
        4 heads, 2 layers
        <br><br>

        ↓
        <b>Classification Head</b>
        <br>
        5 Classes
        <br>
        Healthy / Epilepsy / Alzheimer's /
        Parkinson's / Depression
        <br><br>

        ↓
        <b>Explainability</b>
        <br>
        Grad-CAM + Input Gradient Saliency

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.subheader(
        "Experimental Design"
    )

    st.markdown(
        """
        - Target sampling rate: **256 Hz**
        - Window duration: **5 seconds**
        - Window overlap: **50%**
        - Input representation: **19 × 1,280**
        - Subject-level partitioning
        - Independent train, validation, and test subjects
        - Five-class classification
        """
    )


# =============================================================================
# TAB 7 — RESULTS
# =============================================================================

with nav_tabs[6]:

    st.header(
        "Comprehensive Experimental Summary"
    )

    summary_path = (
        RESULTS_DIR / "FINAL_EXPERIMENT_SUMMARY.md"
    )

    if summary_path.exists():

        try:

            with open(
                summary_path,
                "r",
                encoding="utf-8",
            ) as file:

                summary_content = file.read()

            st.markdown(summary_content)

        except Exception as exc:

            st.error(
                f"Could not read final experiment summary: {exc}"
            )

    else:

        st.warning(
            "FINAL_EXPERIMENT_SUMMARY.md was not found."
        )


# =============================================================================
# TAB 8 — LIMITATIONS
# =============================================================================

with nav_tabs[7]:

    st.header(
        "Scientific Limitations & Scope"
    )

    st.markdown(
        """
        The following limitations should be considered when interpreting
        the final experiment.

        ### 1. Epilepsy Test Cohort

        CHB-MIT epilepsy recordings were separated by independent subject:

        - Training: **chb02, chb03, chb04, chb05, chb08**
        - Validation: **chb07**
        - Test: **chb01**

        The epilepsy test partition contains **55 windows from one
        independent held-out subject**.

        Therefore, the epilepsy result should be interpreted as
        performance on that held-out recording rather than broad
        clinical validation across epilepsy populations.

        ### 2. Multi-Cohort Healthy Reference

        Healthy controls originate from multiple cohorts, including
        SRM, OpenNeuro ds004504, OpenNeuro ds004584, and NEMAR.

        This provides a heterogeneous reference group but also introduces
        differences in acquisition protocols and cohort characteristics.

        ### 3. Alzheimer's Dataset

        Frontotemporal Dementia cases from ds004504 were excluded from
        the five-class experiment.

        ### 4. Subject-Level Leakage

        The final experiment uses subject-level partitioning and reports
        zero subject-level overlap between train, validation, and test
        partitions when subject identity is qualified by dataset.

        ### 5. Clinical Scope

        The model is a computational neuroscience research prototype.
        The reported results do not establish clinical diagnostic
        validity, clinical utility, or readiness for patient care.
        """
    )

    st.markdown(
        """
        <div class="disclaimer-box">
        <b>Research Disclaimer:</b>
        Model predictions and attribution visualizations are intended
        solely for academic research and computational experimentation.
        They must not be used as a substitute for professional medical
        evaluation or diagnosis.
        </div>
        """,
        unsafe_allow_html=True,
    )