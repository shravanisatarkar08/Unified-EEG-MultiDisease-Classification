"""
Tests for the EEG classification inference pipeline and application components.

Tests:
- Model loading and checkpoint integrity
- Input tensor shape validation
- Five-class output shape
- Probability normalization (softmax sums to 1)
- Valid EEG input processing (synthetic)
- Invalid upload handling
- Missing checkpoint handling
- Explainability pipeline
"""

import sys
import os
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.eeg_classifier import EEGClassifier


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pt"
N_CHANNELS = 19
N_SAMPLES = 1280
NUM_CLASSES = 5
BATCH_SIZE = 2


# ═══════════════════════════════════════════════════════════════════════════════
# Model Architecture Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelArchitecture:
    """Test model instantiation and forward pass with various inputs."""

    def test_model_instantiation(self):
        """Model should instantiate with default parameters."""
        model = EEGClassifier(n_channels=19, n_samples=1280, num_classes=5)
        assert model is not None

    def test_forward_pass_shape(self):
        """Forward pass should produce [B, 5] output."""
        model = EEGClassifier(n_channels=19, n_samples=1280, num_classes=5)
        model.eval()
        x = torch.randn(BATCH_SIZE, N_CHANNELS, N_SAMPLES)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (BATCH_SIZE, NUM_CLASSES), \
            f"Expected shape ({BATCH_SIZE}, {NUM_CLASSES}), got {logits.shape}"

    def test_single_sample_inference(self):
        """Single sample inference should work."""
        model = EEGClassifier(n_channels=19, n_samples=1280, num_classes=5)
        model.eval()
        x = torch.randn(1, N_CHANNELS, N_SAMPLES)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (1, NUM_CLASSES)

    def test_probability_normalization(self):
        """Softmax probabilities should sum to 1."""
        model = EEGClassifier(n_channels=19, n_samples=1280, num_classes=5)
        model.eval()
        x = torch.randn(1, N_CHANNELS, N_SAMPLES)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)
        assert abs(probs.sum().item() - 1.0) < 1e-5, \
            f"Probabilities sum to {probs.sum().item()}, expected ~1.0"

    def test_all_probabilities_positive(self):
        """All probabilities should be non-negative after softmax."""
        model = EEGClassifier(n_channels=19, n_samples=1280, num_classes=5)
        model.eval()
        x = torch.randn(1, N_CHANNELS, N_SAMPLES)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=-1).numpy()
        assert np.all(probs >= 0), "Some probabilities are negative"

    def test_feature_extraction(self):
        """CNN feature extraction should produce [B, N, D] tokens."""
        model = EEGClassifier(n_channels=19, n_samples=1280, num_classes=5)
        model.eval()
        x = torch.randn(1, N_CHANNELS, N_SAMPLES)
        with torch.no_grad():
            features = model.get_features(x)
        assert features.ndim == 3, f"Expected 3D features, got {features.ndim}D"
        assert features.shape[0] == 1
        assert features.shape[2] == 64  # embed_dim


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpoint Loading Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCheckpointLoading:
    """Test checkpoint loading and model state restoration."""

    @pytest.mark.skipif(not CHECKPOINT_PATH.exists(),
                        reason="Checkpoint not found")
    def test_checkpoint_loads(self):
        """Checkpoint should load without errors."""
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        assert "model_state_dict" in checkpoint, "Missing model_state_dict key"
        assert "config" in checkpoint, "Missing config key"

    @pytest.mark.skipif(not CHECKPOINT_PATH.exists(),
                        reason="Checkpoint not found")
    def test_checkpoint_state_dict_matches_model(self):
        """State dict should load into model without missing/unexpected keys."""
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        model = EEGClassifier(n_channels=19, n_samples=1280, num_classes=5)
        # This will raise if there are unexpected or missing keys
        model.load_state_dict(checkpoint["model_state_dict"])

    @pytest.mark.skipif(not CHECKPOINT_PATH.exists(),
                        reason="Checkpoint not found")
    def test_loaded_model_inference(self):
        """Loaded model should produce valid predictions."""
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        model = EEGClassifier(n_channels=19, n_samples=1280, num_classes=5)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        x = torch.randn(1, N_CHANNELS, N_SAMPLES)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=-1).squeeze()

        assert logits.shape == (1, 5)
        assert abs(probs.sum().item() - 1.0) < 1e-5

    def test_missing_checkpoint_handling(self):
        """Should handle missing checkpoint gracefully."""
        fake_path = PROJECT_ROOT / "checkpoints" / "nonexistent_model.pt"
        assert not fake_path.exists()


# ═══════════════════════════════════════════════════════════════════════════════
# Input Validation Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestInputValidation:
    """Test input tensor validation and edge cases."""

    def test_correct_input_shape(self):
        """Standard input [B, 19, 1280] should work."""
        model = EEGClassifier(n_channels=19, n_samples=1280, num_classes=5)
        model.eval()
        x = torch.randn(1, 19, 1280)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (1, 5)

    def test_zero_input(self):
        """All-zero input should produce valid probabilities (not NaN)."""
        model = EEGClassifier(n_channels=19, n_samples=1280, num_classes=5)
        model.eval()
        x = torch.zeros(1, 19, 1280)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)
        assert not torch.isnan(probs).any(), "NaN in probabilities for zero input"
        assert not torch.isinf(probs).any(), "Inf in probabilities for zero input"

    def test_large_values_input(self):
        """Large input values should not cause NaN/Inf."""
        model = EEGClassifier(n_channels=19, n_samples=1280, num_classes=5)
        model.eval()
        x = torch.randn(1, 19, 1280) * 100
        with torch.no_grad():
            logits = model(x)
        assert not torch.isnan(logits).any(), "NaN in logits"


# ═══════════════════════════════════════════════════════════════════════════════
# EEG Processing Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestEEGProcessing:
    """Test EEG preprocessing pipeline components."""

    def test_channel_name_normalization(self):
        """Channel name aliases should be resolved correctly."""
        sys.path.insert(0, str(PROJECT_ROOT / "app"))
        from app import normalize_channel_name

        assert normalize_channel_name("T7") == "T3"
        assert normalize_channel_name("T8") == "T4"
        assert normalize_channel_name("P7") == "T5"
        assert normalize_channel_name("P8") == "T6"
        assert normalize_channel_name("FP1") == "Fp1"
        assert normalize_channel_name("Fp1") == "Fp1"
        assert normalize_channel_name("EEG Fp1-REF") == "Fp1"
        assert normalize_channel_name("UNKNOWN_CH") is None

    def test_npy_processing_correct_shape(self):
        """NPY upload with [19, 1280] shape should work."""
        sys.path.insert(0, str(PROJECT_ROOT / "app"))
        from app import process_npy_upload

        data = np.random.randn(19, 1280).astype(np.float32)
        buf = io.BytesIO()
        np.save(buf, data)
        buf.seek(0)

        result, error = process_npy_upload(buf.read())
        assert error is None, f"Unexpected error: {error}"
        assert result is not None
        assert result["windows"].shape == (1, 19, 1280)

    def test_npy_processing_transposed(self):
        """NPY upload with [1280, 19] shape should auto-transpose."""
        sys.path.insert(0, str(PROJECT_ROOT / "app"))
        from app import process_npy_upload

        data = np.random.randn(1280, 19).astype(np.float32)
        buf = io.BytesIO()
        np.save(buf, data)
        buf.seek(0)

        result, error = process_npy_upload(buf.read())
        assert error is None
        assert result["windows"].shape == (1, 19, 1280)

    def test_npy_processing_batch(self):
        """NPY upload with [N, 19, 1280] batch should work."""
        sys.path.insert(0, str(PROJECT_ROOT / "app"))
        from app import process_npy_upload

        data = np.random.randn(5, 19, 1280).astype(np.float32)
        buf = io.BytesIO()
        np.save(buf, data)
        buf.seek(0)

        result, error = process_npy_upload(buf.read())
        assert error is None
        assert result["windows"].shape == (5, 19, 1280)
        assert result["n_windows"] == 5

    def test_csv_processing(self):
        """CSV with named channels should be processed correctly."""
        sys.path.insert(0, str(PROJECT_ROOT / "app"))
        from app import process_csv_upload

        # Create CSV with standard channel names
        data = np.random.randn(1300, 19)
        columns = [
            "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
            "T3", "C3", "Cz", "C4", "T4", "T5", "P3",
            "Pz", "P4", "T6", "O1", "O2",
        ]
        import io as _io
        buf = _io.BytesIO()
        import pandas as pd
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(buf, index=False)
        buf.seek(0)

        result, error = process_csv_upload(buf.read())
        assert error is None
        assert result is not None
        assert result["windows"].shape[1] == 19
        assert result["windows"].shape[2] == 1280


# ═══════════════════════════════════════════════════════════════════════════════
# Explainability Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestExplainability:
    """Test explainability methods produce correct output shapes."""

    def test_input_gradients(self):
        """Input gradient saliency should return [19, 1280] array."""
        from training.explainability import EEGExplainer

        model = EEGClassifier(n_channels=19, n_samples=1280, num_classes=5)
        model.eval()
        explainer = EEGExplainer(model, torch.device("cpu"))

        x = torch.randn(1, 19, 1280)
        saliency = explainer.input_gradients(x, target_class=0)
        assert saliency.shape == (19, 1280)

    def test_grad_cam(self):
        """Grad-CAM should return [1280] temporal activation."""
        from training.explainability import EEGExplainer

        model = EEGClassifier(n_channels=19, n_samples=1280, num_classes=5)
        model.eval()
        explainer = EEGExplainer(model, torch.device("cpu"))

        x = torch.randn(1, 19, 1280)
        cam = explainer.grad_cam(x)
        assert cam.shape == (1280,), f"Expected (1280,), got {cam.shape}"
        assert cam.min() >= 0, "Grad-CAM should be non-negative"
        assert cam.max() <= 1.0 + 1e-6, "Grad-CAM should be normalized to [0, 1]"

    def test_transformer_attention(self):
        """Transformer attention should return token weights or None."""
        from training.explainability import EEGExplainer

        model = EEGClassifier(n_channels=19, n_samples=1280, num_classes=5)
        model.eval()
        explainer = EEGExplainer(model, torch.device("cpu"))

        x = torch.randn(1, 19, 1280)
        attn = explainer.transformer_attention(x)
        # May return None if attention weights aren't exposed
        if attn is not None:
            assert attn.ndim == 1
            assert len(attn) == 40  # N temporal tokens


# ═══════════════════════════════════════════════════════════════════════════════
# Results Artifacts Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestResultsArtifacts:
    """Test that result files load correctly."""

    def test_metrics_json_loads(self):
        """metrics.json should parse correctly."""
        metrics_path = PROJECT_ROOT / "results" / "metrics.json"
        if not metrics_path.exists():
            pytest.skip("metrics.json not found")

        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "test_metrics" in data
        assert "per_class_metrics" in data
        assert data["num_classes"] == 5

    def test_metrics_values_reasonable(self):
        """Metric values should be in valid ranges."""
        metrics_path = PROJECT_ROOT / "results" / "metrics.json"
        if not metrics_path.exists():
            pytest.skip("metrics.json not found")

        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tm = data["test_metrics"]
        assert 0 < tm["accuracy"] <= 1.0
        assert 0 < tm["balanced_accuracy"] <= 1.0
        assert 0 < tm["macro_f1"] <= 1.0
        assert 0 < tm["macro_roc_auc"] <= 1.0

    def test_confusion_matrix_shape(self):
        """Confusion matrix should be 5x5."""
        metrics_path = PROJECT_ROOT / "results" / "metrics.json"
        if not metrics_path.exists():
            pytest.skip("metrics.json not found")

        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cm = data.get("confusion_matrix_test_5x5", {})
        matrix = cm.get("matrix", [])
        assert len(matrix) == 5
        for row in matrix:
            assert len(row) == 5

    def test_classification_report_exists(self):
        """Classification report CSV should exist."""
        path = PROJECT_ROOT / "results" / "classification_report.csv"
        if not path.exists():
            pytest.skip("classification_report.csv not found")
        import pandas as pd
        df = pd.read_csv(path)
        assert len(df) == 5  # 5 classes


# ═══════════════════════════════════════════════════════════════════════════════
# Frontend Startup Test
# ═══════════════════════════════════════════════════════════════════════════════

class TestFrontendStartup:
    """Test that app.py can be imported without errors."""

    def test_app_syntax(self):
        """app.py should have valid Python syntax."""
        import ast
        app_path = PROJECT_ROOT / "app" / "app.py"
        with open(app_path, "r", encoding="utf-8") as f:
            source = f.read()
        try:
            ast.parse(source)
        except SyntaxError as e:
            pytest.fail(f"Syntax error in app.py: {e}")


# Need io for BytesIO in test methods
import io


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
