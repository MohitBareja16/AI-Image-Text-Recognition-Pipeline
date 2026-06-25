# ─────────────────────────────────────────────────────────────────────────────
# tests/test_config_validation.py — Unit Tests: Config Validation (TEST_PLAN §10)
# ─────────────────────────────────────────────────────────────────────────────

import pytest
import src.config as config
from src.config import validate_config


# TC-CFG-001: Valid default config passes validation
def test_valid_config_passes(monkeypatch):
    """Default config.py values must pass all validations."""
    monkeypatch.setattr(config, "PIPELINE_MODE", "ocr")
    monkeypatch.setattr(config, "CONFIDENCE_THRESHOLD", 0.80)
    monkeypatch.setattr(config, "GAUSSIAN_BLUR_KERNEL", (5, 5))
    monkeypatch.setattr(config, "TESSERACT_PSM", 3)
    monkeypatch.setattr(config, "SAVE_OUTPUT_IMAGE", True)
    validate_config()  # Must not raise


# TC-CFG-002: PIPELINE_MODE invalid string fails
def test_invalid_pipeline_mode(monkeypatch):
    monkeypatch.setattr(config, "PIPELINE_MODE", "video")
    with pytest.raises((AssertionError, ValueError)):
        validate_config()


# TC-CFG-003: CONFIDENCE_THRESHOLD below 0.80 fails
def test_threshold_below_80_fails(monkeypatch):
    monkeypatch.setattr(config, "PIPELINE_MODE", "ocr")
    monkeypatch.setattr(config, "CONFIDENCE_THRESHOLD", 0.79)
    with pytest.raises((AssertionError, ValueError)):
        validate_config()


# TC-CFG-004: CONFIDENCE_THRESHOLD exactly 0.80 passes
def test_threshold_exactly_80_passes(monkeypatch):
    monkeypatch.setattr(config, "PIPELINE_MODE", "ocr")
    monkeypatch.setattr(config, "CONFIDENCE_THRESHOLD", 0.80)
    monkeypatch.setattr(config, "SAVE_OUTPUT_IMAGE", True)
    validate_config()  # Must not raise


# TC-CFG-005: SAVE_OUTPUT_IMAGE = False fails
def test_save_output_false_fails(monkeypatch):
    monkeypatch.setattr(config, "PIPELINE_MODE", "ocr")
    monkeypatch.setattr(config, "CONFIDENCE_THRESHOLD", 0.80)
    monkeypatch.setattr(config, "SAVE_OUTPUT_IMAGE", False)
    with pytest.raises((AssertionError, ValueError)):
        validate_config()


# TC-CFG-006: Even GAUSSIAN_BLUR_KERNEL fails
def test_even_kernel_fails(monkeypatch):
    monkeypatch.setattr(config, "PIPELINE_MODE", "ocr")
    monkeypatch.setattr(config, "CONFIDENCE_THRESHOLD", 0.80)
    monkeypatch.setattr(config, "GAUSSIAN_BLUR_KERNEL", (4, 4))
    monkeypatch.setattr(config, "SAVE_OUTPUT_IMAGE", True)
    with pytest.raises((AssertionError, ValueError)):
        validate_config()


# TC-CFG-007: TESSERACT_PSM out of range (14) fails
def test_psm_out_of_range_fails(monkeypatch):
    monkeypatch.setattr(config, "PIPELINE_MODE", "ocr")
    monkeypatch.setattr(config, "CONFIDENCE_THRESHOLD", 0.80)
    monkeypatch.setattr(config, "GAUSSIAN_BLUR_KERNEL", (5, 5))
    monkeypatch.setattr(config, "TESSERACT_PSM", 14)
    monkeypatch.setattr(config, "SAVE_OUTPUT_IMAGE", True)
    with pytest.raises((AssertionError, ValueError)):
        validate_config()


# TC-CFG-008: TESSERACT_PSM = 0 is valid
def test_psm_0_valid(monkeypatch):
    monkeypatch.setattr(config, "PIPELINE_MODE", "ocr")
    monkeypatch.setattr(config, "CONFIDENCE_THRESHOLD", 0.80)
    monkeypatch.setattr(config, "GAUSSIAN_BLUR_KERNEL", (5, 5))
    monkeypatch.setattr(config, "TESSERACT_PSM", 0)
    monkeypatch.setattr(config, "SAVE_OUTPUT_IMAGE", True)
    validate_config()  # Must not raise


# TC-CFG-009: TESSERACT_PSM = 13 is valid
def test_psm_13_valid(monkeypatch):
    monkeypatch.setattr(config, "PIPELINE_MODE", "ocr")
    monkeypatch.setattr(config, "CONFIDENCE_THRESHOLD", 0.80)
    monkeypatch.setattr(config, "GAUSSIAN_BLUR_KERNEL", (5, 5))
    monkeypatch.setattr(config, "TESSERACT_PSM", 13)
    monkeypatch.setattr(config, "SAVE_OUTPUT_IMAGE", True)
    validate_config()  # Must not raise


# TC-CFG-010: CONFIDENCE_THRESHOLD = 1.0 is valid
def test_threshold_1_0_valid(monkeypatch):
    monkeypatch.setattr(config, "PIPELINE_MODE", "ocr")
    monkeypatch.setattr(config, "CONFIDENCE_THRESHOLD", 1.0)
    monkeypatch.setattr(config, "SAVE_OUTPUT_IMAGE", True)
    validate_config()  # Must not raise


# TC-CFG-011: CONFIDENCE_THRESHOLD value is auditable from source
def test_confidence_threshold_auditable():
    """Gate 3 requires CONFIDENCE_THRESHOLD to exist as a named constant."""
    assert hasattr(config, "CONFIDENCE_THRESHOLD"), \
        "CONFIDENCE_THRESHOLD constant not found in config.py"
    assert config.CONFIDENCE_THRESHOLD == 0.80, \
        f"CONFIDENCE_THRESHOLD must be 0.80, got {config.CONFIDENCE_THRESHOLD}"


# TC-CFG-012: PIPELINE_MODE exists as constant
def test_pipeline_mode_constant_exists():
    assert hasattr(config, "PIPELINE_MODE")
    assert config.PIPELINE_MODE in {"ocr", "detection"}


# TC-CFG-013: All model paths are string types
def test_model_paths_are_strings():
    assert isinstance(config.PROTOTXT_PATH, str)
    assert isinstance(config.CAFFEMODEL_PATH, str)
    assert isinstance(config.LABELS_PATH, str)
    assert isinstance(config.OUTPUT_DIR, str)
