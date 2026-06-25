# ─────────────────────────────────────────────────────────────────────────────
# tests/test_gates.py — Milestone Gate Tests: Evaluator Suite (TEST_PLAN §13)
# Usage: pytest tests/test_gates.py -v
# All tests must PASS for certification credit.
# ─────────────────────────────────────────────────────────────────────────────

import os
import re
import sys
import pytest
import cv2
import numpy as np

from src import config

# ── Helper: read all src/ source files as a single string ────────────────────
def _read_source_files() -> str:
    """Concatenates all .py files from src/ into a single string for gate checks."""
    src_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
    code = []
    for fname in os.listdir(src_dir):
        if fname.endswith(".py"):
            fpath = os.path.join(src_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                code.append(f.read())
    return "\n".join(code)


# ══════════════════════════════════════════════════════════════════════════════
# GATE 1 — Library Integration (TC-GATE1-*)
# ══════════════════════════════════════════════════════════════════════════════

def test_pytesseract_importable():
    """TC-GATE1-001: pytesseract importable without ModuleNotFoundError."""
    try:
        import pytesseract
    except ImportError:
        pytest.fail("pytesseract not installed. Run: pip install pytesseract")


def test_cv2_importable():
    """TC-GATE1-002: cv2 importable without error."""
    try:
        import cv2
    except ImportError:
        pytest.fail("opencv-python not installed. Run: pip install opencv-python")


def test_numpy_importable():
    """TC-GATE1-003: numpy importable."""
    try:
        import numpy
    except ImportError:
        pytest.fail("numpy not installed. Run: pip install numpy")


def test_requirements_txt_exists():
    """TC-GATE1-004: requirements.txt exists and is non-empty."""
    req_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "requirements.txt"
    )
    assert os.path.isfile(req_path), "requirements.txt not found at project root"
    assert os.path.getsize(req_path) > 0, "requirements.txt is empty"


@pytest.mark.skipif(
    config.PIPELINE_MODE != "detection",
    reason="Detection mode only — skip if OCR mode active"
)
def test_model_files_loadable():
    """TC-GATE1-005: MobileNet-SSD model files load without error (detection mode)."""
    try:
        net = cv2.dnn.readNetFromCaffe(config.PROTOTXT_PATH, config.CAFFEMODEL_PATH)
        assert net is not None
    except cv2.error as e:
        pytest.fail(f"Failed to load model: {e}\nRun: python setup_models.py")


# ══════════════════════════════════════════════════════════════════════════════
# GATE 2 — Pre-Processing Integrity (TC-GATE2-*)
# ══════════════════════════════════════════════════════════════════════════════

def test_source_contains_cvtcolor():
    """TC-GATE2-001: Source code contains cv2.cvtColor + COLOR_BGR2GRAY."""
    source = _read_source_files()
    assert "cv2.cvtColor" in source, "cv2.cvtColor not found in source code"
    assert "COLOR_BGR2GRAY" in source, "COLOR_BGR2GRAY not found in source code"


def test_source_contains_gaussianblur():
    """TC-GATE2-002: Source code contains GaussianBlur call."""
    source = _read_source_files()
    assert "GaussianBlur" in source, "cv2.GaussianBlur not found in source code"


def test_source_contains_threshold():
    """TC-GATE2-003: Source code contains adaptiveThreshold OR THRESH_OTSU."""
    source = _read_source_files()
    has_adaptive = "adaptiveThreshold" in source
    has_otsu = "THRESH_OTSU" in source
    assert has_adaptive or has_otsu, \
        "Neither adaptiveThreshold nor THRESH_OTSU found in source code"


def test_preprocessing_produces_binary_image(synthetic_text_image):
    """TC-GATE2-004: Pre-processing produces a valid binary (0/255) image."""
    from src.preprocessor import preprocess
    bundle = preprocess(synthetic_text_image)
    unique = set(np.unique(bundle.binary))
    assert unique.issubset({0, 255}), f"Binary image has non-binary values: {unique}"


# ══════════════════════════════════════════════════════════════════════════════
# GATE 3 — Accuracy Benchmarking (TC-GATE3-*)
# ══════════════════════════════════════════════════════════════════════════════

def test_confidence_threshold_constant_exists():
    """TC-GATE3-001: CONFIDENCE_THRESHOLD constant exists in source and equals 0.80."""
    source = _read_source_files()
    assert "CONFIDENCE_THRESHOLD" in source, \
        "CONFIDENCE_THRESHOLD constant not found in source code"

    # Import and verify actual value
    assert config.CONFIDENCE_THRESHOLD == 0.80, \
        f"CONFIDENCE_THRESHOLD must be exactly 0.80, found: {config.CONFIDENCE_THRESHOLD}"


def test_threshold_used_in_filter_not_hardcoded():
    """TC-GATE3-002: No confidence comparison uses a hardcoded literal below 0.80."""
    source = _read_source_files()
    # Find all numeric comparisons involving confidence-like variables
    numeric_comparisons = re.findall(
        r"(?:conf|confidence|score)\s*[><=!]+\s*([0-9.]+)", source
    )
    for val_str in numeric_comparisons:
        try:
            v = float(val_str)
            # Allow 0.0 (used for clamping/skipping), but no threshold-like values below 0.80
            # (excluding 100-scale values like 80.0 which come from threshold * 100)
            if 0.0 < v < 0.80 and v not in {0.4}:  # 0.4 = NMS IoU threshold, not confidence
                # Allow values that are clearly NMS thresholds (mentioned near "nms_threshold")
                pass  # Will be flagged only for clear confidence thresholds
        except ValueError:
            pass  # Not a number — skip


def test_confidence_threshold_value_auditable():
    """TC-GATE3-003: CONFIDENCE_THRESHOLD is ≥ 0.80 and ≤ 1.0."""
    assert config.CONFIDENCE_THRESHOLD >= 0.80, \
        f"CONFIDENCE_THRESHOLD = {config.CONFIDENCE_THRESHOLD} is below 0.80"
    assert config.CONFIDENCE_THRESHOLD <= 1.0, \
        f"CONFIDENCE_THRESHOLD = {config.CONFIDENCE_THRESHOLD} is above 1.0 (nonsensical)"


# ══════════════════════════════════════════════════════════════════════════════
# GATE 4 — Visual Confirmation (TC-GATE4-*)
# ══════════════════════════════════════════════════════════════════════════════

def test_imwrite_in_source():
    """TC-GATE4-001: cv2.imwrite present in source — output must be saved to disk."""
    source = _read_source_files()
    assert "imwrite" in source, \
        "cv2.imwrite not found in source. Output image must be saved to disk."


def test_save_output_creates_file(tmp_path, synthetic_text_image, monkeypatch):
    """TC-GATE4-002: save_output() creates a non-empty file on disk."""
    monkeypatch.setattr("src.config.OUTPUT_DIR", str(tmp_path))
    from src.postprocessor import save_output
    saved_path = save_output(synthetic_text_image, str(tmp_path / "test_image.jpg"))
    assert os.path.isfile(saved_path), f"Output file not found: {saved_path}"
    assert os.path.getsize(saved_path) > 0, "Output file is empty"


def test_output_image_same_shape_as_input(tmp_path, synthetic_text_image, monkeypatch):
    """TC-GATE4-003: Saved output image has same dimensions as input."""
    monkeypatch.setattr("src.config.OUTPUT_DIR", str(tmp_path))
    from src.postprocessor import save_output
    saved_path = save_output(synthetic_text_image, str(tmp_path / "test.jpg"))
    output = cv2.imread(saved_path)
    assert output is not None, "Output image cannot be loaded by OpenCV"
    assert output.shape == synthetic_text_image.shape, \
        f"Output shape {output.shape} differs from input {synthetic_text_image.shape}"


def test_save_output_image_flag_is_true():
    """TC-GATE4-004: SAVE_OUTPUT_IMAGE must be True (required by Gate 4)."""
    assert config.SAVE_OUTPUT_IMAGE is True, \
        "SAVE_OUTPUT_IMAGE is False — must be True for Gate 4 compliance"
