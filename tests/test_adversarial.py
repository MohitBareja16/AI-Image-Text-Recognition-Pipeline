# ─────────────────────────────────────────────────────────────────────────────
# tests/test_adversarial.py — Adversarial / Anti-Cheat Tests (TEST_PLAN §14)
# Tests specifically probe the 7 loopholes identified in PRD.md §12 plus
# additional gaming strategies documented in TEST_PLAN §14.
# ─────────────────────────────────────────────────────────────────────────────

import os
import re
import pytest
import numpy as np
import pandas as pd
import cv2

from src import config
from src.utils import scale_coordinates, validate_bbox_coords, compute_iou
from src.ocr_pipeline import filter_dataframe
from src.schemas import BBox, ObjectDetection
from src.detection_pipeline import _parse_detections


# ── Helper: read all src/ source files ───────────────────────────────────────
def _read_source_files() -> str:
    src_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
    code = []
    for fname in os.listdir(src_dir):
        if fname.endswith(".py"):
            with open(os.path.join(src_dir, fname), "r", encoding="utf-8") as f:
                code.append(f.read())
    return "\n".join(code)


# ── Mock COCO labels for detection tests ─────────────────────────────────────
_MOCK_LABELS = ["background"] + [f"class_{i}" for i in range(1, 91)]


# ══════════════════════════════════════════════════════════════════════════════
# Loophole 1 — image_to_string() bypass (TC-ADV-001)
# ══════════════════════════════════════════════════════════════════════════════

def test_image_to_string_not_used():
    """
    image_to_string() returns no per-word confidence data.
    Must use image_to_data() exclusively. (PRD §12 LOOPHOLE-02)
    """
    source = _read_source_files()
    assert "image_to_string" not in source, \
        ("image_to_string() detected in source. "
         "Must use image_to_data() to access per-word confidence scores.")


# ══════════════════════════════════════════════════════════════════════════════
# Loophole 2 — Output only shown, not saved (TC-ADV-002)
# ══════════════════════════════════════════════════════════════════════════════

def test_imwrite_in_source():
    """
    cv2.imwrite must be present — output must be saved to disk.
    (PRD §12 LOOPHOLE-03)
    """
    source = _read_source_files()
    assert "imwrite" in source, \
        "cv2.imwrite not found in source. Output must be saved to disk (Gate 4)."


# ══════════════════════════════════════════════════════════════════════════════
# Loophole 3 — Bounding boxes not scaled (TC-ADV-003)
# ══════════════════════════════════════════════════════════════════════════════

def test_coordinate_scaling_actually_multiplies():
    """
    If coordinates are not scaled, they remain near 0.0–1.0 as floats.
    Scaled pixel coordinates for a 640×480 image should be in 100–600 range.
    (PRD §12 LOOPHOLE-04)
    """
    raw = (0.2, 0.3, 0.7, 0.8)
    W, H = 640, 480
    bbox = scale_coordinates(raw, W, H)
    assert bbox[0] > 1, f"x1={bbox[0]} looks unscaled (near 0–1 float range)"
    assert bbox[2] > 1, f"x2={bbox[2]} looks unscaled"
    # Verify specific expected values
    assert bbox[0] == int(0.2 * 640), f"Expected x1={int(0.2*640)}, got {bbox[0]}"
    assert bbox[2] == int(0.7 * 640), f"Expected x2={int(0.7*640)}, got {bbox[2]}"


# ══════════════════════════════════════════════════════════════════════════════
# Loophole 4 — No input file type validation (TC-ADV-004)
# ══════════════════════════════════════════════════════════════════════════════

def test_text_file_rejected_cleanly(tmp_path):
    """
    A .txt file must be caught at validation, not mid-pipeline.
    (PRD §12 LOOPHOLE-05)
    """
    from src.utils import validate_image_path
    p = tmp_path / "fake_image.txt"
    p.write_text("this is not an image")
    result = validate_image_path(str(p))
    assert result is False, ".txt file must be rejected by validate_image_path"


def test_pdf_file_rejected_cleanly(tmp_path):
    """A .pdf file must be rejected."""
    from src.utils import validate_image_path
    p = tmp_path / "document.pdf"
    p.write_bytes(b"%PDF-1.4 fake content")
    assert validate_image_path(str(p)) is False


# ══════════════════════════════════════════════════════════════════════════════
# Loophole 5 — Pre-processing result not used as model input (TC-ADV-005)
# ══════════════════════════════════════════════════════════════════════════════

def test_preprocessing_binary_is_2d(synthetic_text_image):
    """
    Pre-processed image must be 2D (grayscale/binary), not 3D BGR.
    If 3D is passed to Tesseract, pre-processing was not actually used.
    (PRD §12 LOOPHOLE-06)
    """
    from src.preprocessor import preprocess
    bundle = preprocess(synthetic_text_image)
    assert bundle.binary.ndim == 2, \
        f"binary must be 2D (grayscale), got {bundle.binary.ndim}D"


# ══════════════════════════════════════════════════════════════════════════════
# Loophole 6 — Zero detections silent exit (TC-ADV-006)
# ══════════════════════════════════════════════════════════════════════════════

def test_zero_results_message_in_console(capsys):
    """
    When zero detections pass the threshold, explicit message must be printed.
    (PRD §12 LOOPHOLE-07, FR-09)
    """
    from src.postprocessor import print_summary
    from src.schemas import PipelineResult

    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    result = PipelineResult(
        mode="detection",
        input_path="/test.jpg",
        output_path="/output/test_output.jpg",
        detections=[],
        annotated_image=img,
        full_text=None,
        total_raw=3,
        total_accepted=0,
        total_rejected=3,
        runtime_seconds=0.5,
        threshold_used=0.80,
    )
    print_summary(result)
    captured = capsys.readouterr()
    assert "No high-confidence" in captured.out, \
        "Must print explicit 'No high-confidence' message for zero results (FR-09)"


# ══════════════════════════════════════════════════════════════════════════════
# Loophole 7 — Threshold lowered in source (TC-ADV-007)
# ══════════════════════════════════════════════════════════════════════════════

def test_confidence_threshold_value_in_source():
    """
    CONFIDENCE_THRESHOLD must be ≥ 0.80 in the actual runtime config.
    (PRD §12 LOOPHOLE-01, Gate 3 check)
    """
    assert config.CONFIDENCE_THRESHOLD >= 0.80, \
        f"CONFIDENCE_THRESHOLD = {config.CONFIDENCE_THRESHOLD} is below 0.80"
    assert config.CONFIDENCE_THRESHOLD <= 1.0, \
        f"CONFIDENCE_THRESHOLD = {config.CONFIDENCE_THRESHOLD} is above 1.0 — nonsensical"


# ══════════════════════════════════════════════════════════════════════════════
# Additional anti-gaming tests
# ══════════════════════════════════════════════════════════════════════════════

def test_model_loading_call_exists_in_source():
    """TC-ADV-008: Model loading call present — not a stub."""
    source = _read_source_files()
    assert "readNetFromCaffe" in source or "pytesseract" in source, \
        "No model loading call found — pipeline appears to be a stub."


def test_conf_neg1_rows_not_in_output(sample_tesseract_dataframe):
    """TC-ADV-009: Layout rows (conf=-1) must never appear as word detections."""
    result = filter_dataframe(sample_tesseract_dataframe)
    assert -1 not in result["conf"].values, \
        "conf=-1 layout rows found in filtered DataFrame"


def test_empty_strings_not_in_word_output(sample_tesseract_dataframe):
    """TC-ADV-010: Empty or whitespace-only text tokens must not appear."""
    result = filter_dataframe(sample_tesseract_dataframe)
    for text in result["text"]:
        assert len(str(text).strip()) > 0, \
            f"Empty text token in output: '{text}'"


def test_confidence_not_rounded_up():
    """
    TC-ADV-011: 0.7999 must NOT be rounded up to 0.80.
    Boundary is strict — 79.99 in Tesseract scale must be rejected.
    """
    df = pd.DataFrame({
        "level":    [5     ], "page_num": [1    ], "block_num":[1    ],
        "par_num":  [1     ], "line_num": [1    ], "word_num": [1    ],
        "left":     [10    ], "top":      [10   ], "width":    [50   ],
        "height":   [20    ], "conf":     [79.99], "text":     ["test"],
    })
    result = filter_dataframe(df)
    assert len(result) == 0, "79.99 must be rejected — not rounded up to 80"


def test_inverted_detection_skipped():
    """
    TC-ADV-012: Detection with inverted coordinates (x_start > x_end) must be silently skipped.
    (DATA_SCHEMA.md §3.1 malformed output handling)
    """
    raw = np.zeros((1, 1, 1, 7), dtype=np.float32)
    # x_start(0.8) > x_end(0.3) → inverted box
    raw[0, 0, 0] = [0, 15, 0.95, 0.8, 0.2, 0.3, 0.9]
    result = _parse_detections(raw, image_width=640, image_height=480, labels=_MOCK_LABELS)
    assert len(result) == 0, "Inverted bounding box should be silently discarded"


def test_background_class_never_in_results():
    """
    TC-ADV-013: class_id=0 (background) never appears in output, even at 0.99 confidence.
    (DATA_SCHEMA.md V-04)
    """
    raw = np.zeros((1, 1, 1, 7), dtype=np.float32)
    raw[0, 0, 0] = [0, 0, 0.99, 0.0, 0.0, 0.8, 0.8]  # Background, very high confidence
    result = _parse_detections(raw, image_width=640, image_height=480, labels=_MOCK_LABELS)
    assert len(result) == 0, "Background class (class_id=0) must never be in results"


def test_below_threshold_never_in_results():
    """
    TC-ADV-014: Detections with confidence below 0.80 must never appear in results.
    """
    raw = np.zeros((1, 1, 2, 7), dtype=np.float32)
    raw[0, 0, 0] = [0, 15, 0.91, 0.1, 0.1, 0.5, 0.9]  # Keep (above threshold)
    raw[0, 0, 1] = [0, 7,  0.79, 0.2, 0.2, 0.6, 0.8]  # Reject (below threshold)
    result = _parse_detections(raw, image_width=640, image_height=480, labels=_MOCK_LABELS)
    assert len(result) == 1, f"Expected 1 detection, got {len(result)}"
    assert result[0]["confidence"] >= 0.80


def test_iou_computation_is_correct():
    """
    TC-ADV-015: IoU computation must be mathematically correct.
    (ALGORITHM_SPEC §5.1)
    """
    # Two boxes with 50% overlap
    box_a = (0, 0, 100, 100)    # area = 10000
    box_b = (50, 50, 150, 150)  # area = 10000
    # Intersection: (50,50) to (100,100) = 50×50 = 2500
    # Union: 10000 + 10000 - 2500 = 17500
    expected_iou = 2500 / 17500
    computed_iou = compute_iou(box_a, box_b)
    assert abs(computed_iou - expected_iou) < 1e-6, \
        f"IoU mismatch: expected {expected_iou:.6f}, got {computed_iou:.6f}"


def test_non_overlapping_boxes_iou_zero():
    """Two non-overlapping boxes must have IoU = 0."""
    box_a = (0, 0, 50, 50)
    box_b = (100, 100, 200, 200)
    assert compute_iou(box_a, box_b) == 0.0


def test_identical_boxes_iou_one():
    """Identical boxes must have IoU = 1.0."""
    box_a = (10, 10, 100, 100)
    assert compute_iou(box_a, box_a) == 1.0
