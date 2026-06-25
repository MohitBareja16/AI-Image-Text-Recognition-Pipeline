# ─────────────────────────────────────────────────────────────────────────────
# tests/test_confidence_filter.py — Unit Tests: Confidence Filtering (TEST_PLAN §5)
# ─────────────────────────────────────────────────────────────────────────────

import pytest
import pandas as pd
import numpy as np

from src.ocr_pipeline import filter_dataframe
from src.schemas import ObjectDetection, BBox
import src.config as config


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_df(confs, texts=None, widths=None, heights=None):
    """Constructs a minimal DataFrame for filter_dataframe testing."""
    n = len(confs)
    texts = texts or ["word"] * n
    widths = widths or [50] * n
    heights = heights or [20] * n
    return pd.DataFrame({
        "level":    [5] * n,
        "page_num": [1] * n,
        "block_num":[1] * n,
        "par_num":  [1] * n,
        "line_num": [1] * n,
        "word_num": list(range(1, n + 1)),
        "left":     [10] * n,
        "top":      [10] * n,
        "width":    widths,
        "height":   heights,
        "conf":     confs,
        "text":     texts,
    })


# ──────────────────────────────────────────────────────────────────────────────
# TC-CONF tests via Tesseract DataFrame (filter_dataframe)
# ──────────────────────────────────────────────────────────────────────────────

# TC-CONF-001: Score of exactly 80 is ACCEPTED (boundary — inclusive)
def test_boundary_exactly_80_accepted():
    df = _make_df([80.0])
    result = filter_dataframe(df)
    assert len(result) == 1


# TC-CONF-002: Score of 79.99 is REJECTED (boundary — exclusive)
def test_boundary_below_80_rejected():
    df = _make_df([79.99])
    result = filter_dataframe(df)
    assert len(result) == 0


# TC-CONF-003: Score of 100.0 is ACCEPTED (maximum)
def test_perfect_score_accepted():
    df = _make_df([100.0])
    result = filter_dataframe(df)
    assert len(result) == 1


# TC-CONF-004: Score of 0.0 is REJECTED
def test_zero_score_rejected():
    df = _make_df([0.0])
    result = filter_dataframe(df)
    assert len(result) == 0


# TC-CONF-005: Tesseract integer 80 (0–100 scale) is ACCEPTED
def test_tesseract_integer_80_is_accepted():
    """conf=80 in Tesseract's 0-100 scale must be accepted (≥ threshold * 100 = 80)."""
    df = _make_df([80])
    result = filter_dataframe(df)
    assert len(result) == 1, "conf=80 should be accepted"


# TC-CONF-006: Tesseract integer 79 is REJECTED
def test_tesseract_integer_79_is_rejected():
    df = _make_df([79])
    result = filter_dataframe(df)
    assert len(result) == 0


# TC-CONF-007: Tesseract conf=-1 (layout row) is ALWAYS REJECTED
def test_tesseract_neg1_always_rejected():
    df = _make_df([-1.0])
    result = filter_dataframe(df)
    assert len(result) == 0, "conf=-1 layout rows must never appear in output"


# TC-CONF-008: Mixed list — only ≥80 pass through
def test_mixed_list_correct_count():
    df = _make_df([91.0, 79.0, 80.0, 50.0, 100.0])
    result = filter_dataframe(df)
    assert len(result) == 3  # 91, 80, 100 pass
    assert all(v >= 80.0 for v in result["conf"])


# TC-CONF-009: Empty input → empty output (no crash)
def test_empty_input_no_crash():
    empty_df = pd.DataFrame({
        "level": [], "page_num": [], "block_num": [], "par_num": [],
        "line_num": [], "word_num": [], "left": [], "top": [],
        "width": [], "height": [], "conf": [], "text": [],
    })
    result = filter_dataframe(empty_df)
    assert len(result) == 0


# TC-CONF-010: Empty text strings filtered out even if conf is high
def test_empty_text_filtered_out():
    df = _make_df([95.0, 90.0], texts=["", "word"])
    result = filter_dataframe(df)
    assert len(result) == 1
    assert "word" in result["text"].values


# TC-CONF-011: Whitespace-only text filtered out
def test_whitespace_text_filtered_out():
    df = _make_df([95.0], texts=["   "])
    result = filter_dataframe(df)
    assert len(result) == 0


# TC-CONF-012: Zero-width boxes filtered out (degenerate bbox)
def test_zero_width_box_filtered():
    df = _make_df([95.0], widths=[0])
    result = filter_dataframe(df)
    assert len(result) == 0


# TC-CONF-013: Zero-height boxes filtered out
def test_zero_height_box_filtered():
    df = _make_df([95.0], heights=[0])
    result = filter_dataframe(df)
    assert len(result) == 0


# TC-CONF-014: Layout-level rows (conf=-1) are never in output
def test_conf_neg1_rows_not_in_output(sample_tesseract_dataframe):
    """Direct replication of TC-ADV-009."""
    result = filter_dataframe(sample_tesseract_dataframe)
    assert -1 not in result["conf"].values, \
        "conf=-1 layout rows found in filtered DataFrame"


# ──────────────────────────────────────────────────────────────────────────────
# TC-CONF-011: Background class_id=0 REJECTED even at 0.95 (ObjectDetection)
# ──────────────────────────────────────────────────────────────────────────────

def test_background_class_raises_on_construction():
    """
    ObjectDetection with class_id=0 must raise ValueError on construction.
    This enforces the background class filter at the schema level.
    (DATA_SCHEMA.md V-04, TC-CONF-011)
    """
    with pytest.raises(ValueError, match="background"):
        ObjectDetection(
            label="background",
            class_id=0,
            confidence=0.95,
            bbox=BBox(x1=0, y1=0, x2=100, y2=100),
        )


# TC-CONF-015: Confidence 0.7999 is NOT rounded up to 0.80 (TC-ADV-011)
def test_confidence_not_rounded_up():
    """Floating point rounding must not cause 79.99 to be accepted."""
    df = _make_df([79.99])
    result = filter_dataframe(df)
    assert len(result) == 0, "79.99 must be rejected — boundary is exclusive below 80"
