# ─────────────────────────────────────────────────────────────────────────────
# tests/test_bounding_box.py — Unit Tests: BBox Schema (TEST_PLAN §7)
# ─────────────────────────────────────────────────────────────────────────────

import pytest
import numpy as np

from src.schemas import BBox


# TC-BBOX-001: width property computed correctly
def test_bbox_width():
    bbox = BBox(x1=10, y1=20, x2=110, y2=80)
    assert bbox.width == 100


# TC-BBOX-002: height property computed correctly
def test_bbox_height():
    bbox = BBox(x1=10, y1=20, x2=110, y2=80)
    assert bbox.height == 60


# TC-BBOX-003: area computed correctly
def test_bbox_area():
    bbox = BBox(x1=0, y1=0, x2=10, y2=10)
    assert bbox.area == 100


# TC-BBOX-004: to_xywh converts correctly for cv2.NMSBoxes
def test_bbox_to_xywh():
    bbox = BBox(x1=50, y1=30, x2=150, y2=130)
    assert bbox.to_xywh() == (50, 30, 100, 100)


# TC-BBOX-005: Invalid BBox (x1 > x2) raises ValueError on construction
def test_invalid_bbox_x1_gt_x2_raises():
    with pytest.raises(ValueError, match="x1"):
        BBox(x1=200, y1=100, x2=100, y2=200)


# TC-BBOX-006: Invalid BBox (y1 > y2) raises ValueError on construction
def test_invalid_bbox_y1_gt_y2_raises():
    with pytest.raises(ValueError, match="y1"):
        BBox(x1=100, y1=200, x2=200, y2=100)


# TC-BBOX-007: x1 == x2 raises ValueError (zero width)
def test_equal_x_raises():
    with pytest.raises(ValueError):
        BBox(x1=100, y1=10, x2=100, y2=50)


# TC-BBOX-008: y1 == y2 raises ValueError (zero height)
def test_equal_y_raises():
    with pytest.raises(ValueError):
        BBox(x1=10, y1=50, x2=100, y2=50)


# TC-BBOX-009: Float coordinates raise TypeError
def test_bbox_requires_int_like():
    """BBox should reject non-integer inputs."""
    with pytest.raises((TypeError, ValueError)):
        BBox(x1=1.5, y1=2.0, x2=3.5, y2=4.0)


# TC-BBOX-010: numpy integers are accepted and converted to Python int
def test_numpy_ints_accepted():
    """BBox must accept np.int32 / np.int64 and normalize to Python int."""
    bbox = BBox(
        x1=np.int32(10), y1=np.int64(20),
        x2=np.int32(110), y2=np.int64(80),
    )
    assert isinstance(bbox.x1, int)
    assert isinstance(bbox.y1, int)
    assert bbox.width == 100
    assert bbox.height == 60


# TC-BBOX-011: area is always > 0 for valid BBox
def test_bbox_area_always_positive():
    bbox = BBox(x1=0, y1=0, x2=1, y2=1)
    assert bbox.area > 0


# TC-BBOX-012: repr contains all coordinate values
def test_bbox_repr():
    bbox = BBox(x1=1, y1=2, x2=3, y2=4)
    r = repr(bbox)
    assert "1" in r and "2" in r and "3" in r and "4" in r


# TC-BBOX-013: Large coordinate values handled correctly
def test_large_coords():
    bbox = BBox(x1=0, y1=0, x2=4096, y2=4096)
    assert bbox.area == 4096 * 4096
