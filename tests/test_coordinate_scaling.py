# ─────────────────────────────────────────────────────────────────────────────
# tests/test_coordinate_scaling.py — Unit Tests: Coordinate Scaling (TEST_PLAN §6)
# ─────────────────────────────────────────────────────────────────────────────

import pytest

from src.utils import scale_coordinates, validate_bbox_coords


# TC-COORD-001: Normalized (0.0, 0.0, 1.0, 1.0) → full image bounds
def test_full_frame_detection():
    bbox = scale_coordinates((0.0, 0.0, 1.0, 1.0), image_width=640, image_height=480)
    assert bbox == (0, 0, 640, 480)


# TC-COORD-002: Normalized (0.5, 0.5, 0.75, 0.75) → correct pixel coords
def test_center_detection():
    bbox = scale_coordinates((0.5, 0.5, 0.75, 0.75), image_width=400, image_height=400)
    assert bbox == (200, 200, 300, 300)


# TC-COORD-003: Negative normalized coord clamped to 0
def test_negative_coord_clamped():
    bbox = scale_coordinates((-0.1, 0.0, 0.5, 0.5), image_width=100, image_height=100)
    assert bbox[0] == 0, f"x1={bbox[0]} should be clamped to 0"


# TC-COORD-004: Normalized coord > 1.0 clamped to image boundary
def test_overflow_coord_clamped():
    bbox = scale_coordinates((0.0, 0.0, 1.5, 1.2), image_width=100, image_height=100)
    assert bbox[2] == 100, f"x2={bbox[2]} should be clamped to width=100"
    assert bbox[3] == 100, f"y2={bbox[3]} should be clamped to height=100"


# TC-COORD-005: All output values are int (not float)
def test_output_values_are_int():
    bbox = scale_coordinates((0.1, 0.2, 0.8, 0.9), image_width=300, image_height=200)
    assert all(isinstance(v, int) for v in bbox), \
        f"All coords must be int, got types: {[type(v).__name__ for v in bbox]}"


# TC-COORD-006: Inverted box (x_start > x_end) → flagged as invalid
def test_inverted_box_detected():
    assert validate_bbox_coords(x1=200, y1=50, x2=100, y2=150) is False


# TC-COORD-007: Zero-area box (x1==x2) → flagged as invalid
def test_zero_width_box_invalid():
    assert validate_bbox_coords(x1=100, y1=100, x2=100, y2=200) is False


# TC-COORD-007b: Zero-area box (y1==y2) → flagged as invalid
def test_zero_height_box_invalid():
    assert validate_bbox_coords(x1=100, y1=100, x2=200, y2=100) is False


# TC-COORD-008: Scaling uses ORIGINAL image dimensions, NOT blob (300×300)
def test_uses_original_not_blob_dimensions():
    """
    Critical anti-loophole test: common bug is using 300 instead of actual W/H.
    (ALGORITHM_SPEC §6.1, TC-COORD-008)
    """
    original_W, original_H = 1920, 1080
    bbox = scale_coordinates((0.5, 0.5, 0.75, 0.75), original_W, original_H)
    # Correct: multiply by original dimensions
    assert bbox[2] == int(0.75 * 1920), f"x2={bbox[2]}, expected {int(0.75*1920)} (not {int(0.75*300)})"
    assert bbox[3] == int(0.75 * 1080), f"y2={bbox[3]}, expected {int(0.75*1080)} (not {int(0.75*300)})"


# TC-COORD-009: (0.0, 0.0, 0.0, 0.0) → invalid (zero area)
def test_all_zero_normalized_invalid():
    bbox = scale_coordinates((0.0, 0.0, 0.0, 0.0), image_width=640, image_height=480)
    assert not validate_bbox_coords(*bbox)


# TC-COORD-010: Valid coords pass validate_bbox_coords
def test_valid_coords_pass():
    assert validate_bbox_coords(x1=0, y1=0, x2=100, y2=100) is True
    assert validate_bbox_coords(x1=50, y1=50, x2=200, y2=200) is True


# TC-COORD-011: Scaled values are actually multiplied (anti-loophole TC-ADV-003)
def test_coordinate_scaling_actually_multiplies():
    """
    If coords are not scaled, they remain near 0.0–1.0.
    Scaled pixel coordinates for a 640×480 image should be 100–600 range.
    """
    raw = (0.2, 0.3, 0.7, 0.8)
    W, H = 640, 480
    bbox = scale_coordinates(raw, W, H)
    assert bbox[0] > 1, f"x1={bbox[0]} looks unscaled (still near 0–1 float range)"
    assert bbox[2] > 1, f"x2={bbox[2]} looks unscaled"
