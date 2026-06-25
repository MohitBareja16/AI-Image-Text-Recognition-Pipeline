# ─────────────────────────────────────────────────────────────────────────────
# tests/test_preprocessing.py — Unit Tests: Pre-Processing (TEST_PLAN §4)
# ─────────────────────────────────────────────────────────────────────────────

import pytest
import numpy as np
import cv2

from src.preprocessor import preprocess
from src.schemas import PreprocessedBundle


# TC-PRE-001: Output grayscale shape is (H, W) not (H, W, 1)
def test_grayscale_shape(black_image):
    bundle = preprocess(black_image)
    assert bundle.gray.ndim == 2
    assert bundle.gray.shape == black_image.shape[:2]


# TC-PRE-002: Grayscale dtype is uint8
def test_grayscale_dtype(white_image):
    bundle = preprocess(white_image)
    assert bundle.gray.dtype == np.uint8


# TC-PRE-003: Binary image contains only 0 and 255
def test_binary_only_zero_and_255(synthetic_text_image):
    bundle = preprocess(synthetic_text_image)
    unique_vals = set(np.unique(bundle.binary))
    extra = unique_vals - {0, 255}
    assert not extra, f"Binary image contains unexpected values: {extra}"


# TC-PRE-004: Original image is not modified
def test_original_image_unmodified(synthetic_text_image):
    original_copy = synthetic_text_image.copy()
    preprocess(synthetic_text_image)
    np.testing.assert_array_equal(synthetic_text_image, original_copy)


# TC-PRE-005: Output shapes match input shape
def test_all_shapes_match(synthetic_text_image):
    H, W = synthetic_text_image.shape[:2]
    bundle = preprocess(synthetic_text_image)
    assert bundle.gray.shape == (H, W)
    assert bundle.blurred.shape == (H, W)
    assert bundle.binary.shape == (H, W)


# TC-PRE-006: None input raises ValueError
def test_none_input_raises():
    with pytest.raises(ValueError, match="None"):
        preprocess(None)


# TC-PRE-007: Wrong dtype (float32) raises ValueError
def test_float_image_raises():
    float_image = np.zeros((100, 100, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="dtype"):
        preprocess(float_image)


# TC-PRE-008: Too-small image raises ValueError
def test_tiny_image_raises():
    tiny = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="too small"):
        preprocess(tiny)


# TC-PRE-009: Gaussian blur produces different values than grayscale input
def test_blur_is_not_noop(synthetic_text_image):
    """Blurred image must differ from grayscale (except all-uniform images).
    synthetic_text_image has edges — blur will change them."""
    bundle = preprocess(synthetic_text_image)
    # Blur should differ from gray near edges
    assert not np.array_equal(bundle.gray, bundle.blurred), \
        "Gaussian blur appears to be a no-op — check kernel size"


# TC-PRE-010: Non-3-channel image raises ValueError
def test_non_3channel_raises():
    gray_2d = np.zeros((100, 100), dtype=np.uint8)
    with pytest.raises(ValueError):
        preprocess(gray_2d)


# TC-PRE-011: PreprocessedBundle invariants hold for normal image
def test_bundle_invariants(synthetic_text_image):
    bundle = preprocess(synthetic_text_image)
    H, W = synthetic_text_image.shape[:2]
    assert bundle.gray.ndim == 2
    assert bundle.blurred.ndim == 2
    assert bundle.binary.ndim == 2
    assert bundle.gray.shape == (H, W)
    assert bundle.original.shape == (H, W, 3)
    assert bundle.original.dtype == np.uint8


# TC-PRE-012: Deskew disabled → angle is 0.0
def test_deskew_disabled_angle_zero(synthetic_text_image, monkeypatch):
    monkeypatch.setattr("src.config.DESKEW_ENABLED", False)
    bundle = preprocess(synthetic_text_image)
    assert bundle.deskewed is False
    assert bundle.angle == 0.0


# TC-PRE-013: Large image processes without crash
def test_large_image_processes():
    large = np.full((1024, 1024, 3), 128, dtype=np.uint8)
    bundle = preprocess(large)
    assert bundle.binary.shape == (1024, 1024)


# TC-PRE-014: White-only image produces valid binary output
def test_white_image_binary(white_image):
    """White image should produce valid binary without crashing."""
    bundle = preprocess(white_image)
    assert bundle.binary.dtype == np.uint8
    unique = set(np.unique(bundle.binary))
    assert unique.issubset({0, 255})
