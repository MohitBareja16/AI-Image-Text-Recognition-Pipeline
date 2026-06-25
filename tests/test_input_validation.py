# ─────────────────────────────────────────────────────────────────────────────
# tests/test_input_validation.py — Unit Tests: Input Validation (TEST_PLAN §3)
# ─────────────────────────────────────────────────────────────────────────────

import os
import pytest
import numpy as np
import cv2

from src.utils import validate_image_path, load_image


# TC-INV-001: Valid .jpg path accepted
def test_valid_jpg_path(tmp_path):
    """A valid .jpg file path returns True."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    p = tmp_path / "test.jpg"
    cv2.imwrite(str(p), img)
    assert validate_image_path(str(p)) is True


# TC-INV-002: Valid .png path accepted
def test_valid_png_path(tmp_path):
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    p = tmp_path / "test.png"
    cv2.imwrite(str(p), img)
    assert validate_image_path(str(p)) is True


# TC-INV-003: Empty string rejected
def test_empty_path_rejected():
    assert validate_image_path("") is False


# TC-INV-004: PDF extension rejected
def test_pdf_extension_rejected(tmp_path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"fake pdf content")
    assert validate_image_path(str(p)) is False


# TC-INV-005: Non-existent file rejected
def test_nonexistent_file_rejected():
    assert validate_image_path("/definitely/does/not/exist.jpg") is False


# TC-INV-006: Zero-byte file rejected
def test_zero_byte_file_rejected(tmp_path):
    p = tmp_path / "empty.jpg"
    p.write_bytes(b"")
    assert validate_image_path(str(p)) is False


# TC-INV-007: File over 50MB rejected
def test_oversized_file_rejected(tmp_path):
    p = tmp_path / "huge.jpg"
    p.write_bytes(b"x" * (51 * 1024 * 1024))  # 51 MB
    assert validate_image_path(str(p)) is False


# TC-INV-008: .txt extension rejected
def test_txt_extension_rejected(tmp_path):
    p = tmp_path / "data.txt"
    p.write_text("hello")
    assert validate_image_path(str(p)) is False


# TC-INV-009: Case-insensitive extension accepted (.JPG)
def test_uppercase_extension_accepted(tmp_path):
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    p = tmp_path / "TEST.JPG"
    cv2.imwrite(str(p), img)
    assert validate_image_path(str(p)) is True


# TC-INV-010: .bmp extension accepted
def test_bmp_extension_accepted(tmp_path):
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    p = tmp_path / "test.bmp"
    cv2.imwrite(str(p), img)
    assert validate_image_path(str(p)) is True


# TC-INV-011: .tiff extension accepted
def test_tiff_extension_accepted(tmp_path):
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    p = tmp_path / "test.tiff"
    cv2.imwrite(str(p), img)
    assert validate_image_path(str(p)) is True


# TC-INV-012: load_image raises FileNotFoundError on corrupt file
def test_load_image_raises_on_corrupt_file(tmp_path):
    """A file that exists but can't be decoded must raise FileNotFoundError."""
    p = tmp_path / "corrupt.jpg"
    p.write_bytes(b"this is definitely not a real jpeg file at all !!!!")
    with pytest.raises(FileNotFoundError):
        load_image(str(p))


# TC-INV-013: load_image returns ndarray with correct shape
def test_load_image_returns_correct_shape(tmp_path):
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    p = tmp_path / "valid.jpg"
    cv2.imwrite(str(p), img)
    loaded = load_image(str(p))
    assert isinstance(loaded, np.ndarray)
    assert loaded.ndim == 3
    assert loaded.shape[2] == 3
    assert loaded.dtype == np.uint8


# TC-INV-014: Whitespace-only path rejected
def test_whitespace_path_rejected():
    assert validate_image_path("   ") is False
