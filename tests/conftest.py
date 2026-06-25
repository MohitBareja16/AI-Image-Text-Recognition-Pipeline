# ─────────────────────────────────────────────────────────────────────────────
# tests/conftest.py — Shared Fixtures (TEST_PLAN.md §2.3)
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import pytest
import numpy as np
import cv2
import pandas as pd

# Ensure project root is on path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.schemas import BBox, WordDetection, ObjectDetection, PipelineResult

# ── Directory for synthetic test images ──────────────────────────────────────
TEST_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "test_images")
os.makedirs(TEST_IMAGES_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Basic image fixtures (TEST_PLAN §2.3)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def black_image():
    """Pure black 100×100 BGR image."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def white_image():
    """Pure white 100×100 BGR image."""
    return np.full((100, 100, 3), 255, dtype=np.uint8)


@pytest.fixture
def synthetic_text_image():
    """
    White image with 'HELLO' printed in black — guaranteed OCR target.
    (TEST_PLAN §2.3)
    """
    img = np.full((100, 400, 3), 255, dtype=np.uint8)
    cv2.putText(
        img, "HELLO", (50, 70),
        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA,
    )
    return img


@pytest.fixture
def synthetic_multiword_image():
    """White image with 'HELLO WORLD TEST' for multi-word OCR testing."""
    img = np.full((100, 700, 3), 255, dtype=np.uint8)
    cv2.putText(img, "HELLO", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, "WORLD", (240, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, "TEST", (470, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3, cv2.LINE_AA)
    return img


# ══════════════════════════════════════════════════════════════════════════════
# On-disk test image fixtures (generated programmatically)
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_test_image(name: str, content_fn) -> str:
    """Creates a test image on disk if it doesn't already exist."""
    path = os.path.join(TEST_IMAGES_DIR, name)
    if not os.path.isfile(path):
        img = content_fn()
        cv2.imwrite(path, img)
    return path


@pytest.fixture(scope="session")
def ocr_clean_path():
    """Path to a clean OCR test image (high-quality text)."""
    def make():
        img = np.full((200, 600, 3), 255, dtype=np.uint8)
        cv2.putText(img, "HELLO WORLD TEST", (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3, cv2.LINE_AA)
        return img
    return _ensure_test_image("ocr_clean.png", make)


@pytest.fixture(scope="session")
def ocr_blank_path():
    """Path to a blank white image — expected: 0 OCR detections."""
    def make():
        return np.full((200, 600, 3), 255, dtype=np.uint8)
    return _ensure_test_image("ocr_blank.png", make)


@pytest.fixture(scope="session")
def tiny_image_path():
    """Path to a 10×10 image — expected: rejected by size validation."""
    def make():
        return np.zeros((10, 10, 3), dtype=np.uint8)
    return _ensure_test_image("tiny.png", make)


@pytest.fixture(scope="session")
def corrupt_image_path():
    """Path to a file with corrupted JPEG content."""
    path = os.path.join(TEST_IMAGES_DIR, "corrupt.jpg")
    if not os.path.isfile(path):
        with open(path, "wb") as f:
            f.write(b"this is not a real jpeg header at all")
    return path


@pytest.fixture(scope="session")
def zero_bytes_path():
    """Path to a zero-byte file — expected: rejected by size validation."""
    path = os.path.join(TEST_IMAGES_DIR, "zero_bytes.jpg")
    if not os.path.isfile(path):
        open(path, "w").close()
    return path


# ══════════════════════════════════════════════════════════════════════════════
# DataFrame fixtures (TEST_PLAN §2.3)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_tesseract_dataframe():
    """
    Realistic mock DataFrame as returned by pytesseract.image_to_data.
    Row conf values: -1 (layout), -1, -1, -1, 95.0 (pass), 45.0 (fail), 82.0 (pass)
    """
    return pd.DataFrame({
        "level":    [1,   2,   3,   4,   5,      5,      5   ],
        "page_num": [1,   1,   1,   1,   1,      1,      1   ],
        "block_num":[0,   1,   1,   1,   1,      1,      1   ],
        "par_num":  [0,   0,   1,   1,   1,      1,      1   ],
        "line_num": [0,   0,   0,   1,   1,      1,      1   ],
        "word_num": [0,   0,   0,   0,   1,      2,      3   ],
        "left":     [0,   10,  10,  10,  10,     100,    200 ],
        "top":      [0,   10,  10,  10,  10,     10,     10  ],
        "width":    [400, 380, 380, 380, 80,     90,     70  ],
        "height":   [100, 80,  80,  80,  30,     30,     30  ],
        "conf":     [-1,  -1,  -1,  -1,  95.0,   45.0,   82.0],
        "text":     ["",  "",  "",  "",  "Hello", "world", "test"],
    })


# ══════════════════════════════════════════════════════════════════════════════
# Mock MobileNet-SSD raw output (TEST_PLAN §2.3)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def mock_detections_raw():
    """
    Simulated raw MobileNet-SSD output array: shape (1, 1, 3, 7).
    Detection 0: Person (class 15), conf=0.91 → KEPT
    Detection 1: Car (class 7),     conf=0.67 → REJECTED (below threshold)
    Detection 2: Background (class 0), conf=0.95 → REJECTED (background)
    """
    detections = np.zeros((1, 1, 3, 7), dtype=np.float32)
    detections[0, 0, 0] = [0, 15, 0.91, 0.1, 0.2, 0.4, 0.8]   # Person — keep
    detections[0, 0, 1] = [0, 7,  0.67, 0.5, 0.1, 0.9, 0.6]   # Car — reject
    detections[0, 0, 2] = [0, 0,  0.95, 0.0, 0.0, 1.0, 1.0]   # Background — reject
    return detections


# ══════════════════════════════════════════════════════════════════════════════
# Schema object fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def valid_bbox():
    return BBox(x1=10, y1=20, x2=110, y2=80)


@pytest.fixture
def word_detection_fixture():
    return WordDetection(
        text="Hello",
        confidence=0.95,
        bbox=BBox(x1=10, y1=10, x2=90, y2=40),
        line_num=1,
        block_num=1,
        word_num=1,
    )


@pytest.fixture
def object_detection_fixture():
    return ObjectDetection(
        label="person",
        class_id=15,
        confidence=0.91,
        bbox=BBox(x1=64, y1=96, x2=256, y2=384),
    )
