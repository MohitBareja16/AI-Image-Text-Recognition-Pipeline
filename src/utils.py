# ─────────────────────────────────────────────────────────────────────────────
# utils.py — File Validation, Image I/O, Model Loading, Logging Helpers
# Implements: ARCHITECTURE.md §4 utils.py contract, DATA_SCHEMA.md §1 & §9
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import sys
import logging
import cv2
import numpy as np
from typing import List

from src import config
from src.schemas import (
    InputValidationError,
    ModelLoadError,
)

# ── Logger setup ──────────────────────────────────────────────────────────────
# Single named logger for the entire pipeline; propagates to root by default.
logger = logging.getLogger("decodelabs.project4")

# Valid input image extensions (DATA_SCHEMA.md §1.1)
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# Maximum allowed file size: 50 MB
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024

# Minimum image dimension (prevents tiny images that crash preprocessing)
MIN_DIMENSION = 32


def validate_image_path(path: str) -> bool:
    """
    Validates that the path exists, is readable, and has a valid image extension.
    Prints a descriptive error message and returns False if any check fails.
    Returns True only when all checks pass.

    Checks (in order):
        1. Path is a non-empty string
        2. File exists on disk
        3. Extension is in VALID_EXTENSIONS (case-insensitive)
        4. File size > 0 bytes
        5. File size ≤ 50 MB
        6. File is readable (no permission error)
    """
    # Check 1: Non-empty path
    if not path or not path.strip():
        _print_error("Image path cannot be empty.")
        return False

    # Check 2: File exists
    if not os.path.isfile(path):
        _print_error(f"File not found: {path}")
        return False

    # Check 3: Valid extension (case-insensitive per TC-INV-009)
    ext = os.path.splitext(path)[1].lower()
    if ext not in VALID_EXTENSIONS:
        _print_error(
            f"Unsupported format: {ext or '(no extension)'}. "
            f"Use: {', '.join(sorted(VALID_EXTENSIONS))}"
        )
        return False

    # Check 4 & 5: File size
    try:
        file_size = os.path.getsize(path)
    except OSError as e:
        _print_error(f"Cannot read file metadata: {e}")
        return False

    if file_size == 0:
        _print_error(f"File is empty (0 bytes): {path}")
        return False

    if file_size > MAX_FILE_SIZE_BYTES:
        mb = file_size / (1024 * 1024)
        _print_error(f"File too large ({mb:.1f} MB). Maximum allowed: 50 MB.")
        return False

    # Check 6: Read permission
    if not os.access(path, os.R_OK):
        _print_error(f"Permission denied: {path}")
        return False

    return True


def load_image(path: str) -> np.ndarray:
    """
    Loads an image from disk using OpenCV.
    Raises FileNotFoundError (not AttributeError) when cv2.imread returns None.

    Args:
        path: A validated, readable file path.

    Returns:
        np.ndarray: BGR image, shape=(H, W, 3), dtype=uint8.

    Raises:
        FileNotFoundError: If cv2.imread cannot decode the file.
    """
    image = cv2.imread(path)

    # cv2.imread returns None for unreadable/corrupt files — must be caught explicitly
    # (DATA_SCHEMA.md §1.2 — CORRECT check pattern)
    if image is None:
        raise FileNotFoundError(
            f"OpenCV could not decode image: '{path}'\n"
            "The file may be corrupt, truncated, or not a valid image format."
        )

    return image


def load_labels(labels_path: str) -> List[str]:
    """
    Loads class labels from a plain-text file. One label per line.
    Strips whitespace; skips blank lines and comment lines (starting with #).

    Args:
        labels_path: Absolute or relative path to the labels .txt file.

    Returns:
        List[str]: Ordered list of class label strings.

    Raises:
        ModelLoadError: If the file does not exist or is empty.
    """
    if not os.path.isfile(labels_path):
        raise ModelLoadError(
            stage="model_load",
            message=(
                f"Labels file not found: {labels_path}\n"
                "Run: python setup_models.py"
            ),
            recoverable=True,
        )

    labels = []
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                labels.append(stripped)

    if not labels:
        raise ModelLoadError(
            stage="model_load",
            message=f"Labels file is empty: {labels_path}",
            recoverable=True,
        )

    logger.debug("Loaded %d class labels from %s", len(labels), labels_path)
    return labels


def load_model(prototxt: str, caffemodel: str) -> "cv2.dnn_Net":
    """
    Loads a MobileNet-SSD model from .prototxt + .caffemodel files.
    Raises ModelLoadError with a helpful message if any file is missing.

    Args:
        prototxt:   Path to MobileNetSSD_deploy.prototxt
        caffemodel: Path to MobileNetSSD_deploy.caffemodel

    Returns:
        cv2.dnn_Net: Loaded Caffe network ready for inference.

    Raises:
        ModelLoadError: If either file is missing or OpenCV fails to load.
    """
    # Validate both files exist before attempting load
    for fpath, fname in [(prototxt, "prototxt"), (caffemodel, "caffemodel")]:
        if not os.path.isfile(fpath):
            raise ModelLoadError(
                stage="model_load",
                message=(
                    f"Model file not found ({fname}): {fpath}\n"
                    "Run: python setup_models.py"
                ),
                recoverable=True,
            )

    try:
        net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
    except cv2.error as e:
        raise ModelLoadError(
            stage="model_load",
            message=f"OpenCV failed to load model: {e}",
            recoverable=False,
        )

    logger.debug("MobileNet-SSD model loaded successfully.")
    return net


def ensure_output_dir() -> str:
    """
    Creates the output directory if it does not already exist.
    Returns the absolute path to the output directory.
    """
    output_dir = os.path.abspath(config.OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def compute_iou(box_a: tuple, box_b: tuple) -> float:
    """
    Computes Intersection over Union (IoU) for two bounding boxes.
    Used in NMS validation tests (ALGORITHM_SPEC.md §5.1).

    Args:
        box_a: (x1, y1, x2, y2) pixel coordinates
        box_b: (x1, y1, x2, y2) pixel coordinates

    Returns:
        float: IoU value in [0.0, 1.0]
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # Intersection rectangle
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    intersection = inter_w * inter_h

    # Union
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection

    if union == 0:
        return 0.0

    return intersection / union


def scale_coordinates(
    normalized: tuple,
    image_width: int,
    image_height: int,
) -> tuple:
    """
    Scales normalized [0, 1] MobileNet-SSD coordinates to pixel coordinates
    using the ORIGINAL image dimensions (not the 300×300 blob dimensions).

    Applies boundary clamping to prevent coordinates from going outside the image.
    (ALGORITHM_SPEC.md §6.1, DATA_SCHEMA.md V-03)

    Args:
        normalized: (x_n_start, y_n_start, x_n_end, y_n_end) floats in [0, 1]
        image_width:  Original image width in pixels (NOT 300)
        image_height: Original image height in pixels (NOT 300)

    Returns:
        tuple: (x1, y1, x2, y2) as Python ints in pixel coordinates.
    """
    x_n_start, y_n_start, x_n_end, y_n_end = normalized

    # Scale to pixel coordinates
    x1 = int(x_n_start * image_width)
    y1 = int(y_n_start * image_height)
    x2 = int(x_n_end * image_width)
    y2 = int(y_n_end * image_height)

    # Clamp to image boundaries (ALGORITHM_SPEC.md §6.1 — mandatory clamping)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image_width, x2)
    y2 = min(image_height, y2)

    return (int(x1), int(y1), int(x2), int(y2))


def validate_bbox_coords(x1: int, y1: int, x2: int, y2: int) -> bool:
    """
    Returns True only if the bounding box has positive area and valid ordering.
    Used to skip inverted or zero-area boxes from MobileNet output.
    (DATA_SCHEMA.md §3.1 — malformed output handling)
    """
    return x1 < x2 and y1 < y2


def configure_logging(level: int = logging.INFO) -> None:
    """
    Configures the pipeline logger with a simple console handler.
    Call once from main.py before running the pipeline.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s.%(msecs)03d] %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)


def _print_error(message: str) -> None:
    """Helper: prints a standardized ERROR-prefixed message to stderr."""
    print(f"ERROR: {message}", file=sys.stderr)
