# ─────────────────────────────────────────────────────────────────────────────
# preprocessor.py — Image Pre-Processing Pipeline
# Implements ARCHITECTURE.md §4 preprocessor contract + ALGORITHM_SPEC.md §1 + §8
# Steps (in mandatory order): Grayscale → GaussianBlur → [Deskew] → Threshold
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import cv2
import numpy as np

from src import config
from src.schemas import PreprocessedBundle
from src.utils import MIN_DIMENSION

logger = logging.getLogger("decodelabs.project4.preprocessor")


def preprocess(image: np.ndarray) -> PreprocessedBundle:
    """
    Applies the full pre-processing pipeline to a raw BGR image.

    Pipeline steps (ALGORITHM_SPEC.md §1, in mandatory order):
        1. Grayscale conversion  (ARCH §3.2, ALGO §1.1)
        2. Gaussian Blur 5×5    (ALGO §1.2)
        3. Deskew               (ALGO §8) — if DESKEW_ENABLED and angle > threshold
        4. Adaptive Threshold   (ALGO §1.3)

    Args:
        image: Raw BGR image loaded via cv2.imread(), shape=(H, W, 3), dtype=uint8.

    Returns:
        PreprocessedBundle: Contains gray, blurred, binary, deskewed flag, angle, original.

    Raises:
        ValueError: If image is None, has invalid shape, wrong dtype, or is too small.
    """
    # ── Validate input ────────────────────────────────────────────────────────
    _validate_input_image(image)

    # Preserve reference to the original so we can confirm it is never modified.
    original = image  # No copy needed here — we never write to `image`

    # ── Step 1: Grayscale Conversion (ALGO §1.1) ─────────────────────────────
    # Formula (ITU-R BT.601 luma): Y = 0.114×B + 0.587×G + 0.299×R
    # Removes color noise; collapses 3-channel BGR to 1-channel luminance.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    logger.debug("Step 1 complete: grayscale shape=%s dtype=%s", gray.shape, gray.dtype)

    # ── Step 2: Gaussian Blur 5×5 (ALGO §1.2) ────────────────────────────────
    # Kernel (5×5, σ=0 → auto σ≈1.1). Smooths high-frequency noise that creates
    # false character edges or false positives in subsequent thresholding.
    blurred = cv2.GaussianBlur(
        gray,
        ksize=config.GAUSSIAN_BLUR_KERNEL,
        sigmaX=0,  # σ=0: OpenCV auto-computes from kernel size
    )
    logger.debug("Step 2 complete: gaussian blur kernel=%s", config.GAUSSIAN_BLUR_KERNEL)

    # ── Step 3: Deskew (Optional, ALGO §8) ───────────────────────────────────
    # Corrects tilt that degrades OCR accuracy by 15–40% even at 2° rotation.
    # Applied AFTER blur, BEFORE threshold (grayscale edge detection).
    deskewed = False
    angle = 0.0

    if config.DESKEW_ENABLED:
        blurred, angle = _deskew(blurred)
        if abs(angle) > config.DESKEW_ANGLE_THRESHOLD:
            deskewed = True
            logger.info("Deskew applied: corrected %.2f°", angle)
        else:
            # Angle below threshold — _deskew returns original if angle is small
            angle = 0.0
            logger.debug("Deskew skipped: angle %.2f° ≤ threshold %.2f°",
                         angle, config.DESKEW_ANGLE_THRESHOLD)
    else:
        logger.debug("Deskew disabled via config.")

    # ── Step 4: Adaptive Thresholding (ALGO §1.3) ────────────────────────────
    # Converts gradient grayscale to sharp black/white (binary) image.
    # Uses ADAPTIVE_THRESH_GAUSSIAN_C — robust under uneven/shadowed lighting.
    # Each pixel threshold = local Gaussian-weighted mean − C.
    binary = cv2.adaptiveThreshold(
        blurred,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=11,   # Neighborhood size (must be odd; 11 is robust default)
        C=2,            # Constant subtracted from local mean; tunes sensitivity
    )
    logger.debug("Step 4 complete: adaptive threshold binary shape=%s", binary.shape)

    # ── Reconstruct gray to match blurred (deskew may have altered blurred) ──
    # After deskew, blurred is the corrected image; gray must stay in sync.
    # We reuse the pre-deskew gray for reference and expose blurred as canonical.
    # But gray must match the spatial shape of binary for the invariant to hold.
    if deskewed:
        # Recompute gray from the deskewed blurred image (reverse is impossible;
        # we re-derive a compatible grayscale for bundle shape consistency)
        gray = blurred  # Both are (H,W) grayscale — shape invariant satisfied

    # ── Construct and return validated bundle ──────────────────────────────────
    return PreprocessedBundle(
        gray=gray,
        blurred=blurred,
        binary=binary,
        original=original,
        deskewed=deskewed,
        angle=angle,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────────────

def _validate_input_image(image) -> None:
    """
    Validates that the input image is a proper BGR ndarray.
    Raises ValueError with descriptive messages for all failure cases.
    (DATA_SCHEMA.md §1.2, TEST_PLAN §4 TC-PRE-006/007/008)
    """
    if image is None:
        raise ValueError(
            "image cannot be None. Load the image first with utils.load_image()."
        )

    if not isinstance(image, np.ndarray):
        raise ValueError(
            f"image must be a numpy ndarray, got {type(image).__name__}"
        )

    if image.dtype != np.uint8:
        raise ValueError(
            f"image must have dtype uint8. Got: {image.dtype}. "
            "Convert with image.astype(np.uint8) if needed."
        )

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"image must be a 3-channel BGR array (H, W, 3). Got shape: {image.shape}"
        )

    h, w = image.shape[:2]
    if h < MIN_DIMENSION or w < MIN_DIMENSION:
        raise ValueError(
            f"image too small: {w}×{h} pixels. Minimum: {MIN_DIMENSION}×{MIN_DIMENSION}."
        )


def _deskew(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Detects and corrects image skew using Canny edges + minAreaRect.
    (ALGORITHM_SPEC.md §8)

    Returns the (possibly rotated) image and the detected angle in degrees.
    If the detected angle is below DESKEW_ANGLE_THRESHOLD, returns the
    original image unchanged and the raw detected angle.

    Args:
        gray: 2D grayscale image (after GaussianBlur).

    Returns:
        (corrected_image, angle_degrees):
            corrected_image — rotated image (or original if angle < threshold)
            angle_degrees   — raw detected angle (before threshold check)
    """
    # Step 1: Canny edge detection to find structure in the image
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    # Step 2: Find all contours and collect their pixel coordinates
    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        # No edges found — cannot determine skew angle; return unchanged
        return gray, 0.0

    all_points = np.concatenate([c.reshape(-1, 2) for c in contours])

    # Step 3: Fit minimum-area bounding rectangle to all edge points
    rect = cv2.minAreaRect(all_points)
    angle = rect[2]  # Rotation angle in degrees, range (-90, 0]

    # Step 4: Normalize angle to (-45°, 45°)
    # minAreaRect returns angles in (-90, 0]; values below -45° need adjustment.
    if angle < -45:
        angle = 90 + angle  # Maps (-90,-45) → (0,45)

    # Step 5: Check if correction is worth the resampling cost
    if abs(angle) <= config.DESKEW_ANGLE_THRESHOLD:
        # Tilt too small to warrant rotation; return original
        return gray, angle

    # Step 6: Compute rotation matrix about image center
    h, w = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale=1.0)

    # Step 7: Apply rotation with BORDER_REPLICATE to prevent black corner artifacts
    # (ALGO §8 — black corners from BORDER_CONSTANT cause false threshold detections)
    corrected = cv2.warpAffine(
        gray, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return corrected, angle
