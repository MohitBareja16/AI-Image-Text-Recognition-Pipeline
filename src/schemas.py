# ─────────────────────────────────────────────────────────────────────────────
# schemas.py — Data Schemas and Exceptions for DecodeLabs Project 4
# All dataclasses and exception hierarchy defined here per DATA_SCHEMA.md §2 & §8.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# § Exception Hierarchy  (DATA_SCHEMA.md §8)
# ══════════════════════════════════════════════════════════════════════════════

class PipelineError(Exception):
    """Base exception for all Project 4 pipeline errors."""

    def __init__(self, stage: str, message: str, recoverable: bool = False):
        self.stage = stage              # Which pipeline stage failed
        self.message = message          # Human-readable description
        self.recoverable = recoverable  # Can the user fix without code changes?
        super().__init__(f"[{stage}] {message}")


class InputValidationError(PipelineError):
    """Raised when input image fails validation checks."""
    pass


class ModelLoadError(PipelineError):
    """Raised when model files cannot be loaded."""
    pass


class InferenceError(PipelineError):
    """Raised when the model forward pass fails."""
    pass


class OutputError(PipelineError):
    """Raised when the annotated output image cannot be saved."""
    pass


# ══════════════════════════════════════════════════════════════════════════════
# § BBox — Bounding Box in Pixel Space  (DATA_SCHEMA.md §2.4)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BBox:
    """
    Bounding box always stored in pixel-integer coordinates.
    No normalized floats are ever stored in a BBox object — they are converted
    before construction (see validation rule V-03).

    Coordinate convention:
        (x1, y1) = top-left corner
        (x2, y2) = bottom-right corner
    """
    x1: int  # Left edge  — 0 ≤ x1 < image_width
    y1: int  # Top edge   — 0 ≤ y1 < image_height
    x2: int  # Right edge — x1 < x2 ≤ image_width
    y2: int  # Bottom edge — y1 < y2 ≤ image_height

    def __post_init__(self):
        # Enforce integer types (DATA_SCHEMA.md V-03)
        for attr in ("x1", "y1", "x2", "y2"):
            val = getattr(self, attr)
            if not isinstance(val, (int, np.integer)):
                raise TypeError(
                    f"BBox coordinates must be integers. "
                    f"Got {attr}={val!r} (type: {type(val).__name__})"
                )
            object.__setattr__(self, attr, int(val))  # Normalize numpy ints

        # Enforce validity: x1 < x2 and y1 < y2  (DATA_SCHEMA.md V-03)
        if self.x1 >= self.x2:
            raise ValueError(
                f"BBox requires x1 < x2, got x1={self.x1}, x2={self.x2}"
            )
        if self.y1 >= self.y2:
            raise ValueError(
                f"BBox requires y1 < y2, got y1={self.y1}, y2={self.y2}"
            )

    @property
    def width(self) -> int:
        """Horizontal span in pixels. Always > 0."""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """Vertical span in pixels. Always > 0."""
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        """Area in square pixels. Always > 0."""
        return self.width * self.height

    def to_xywh(self) -> tuple:
        """Convert to (x, y, w, h) format required by cv2.dnn.NMSBoxes."""
        return (self.x1, self.y1, self.width, self.height)

    def __repr__(self) -> str:
        return f"BBox(x1={self.x1}, y1={self.y1}, x2={self.x2}, y2={self.y2})"


# ══════════════════════════════════════════════════════════════════════════════
# § PreprocessedBundle  (DATA_SCHEMA.md §2.1)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PreprocessedBundle:
    """
    Output of preprocessor.preprocess(). Contains every intermediate image
    stage needed by the OCR and detection pipelines.

    Invariants (asserted on construction):
        gray.ndim == blurred.ndim == binary.ndim == 2
        gray.shape == blurred.shape == binary.shape == original.shape[:2]
        binary dtype == uint8 and values ∈ {0, 255}
        original dtype == uint8 and original.ndim == 3
    """
    gray:     np.ndarray   # Grayscale image.     shape=(H,W)   dtype=uint8
    blurred:  np.ndarray   # After GaussianBlur.  shape=(H,W)   dtype=uint8
    binary:   np.ndarray   # After thresholding.  shape=(H,W)   dtype=uint8  values∈{0,255}
    original: np.ndarray   # Original BGR image.  shape=(H,W,3) dtype=uint8  unmodified
    deskewed: bool         # True if deskew rotation was applied
    angle:    float        # Detected skew angle in degrees; 0.0 if not applied

    def __post_init__(self):
        """Validate structural invariants after construction."""
        assert self.gray.ndim == 2,    "gray must be 2D (H, W)"
        assert self.blurred.ndim == 2, "blurred must be 2D (H, W)"
        assert self.binary.ndim == 2,  "binary must be 2D (H, W)"

        assert self.gray.shape == self.blurred.shape == self.binary.shape == self.original.shape[:2], \
            "All preprocessed images must share the same spatial dimensions"

        assert self.binary.dtype == np.uint8, "binary must be dtype uint8"

        binary_unique = set(np.unique(self.binary))
        assert binary_unique.issubset({0, 255}), \
            f"binary image must contain only 0 and 255, found: {binary_unique - {0, 255}}"

        assert self.original.dtype == np.uint8, "original must be dtype uint8"
        assert self.original.ndim == 3 and self.original.shape[2] == 3, \
            "original must be 3-channel BGR (H, W, 3)"


# ══════════════════════════════════════════════════════════════════════════════
# § WordDetection  (DATA_SCHEMA.md §2.2) — OCR Path
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class WordDetection:
    """
    A single high-confidence word extracted by Tesseract OCR.
    confidence is always normalized to [0.0, 1.0] — never stored as the raw
    Tesseract integer (0–100).
    """
    text:       str    # Recognized word. Non-empty. No leading/trailing whitespace.
    confidence: float  # [0.0, 1.0]. Always ≥ CONFIDENCE_THRESHOLD.
    bbox:       BBox   # Pixel-space bounding box.
    line_num:   int    # Line index within block.
    block_num:  int    # Block index (multi-column layout).
    word_num:   int    # Word index within the line.

    def __post_init__(self):
        if not self.text or not self.text.strip():
            raise ValueError("WordDetection.text must be non-empty")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"WordDetection.confidence must be in [0,1], got {self.confidence}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# § ObjectDetection  (DATA_SCHEMA.md §2.3) — Detection Path
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ObjectDetection:
    """
    A single high-confidence object detection from MobileNet-SSD.
    class_id = 0 (background) is never included — filtered before construction.
    """
    label:      str    # Human-readable class name from coco_labels.txt
    class_id:   int    # Integer class index. 0 < class_id < 91
    confidence: float  # [0.0, 1.0]. Always ≥ CONFIDENCE_THRESHOLD.
    bbox:       BBox   # Pixel-space bounding box.

    def __post_init__(self):
        if not self.label:
            raise ValueError("ObjectDetection.label must be non-empty")
        if self.class_id == 0:
            raise ValueError(
                "ObjectDetection.class_id=0 is background — never include in results"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"ObjectDetection.confidence must be in [0,1], got {self.confidence}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# § PipelineResult  (DATA_SCHEMA.md §2.5) — Unified Output
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineResult:
    """
    Unified output from either pipeline path. Returned to main.py for
    console printing, JSON export, and image saving.

    Invariants (verified by property checks):
        total_accepted == len(detections)
        total_accepted + total_rejected == total_raw
        threshold_used == CONFIDENCE_THRESHOLD
        annotated_image.shape == original image shape
        full_text is not None iff mode == "ocr"
    """
    mode:             str                  # "ocr" | "detection"
    input_path:       str                  # Absolute path to input image
    output_path:      str                  # Absolute path to saved output image
    detections:       list                 # List of WordDetection | ObjectDetection
    annotated_image:  np.ndarray           # BGR image with boxes drawn
    full_text:        Optional[str]        # OCR path only. None for detection.
    total_raw:        int                  # Count before confidence filter
    total_accepted:   int                  # Count after filter (== len(detections))
    total_rejected:   int                  # total_raw - total_accepted
    runtime_seconds:  float                # Wall-clock inference time
    threshold_used:   float                # CONFIDENCE_THRESHOLD value (audit)

    def __post_init__(self):
        if self.mode not in {"ocr", "detection"}:
            raise ValueError(f"mode must be 'ocr' or 'detection', got '{self.mode}'")
        if self.total_accepted != len(self.detections):
            raise ValueError(
                f"total_accepted ({self.total_accepted}) != len(detections) ({len(self.detections)})"
            )
        if self.total_accepted + self.total_rejected != self.total_raw:
            raise ValueError(
                "total_accepted + total_rejected must equal total_raw"
            )
        if self.mode == "ocr" and self.full_text is None:
            raise ValueError("full_text must not be None in OCR mode")
        if self.mode == "detection" and self.full_text is not None:
            raise ValueError("full_text must be None in detection mode")
