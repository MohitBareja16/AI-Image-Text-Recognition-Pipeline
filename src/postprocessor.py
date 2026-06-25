# ─────────────────────────────────────────────────────────────────────────────
# postprocessor.py — Confidence Filtering, Image Annotation, Console Output, Save
# Implements ARCHITECTURE.md §4 postprocessor contract + DATA_SCHEMA.md §6 + DESIGN.md §8
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import logging
import cv2
import numpy as np

from src import config
from src.schemas import PipelineResult
from src.utils import ensure_output_dir

logger = logging.getLogger("decodelabs.project4.postprocessor")

# ── Drawing constants (DESIGN.md §8.1) ───────────────────────────────────────
# Colors are BGR format for OpenCV. High confidence (≥90%) gets cyan/teal;
# medium confidence (80–89%) gets blue.
COLOR_HIGH_CONFIDENCE: tuple = (255, 200, 0)   # Cyan/Teal — ≥ 90%
COLOR_MED_CONFIDENCE: tuple  = (255, 150, 50)  # Blue — 80–89%
COLOR_TEXT_ON_BOX: tuple     = (255, 255, 255) # White label text

# Text rendering constants (DESIGN.md §8.2)
FONT            = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE      = 0.6
FONT_THICKNESS  = 2
FONT_LINE       = cv2.LINE_AA   # Always anti-aliased (DESIGN.md §8.2)
LABEL_PADDING   = 4             # Pixels above/below text in label background rect


# ══════════════════════════════════════════════════════════════════════════════
# Annotation Functions
# ══════════════════════════════════════════════════════════════════════════════

def draw_ocr_boxes(image: np.ndarray, words: list) -> np.ndarray:
    """
    Draws green bounding boxes and confidence labels around each detected word.
    Operates on a COPY — never modifies the original (DATA_SCHEMA.md V-01).

    Args:
        image: Original BGR image.
        words: List of WordDetection objects.

    Returns:
        np.ndarray: Annotated BGR image, same shape as input.
    """
    # Always copy — never modify the original in place (V-01)
    annotated = image.copy()

    for word in words:
        b = word.bbox
        confidence = word.confidence

        # Select box color based on confidence tier (DESIGN.md §6)
        color = COLOR_HIGH_CONFIDENCE if confidence >= 0.90 else COLOR_MED_CONFIDENCE
        thickness = 3 if confidence >= 0.90 else 2

        # Draw bounding box rectangle
        cv2.rectangle(annotated, (b.x1, b.y1), (b.x2, b.y2), color, thickness)

        # Format label: "word: 91.3%"
        label_str = f"{word.text}: {confidence * 100:.1f}%"

        # Compute safe label position (clamped above the box, or inside if near top)
        text_pos, bg_rect = _compute_label_position(b.x1, b.y1, label_str)

        # Draw label background (semi-transparent overlay for legibility)
        _draw_label_background(annotated, bg_rect, color)

        # Draw label text
        cv2.putText(
            annotated, label_str, text_pos,
            FONT, FONT_SCALE, COLOR_TEXT_ON_BOX, FONT_THICKNESS, FONT_LINE,
        )

    return annotated


def draw_detection_boxes(image: np.ndarray, detections: list) -> np.ndarray:
    """
    Draws colored bounding boxes with labels around each detected object.
    Label format: "ClassName: Confidence%"  e.g. "Person: 91.3%"
    Operates on a COPY — never modifies the original (DATA_SCHEMA.md V-01).

    Args:
        image: Original BGR image.
        detections: List of ObjectDetection objects.

    Returns:
        np.ndarray: Annotated BGR image, same shape as input.
    """
    # Always copy — never modify the original in place (V-01)
    annotated = image.copy()

    for det in detections:
        b = det.bbox
        confidence = det.confidence

        # Select color based on confidence tier
        color = COLOR_HIGH_CONFIDENCE if confidence >= 0.90 else COLOR_MED_CONFIDENCE
        thickness = 3 if confidence >= 0.90 else 2

        # Draw bounding box
        cv2.rectangle(annotated, (b.x1, b.y1), (b.x2, b.y2), color, thickness)

        # Format label: "Person: 91.3%"
        label_str = f"{det.label}: {confidence * 100:.1f}%"

        # Compute safe label position
        text_pos, bg_rect = _compute_label_position(b.x1, b.y1, label_str)

        # Draw label background
        _draw_label_background(annotated, bg_rect, color)

        # Draw label text
        cv2.putText(
            annotated, label_str, text_pos,
            FONT, FONT_SCALE, COLOR_TEXT_ON_BOX, FONT_THICKNESS, FONT_LINE,
        )

    return annotated


# ══════════════════════════════════════════════════════════════════════════════
# Console Output
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(result: PipelineResult) -> None:
    """
    Prints the standardized console output block as specified in DATA_SCHEMA.md §6.2.

    Format:
        ==================================================
        === DecodeLabs Project 4 — Recognition Output ===
        ==================================================
        Mode:       OCR
        Input:      /path/to/image.jpg
        Threshold:  80%
        ──────────────────────────────────────────────────
        [01] Hello               | 95.0%
        [02] Test                | 82.0%
        ──────────────────────────────────────────────────
        Total High-Confidence Results: 2
        Total Rejected (below 80%): 3
        Runtime: 1.43s
        Output saved to: /path/to/output/image_output.jpg
        ==================================================
    """
    SEP = "=" * 50
    DIV = "─" * 50

    print(SEP)
    print("=== DecodeLabs Project 4 — Recognition Output ===")
    print(SEP)
    print(f"Mode:       {result.mode.upper()}")
    print(f"Input:      {result.input_path}")
    print(f"Threshold:  {result.threshold_used:.0%}")
    print(DIV)

    if result.total_accepted == 0:
        # FR-09: Explicit zero-results message (not a silent exit)
        print("WARNING: No high-confidence detections found.")
        print("         Tip: Try a clearer image or adjust PSM mode (OCR) / "
              "check object visibility (Detection).")
    else:
        for idx, det in enumerate(result.detections):
            # Resolve label: WordDetection has .text; ObjectDetection has .label
            label = getattr(det, "text", None) or getattr(det, "label", "Unknown")
            conf = det.confidence
            print(f"[{idx + 1:02d}] {label:<20} | {conf:.1%}")

    print(DIV)
    print(f"Total High-Confidence Results: {result.total_accepted}")
    print(f"Total Rejected (below {result.threshold_used:.0%}): {result.total_rejected}")
    print(f"Runtime: {result.runtime_seconds:.2f}s")
    print(f"Output saved to: {result.output_path}")
    print(SEP)

    # If OCR mode, also print the reconstructed text block
    if result.mode == "ocr" and result.full_text:
        print("\n── Extracted Text ──────────────────────────────")
        print(result.full_text)
        print("────────────────────────────────────────────────")


# ══════════════════════════════════════════════════════════════════════════════
# Output File Saving
# ══════════════════════════════════════════════════════════════════════════════

def save_output(image: np.ndarray, input_path: str) -> str:
    """
    Saves the annotated image to the output/ directory.
    Filename: <original_stem>_output.<original_extension>
    Overwrites any existing file with the same name (NFR-06: determinism).

    Args:
        image:      Annotated BGR image (np.ndarray).
        input_path: Original input image path (used to derive output filename).

    Returns:
        str: Absolute path where the image was saved.

    Raises:
        OutputError: If the file cannot be written (permission, disk full, etc.)
    """
    from src.schemas import OutputError

    # Ensure output directory exists (creates it if needed)
    output_dir = ensure_output_dir()

    # Derive output filename: invoice.jpg → invoice_output.jpg
    stem, ext = os.path.splitext(os.path.basename(input_path))
    output_filename = f"{stem}{config.OUTPUT_SUFFIX}{ext}"
    output_path = os.path.join(output_dir, output_filename)

    # Save annotated image to disk (cv2.imwrite — Gate 4 requirement)
    try:
        success = cv2.imwrite(output_path, image)
        if not success:
            raise OSError(f"cv2.imwrite returned False for path: {output_path}")
    except Exception as e:
        # Print warning but do not hard-exit — Gate 4 requires the *attempt* be made
        import sys
        print(
            f"WARNING: Could not save output image — {e}",
            file=sys.stderr,
        )
        raise OutputError(
            stage="output",
            message=f"Could not save output image: {e}",
            recoverable=True,
        )

    logger.info("Output image saved: %s", output_path)
    return os.path.abspath(output_path)


# ══════════════════════════════════════════════════════════════════════════════
# Private Drawing Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _compute_label_position(
    x_start: int, y_start: int, label: str
) -> tuple[tuple, tuple]:
    """
    Computes where to place the label text and its background rectangle.
    Moves label inside the box if it would render above the image boundary.
    (ALGORITHM_SPEC.md §6.2)

    Returns:
        (text_pos, bg_rect):
            text_pos = (x, y) pixel position for cv2.putText baseline
            bg_rect  = (bg_x1, bg_y1, bg_x2, bg_y2) for background rectangle
    """
    text_size, _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
    text_w, text_h = text_size

    # Preferred: label floats above the box top edge
    label_y = y_start - 10

    # If label would go above the image top, place it inside the box
    if label_y - text_h < 0:
        label_y = y_start + text_h + LABEL_PADDING + 5

    label_x = x_start

    # Background rectangle bounds
    bg_x1 = label_x
    bg_y1 = label_y - text_h - LABEL_PADDING
    bg_x2 = label_x + text_w
    bg_y2 = label_y + LABEL_PADDING

    return (label_x, label_y), (bg_x1, bg_y1, bg_x2, bg_y2)


def _draw_label_background(
    image: np.ndarray,
    bg_rect: tuple,
    color: tuple,
) -> None:
    """
    Draws a semi-transparent filled rectangle as the label background.
    Uses alpha blending at 60% opacity (DESIGN.md §8 / ALGO §7).

    Modifies `image` in place (image is already a copy at this point).
    """
    bg_x1, bg_y1, bg_x2, bg_y2 = bg_rect

    # Clamp coordinates to image boundaries (DESIGN.md §8.3)
    h, w = image.shape[:2]
    bg_x1 = max(0, bg_x1)
    bg_y1 = max(0, bg_y1)
    bg_x2 = min(w, bg_x2)
    bg_y2 = min(h, bg_y2)

    if bg_x2 <= bg_x1 or bg_y2 <= bg_y1:
        # Degenerate rectangle — skip drawing
        return

    # Alpha blend: overlay (60%) + original (40%)
    overlay = image.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), color, cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
