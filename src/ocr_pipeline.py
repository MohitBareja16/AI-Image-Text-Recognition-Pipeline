# ─────────────────────────────────────────────────────────────────────────────
# ocr_pipeline.py — Path 1: Tesseract OCR Logic
# Implements ARCHITECTURE.md §4 ocr_pipeline contract + ALGORITHM_SPEC.md §2
# Uses pytesseract.image_to_data() (NOT image_to_string()) per FR-OCR-01.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import time
import logging
import sys

import cv2
import numpy as np
import pandas as pd

try:
    import pytesseract
    from pytesseract import Output
except ImportError:
    print(
        "ERROR: pytesseract is not installed. Run: pip install pytesseract\n"
        "Also ensure Tesseract binary is installed — see docs/SETUP.md",
        file=sys.stderr,
    )
    pytesseract = None  # type: ignore

from src import config
from src.schemas import (
    PreprocessedBundle,
    PipelineResult,
    WordDetection,
    BBox,
    InferenceError,
)
from src.postprocessor import draw_ocr_boxes, save_output

logger = logging.getLogger("decodelabs.project4.ocr_pipeline")


def run_ocr(bundle: PreprocessedBundle) -> PipelineResult:
    """
    Runs the full OCR pipeline on a preprocessed image bundle.

    Pipeline steps:
        1. PSM mode selection (config or heuristic)
        2. Tesseract image_to_data() → DataFrame (MANDATORY per FR-OCR-01)
        3. DataFrame filtering (ALGORITHM_SPEC.md §2.3, DATA_SCHEMA.md §4)
        4. Text reconstruction (ALGO §2.4)
        5. Bounding box annotation on original image
        6. PipelineResult construction

    Args:
        bundle: PreprocessedBundle from preprocessor.preprocess()

    Returns:
        PipelineResult with mode="ocr"

    Raises:
        InferenceError: If Tesseract fails to run.
    """
    if pytesseract is None:
        raise InferenceError(
            stage="inference",
            message="pytesseract not installed. Run: pip install pytesseract",
            recoverable=True,
        )

    start_time = time.perf_counter()

    # ── Step 1: PSM mode selection ────────────────────────────────────────────
    psm = _select_psm(bundle.binary)
    logger.info("Using Tesseract PSM mode %d", psm)

    # ── Step 2: Tesseract OCR → DataFrame (FR-OCR-01: image_to_data ONLY) ────
    # The binary image (preprocessed) is passed to Tesseract — not the raw BGR.
    # This satisfies TC-ADV-005: OCR must receive preprocessed not raw image.
    logger.info("Running Tesseract OCR (image_to_data)...")
    try:
        config_str = f"--psm {psm}"
        raw_df: pd.DataFrame = pytesseract.image_to_data(
            bundle.binary,  # Preprocessed binary image — NOT original BGR
            config=config_str,
            output_type=Output.DATAFRAME,
        )
    except Exception as e:
        raise InferenceError(
            stage="inference",
            message=f"Tesseract OCR failed: {e}",
            recoverable=False,
        )

    total_raw = len(raw_df[raw_df["conf"] != -1])
    logger.debug("Raw Tesseract output: %d candidate rows", len(raw_df))

    # ── Step 3: DataFrame filtering (DATA_SCHEMA.md §4 — in exact order) ─────
    filtered_df = _filter_dataframe(raw_df)
    logger.info(
        "Confidence filter: %d accepted / %d rejected",
        len(filtered_df),
        total_raw - len(filtered_df),
    )

    total_accepted = len(filtered_df)
    total_rejected = total_raw - total_accepted

    # ── Step 4: Convert rows to WordDetection objects ─────────────────────────
    word_detections = _build_word_detections(filtered_df, bundle.binary)

    # ── Step 5: Reconstruct full text from high-confidence words ──────────────
    # FR-OCR-03: preserve line groupings
    full_text = _reconstruct_text(filtered_df)

    # ── Step 6: Annotate image with bounding boxes ───────────────────────────
    # Boxes are drawn on the ORIGINAL BGR image for color accuracy.
    annotated = draw_ocr_boxes(bundle.original, word_detections)

    runtime = time.perf_counter() - start_time
    logger.info("OCR pipeline complete. Runtime: %.2fs", runtime)

    # Zero-results logging (FR-09)
    if total_accepted == 0:
        logger.warning("No high-confidence detections found. Try a clearer image.")

    # ── Step 7: Save output image ─────────────────────────────────────────────
    output_path = _attempt_save(annotated, bundle)

    # ── Construct unified PipelineResult ──────────────────────────────────────
    return PipelineResult(
        mode="ocr",
        input_path=bundle.original_path if hasattr(bundle, "original_path") else "",
        output_path=output_path,
        detections=word_detections,
        annotated_image=annotated,
        full_text=full_text if full_text else "",
        total_raw=total_raw,
        total_accepted=total_accepted,
        total_rejected=total_rejected,
        runtime_seconds=runtime,
        threshold_used=config.CONFIDENCE_THRESHOLD,
    )


def _select_psm(image: np.ndarray) -> int:
    """
    Returns the PSM mode to use for Tesseract.
    Config always wins (ALGO §2.2 — 'Config always wins').
    Falls back to a heuristic based on image aspect ratio.
    """
    # Config value takes priority
    if config.TESSERACT_PSM is not None:
        return config.TESSERACT_PSM

    # Heuristic: choose PSM based on aspect ratio
    h, w = image.shape[:2]
    aspect_ratio = w / h

    if aspect_ratio > 5.0:
        return 7   # Very wide, short → single line
    elif aspect_ratio < 0.5:
        return 4   # Tall, narrow → single column
    else:
        return 3   # Default: fully automatic layout


def _filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the mandatory 5-step filtering sequence to the Tesseract DataFrame.
    (DATA_SCHEMA.md §4 — must apply steps IN THIS EXACT ORDER)

    Step 1: Remove non-word rows (conf == -1)
    Step 2: Remove rows where text is empty or whitespace
    Step 3: Remove rows where width or height is 0 (degenerate boxes)
    Step 4: Apply confidence threshold (Tesseract 0–100 → threshold * 100)
    Step 5: Reset index
    """
    # Step 1: Remove layout-level rows (conf=-1 means block/para/line, not word)
    df = df[df["conf"] != -1].copy()

    # Step 2: Remove rows where text is empty or whitespace-only
    df = df[df["text"].str.strip() != ""].copy()

    # Step 3: Remove degenerate boxes (width=0 or height=0)
    df = df[(df["width"] > 0) & (df["height"] > 0)].copy()

    # Step 4: Apply confidence threshold
    # CRITICAL: Tesseract conf is 0–100 int; CONFIDENCE_THRESHOLD is 0.0–1.0 float.
    # Multiply threshold × 100 before comparison. (ALGO §2.3 critical note)
    threshold_int = config.CONFIDENCE_THRESHOLD * 100  # = 80
    df = df[df["conf"] >= threshold_int].copy()

    # Step 5: Reset index for clean iteration
    df = df.reset_index(drop=True)

    return df


def _build_word_detections(df: pd.DataFrame, image: np.ndarray) -> list:
    """
    Converts filtered DataFrame rows into WordDetection objects.
    Normalizes Tesseract confidence from 0–100 to 0.0–1.0 (V-02).
    """
    h, w = image.shape[:2]
    word_detections = []

    for _, row in df.iterrows():
        # Normalize confidence from 0–100 → 0.0–1.0 (DATA_SCHEMA.md V-02)
        confidence_normalized = float(row["conf"]) / 100.0

        # Clamp to [0, 1] for numerical safety
        confidence_normalized = min(1.0, max(0.0, confidence_normalized))

        # Build pixel-space BBox from Tesseract's left/top/width/height
        x1 = max(0, int(row["left"]))
        y1 = max(0, int(row["top"]))
        x2 = min(w, int(row["left"]) + int(row["width"]))
        y2 = min(h, int(row["top"]) + int(row["height"]))

        # Skip degenerate boxes (shouldn't happen after step 3, but be safe)
        if x1 >= x2 or y1 >= y2:
            continue

        try:
            bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2)
        except (ValueError, TypeError):
            logger.warning("Skipping invalid bbox: (%d,%d,%d,%d)", x1, y1, x2, y2)
            continue

        word_detections.append(WordDetection(
            text=str(row["text"]).strip(),
            confidence=confidence_normalized,
            bbox=bbox,
            line_num=int(row["line_num"]),
            block_num=int(row["block_num"]),
            word_num=int(row["word_num"]),
        ))

    return word_detections


def _reconstruct_text(df: pd.DataFrame) -> str:
    """
    Reconstructs multi-line text from the filtered word DataFrame.
    Groups words by (block_num, par_num, line_num) to preserve structure.
    Words within the same line are joined with a space.
    Lines are joined with a newline character.
    (ALGORITHM_SPEC.md §2.4)
    """
    if df.empty:
        return ""

    lines = []
    # Group by structural hierarchy to preserve line groupings (FR-OCR-03)
    for _, group in df.groupby(["block_num", "par_num", "line_num"]):
        line_words = group.sort_values("word_num")["text"].tolist()
        line_text = " ".join(str(w).strip() for w in line_words if str(w).strip())
        if line_text:
            lines.append(line_text)

    return "\n".join(lines)


def _attempt_save(annotated: np.ndarray, bundle: PreprocessedBundle) -> str:
    """
    Attempts to save the annotated image. Returns the output path.
    On failure, prints a warning and returns an empty string (non-fatal).
    """
    try:
        # Use original_path attribute if available; else use a placeholder
        input_path = getattr(bundle, "original_path", "unknown.jpg")
        return save_output(annotated, input_path)
    except Exception as e:
        logger.warning("Could not save output image: %s", e)
        return ""


# ── Public helper for filter_dataframe (used in tests) ───────────────────────
def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Public alias for _filter_dataframe, used in test suite."""
    return _filter_dataframe(df)
