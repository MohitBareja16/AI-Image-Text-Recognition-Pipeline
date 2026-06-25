#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# main.py — Entry Point: Path Selection and Pipeline Orchestration
# Implements ARCHITECTURE.md §3.1 overall flow + DATA_SCHEMA.md §9 master flow
#
# Usage:
#   python src/main.py --image sample_images/ocr_sample.jpg
#   python src/main.py --image sample_images/detection_sample.jpg --mode detection
#   python src/main.py   (prompts for path interactively)
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import logging
import sys
import os

# Ensure project root is importable when running as `python src/main.py`
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src import config
from src.config import validate_config
from src.utils import (
    validate_image_path,
    load_image,
    load_labels,
    load_model,
    configure_logging,
)
from src.schemas import (
    PipelineError,
    InputValidationError,
    ModelLoadError,
    InferenceError,
)
from src.preprocessor import preprocess
from src.postprocessor import print_summary

logger = logging.getLogger("decodelabs.project4.main")


def main() -> int:
    """
    Main entry point for the AI Recognition Pipeline.

    Algorithm Decision Tree (DATA_SCHEMA.md §9 master flow):
        1. Parse CLI args / prompt for input image path
        2. Validate file path (utils.validate_image_path)
        3. Read PIPELINE_MODE from config
        4. Branch: OCR → run_ocr_pipeline() | Detection → run_detection_pipeline()
        5. Print summary output
        6. [Handled inside pipelines] Save annotated image
        7. Exit 0

    Returns:
        int: Exit code (0 = success, 1 = error)
    """
    # ── Configure logging ─────────────────────────────────────────────────────
    configure_logging(level=logging.INFO)

    # ── Parse CLI arguments ───────────────────────────────────────────────────
    args = _parse_args()

    # ── Validate config at startup ────────────────────────────────────────────
    # Raises ValueError immediately if any constant is misconfigured.
    try:
        # Allow CLI --mode to temporarily override config for validation
        if args.mode:
            import src.config as _cfg
            original_mode = _cfg.PIPELINE_MODE
            _cfg.PIPELINE_MODE = args.mode

        validate_config()

    except ValueError as e:
        print(f"ERROR: Configuration error — {e}", file=sys.stderr)
        return 1

    # ── Determine image path ──────────────────────────────────────────────────
    image_path = args.image
    if not image_path:
        # Interactive fallback (FR-01: CLI arg OR input() prompt)
        image_path = input("Enter path to image file: ").strip()

    # ── Validate image path (FR-02) ───────────────────────────────────────────
    if not validate_image_path(image_path):
        # validate_image_path already printed a descriptive error
        return 1

    # ── Load image ────────────────────────────────────────────────────────────
    try:
        image = load_image(image_path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    logger.info("Image loaded: %s  (shape=%s)", image_path, image.shape)

    # ── Pre-processing (mandatory for both paths) ─────────────────────────────
    logger.info("Running pre-processing pipeline...")
    try:
        bundle = preprocess(image)
        # Attach original path to bundle for downstream use
        bundle.__dict__["original_path"] = os.path.abspath(image_path)
    except ValueError as e:
        print(f"ERROR: Preprocessing failed — {e}", file=sys.stderr)
        return 1

    # ── Determine pipeline mode ───────────────────────────────────────────────
    mode = (args.mode or config.PIPELINE_MODE).lower()
    logger.info("Pipeline mode: %s", mode.upper())

    # ── Branch: OCR or Detection ──────────────────────────────────────────────
    try:
        if mode == "ocr":
            result = _run_ocr(bundle)
        elif mode == "detection":
            result = _run_detection(bundle)
        else:
            print(
                f"ERROR: Unknown mode '{mode}'. Must be 'ocr' or 'detection'.",
                file=sys.stderr,
            )
            return 1

    except (PipelineError, InferenceError, ModelLoadError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"ERROR: Unexpected pipeline failure — {e}", file=sys.stderr)
        logger.exception("Unhandled exception in pipeline:")
        return 1

    # ── Print standardized summary (DATA_SCHEMA.md §6.2) ─────────────────────
    print_summary(result)

    # ── Optional: save JSON report (G-08: Nice to Have) ──────────────────────
    if args.json:
        _save_json_report(result)

    return 0


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline runners
# ──────────────────────────────────────────────────────────────────────────────

def _run_ocr(bundle) -> "PipelineResult":
    """Runs the OCR pipeline (Path 1)."""
    from src.ocr_pipeline import run_ocr
    logger.info("Starting OCR pipeline...")
    return run_ocr(bundle)


def _run_detection(bundle) -> "PipelineResult":
    """Loads model files and runs the detection pipeline (Path 2)."""
    from src.detection_pipeline import run_detection

    logger.info("Loading MobileNet-SSD model...")
    try:
        net = load_model(config.PROTOTXT_PATH, config.CAFFEMODEL_PATH)
        labels = load_labels(config.LABELS_PATH)
    except ModelLoadError as e:
        raise

    logger.info("Starting detection pipeline...")
    return run_detection(bundle, net, labels)


# ──────────────────────────────────────────────────────────────────────────────
# JSON report (optional, G-08)
# ──────────────────────────────────────────────────────────────────────────────

def _save_json_report(result) -> None:
    """
    Saves a structured JSON report alongside the output image.
    (DATA_SCHEMA.md §6.3 optional JSON schema)
    """
    import json
    import datetime

    detections_out = []
    for idx, det in enumerate(result.detections):
        label = getattr(det, "text", None) or getattr(det, "label", "Unknown")
        b = det.bbox
        detections_out.append({
            "index": idx + 1,
            "label": label,
            "confidence": round(det.confidence, 4),
            "bbox": {
                "x1": b.x1, "y1": b.y1,
                "x2": b.x2, "y2": b.y2,
                "width": b.width, "height": b.height,
            },
        })

    report = {
        "schema_version": "1.0",
        "run_metadata": {
            "mode": result.mode,
            "input_path": result.input_path,
            "output_path": result.output_path,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "runtime_seconds": round(result.runtime_seconds, 4),
            "confidence_threshold": result.threshold_used,
            "pipeline_version": config.PIPELINE_VERSION,
        },
        "summary": {
            "total_raw_detections": result.total_raw,
            "total_accepted": result.total_accepted,
            "total_rejected": result.total_rejected,
        },
        "detections": detections_out,
        "full_text": result.full_text,
    }

    # Save alongside the output image
    import os
    if result.output_path:
        stem, _ = os.path.splitext(result.output_path)
        json_path = stem + "_report.json"
    else:
        json_path = "results.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("JSON report saved: %s", json_path)
    print(f"JSON report saved to: {json_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI argument parser
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.
    All args are optional — defaults fall back to config.py values.
    """
    parser = argparse.ArgumentParser(
        prog="decodelabs-project4",
        description="DecodeLabs AI Recognition Pipeline — OCR & Object Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python src/main.py --image sample_images/ocr_sample.jpg\n"
            "  python src/main.py --image sample_images/detection_sample.jpg --mode detection\n"
            "  python src/main.py --image invoice.jpg --mode ocr --json\n"
        ),
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        default=None,
        help="Path to input image (jpg, png, bmp, tiff). If omitted, prompts interactively.",
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["ocr", "detection"],
        default=None,
        help="Pipeline mode. Overrides PIPELINE_MODE in config.py.",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        default=False,
        help="Save a JSON results report alongside the output image.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        default=False,
        help="Suppress the cv2 display window (useful on headless servers).",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.exit(main())
