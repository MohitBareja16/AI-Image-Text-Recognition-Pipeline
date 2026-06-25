# ─────────────────────────────────────────────────────────────────────────────
# config.py — DecodeLabs Project 4 Configuration
# Single source of truth. No other module defines thresholds, paths, or flags.
# ─────────────────────────────────────────────────────────────────────────────

import os

# ── Pipeline Mode ─────────────────────────────────────────────────────────────
# Options: "ocr" | "detection"
# Change this constant to switch between execution paths.
PIPELINE_MODE = "ocr"

# ── Confidence Threshold ──────────────────────────────────────────────────────
# Minimum confidence for a detection/extraction to be included in output.
# MANDATORY: Must remain at 0.80 or above. Lowering this value fails Gate 3.
CONFIDENCE_THRESHOLD = 0.80

# ── Pre-Processing Flags ──────────────────────────────────────────────────────
GAUSSIAN_BLUR_KERNEL = (5, 5)     # Kernel size for noise smoothing (must be odd, ≥3)
DESKEW_ENABLED = True              # Enable/disable rotation correction
DESKEW_ANGLE_THRESHOLD = 2.0      # Degrees. Only deskew if tilt exceeds this.

# ── OCR Configuration ─────────────────────────────────────────────────────────
# PSM Modes:
#   3  = Auto (default, good for mixed layouts)
#   6  = Single uniform text block (book pages)
#   7  = Single line (number plates, headers)
#   11 = Sparse text (invoices, scattered labels)
TESSERACT_PSM = 3

# ── Model File Paths ──────────────────────────────────────────────────────────
# BASE_DIR is the src/ directory; models live one level up.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
PROTOTXT_PATH = os.path.join(MODELS_DIR, "MobileNetSSD_deploy.prototxt")
CAFFEMODEL_PATH = os.path.join(MODELS_DIR, "MobileNetSSD_deploy.caffemodel")
LABELS_PATH = os.path.join(MODELS_DIR, "coco_labels.txt")

# ── Output Configuration ──────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "output")
OUTPUT_SUFFIX = "_output"          # Appended to input filename stem
SAVE_OUTPUT_IMAGE = True           # Must be True — required for Gate 4
SHOW_OUTPUT_WINDOW = False         # Set False on headless systems (CI, SSH)

# ── Blob Parameters (Object Detection Only) ───────────────────────────────────
BLOB_SIZE = (300, 300)             # Required input size for MobileNet-SSD
BLOB_SCALE_FACTOR = 1 / 127.5     # Normalizes pixel values to [-1, 1]
BLOB_MEAN = (127.5, 127.5, 127.5) # Mean subtraction for RGB channels

# ── Pipeline Version ──────────────────────────────────────────────────────────
PIPELINE_VERSION = "1.0.0"


def validate_config() -> None:
    """
    Validates all config constants at startup.
    Raises ValueError on any violation.
    Must be called once at the beginning of main.py before any pipeline runs.
    """
    # Validate pipeline mode
    if PIPELINE_MODE not in {"ocr", "detection"}:
        raise ValueError(
            f"PIPELINE_MODE must be 'ocr' or 'detection', got: '{PIPELINE_MODE}'"
        )

    # Validate confidence threshold — Gate 3 requirement
    if CONFIDENCE_THRESHOLD < 0.80:
        raise ValueError(
            f"CONFIDENCE_THRESHOLD cannot be below 0.80. Got: {CONFIDENCE_THRESHOLD}"
        )

    # Validate gaussian kernel — must be odd integers ≥ 3
    if not all(k % 2 == 1 and k >= 3 for k in GAUSSIAN_BLUR_KERNEL):
        raise ValueError(
            "GAUSSIAN_BLUR_KERNEL values must be odd integers ≥ 3. "
            f"Got: {GAUSSIAN_BLUR_KERNEL}"
        )

    # Validate PSM mode range
    if not (0 <= TESSERACT_PSM <= 13):
        raise ValueError(
            f"TESSERACT_PSM must be between 0 and 13. Got: {TESSERACT_PSM}"
        )

    # Validate save output flag — required for Gate 4
    if not SAVE_OUTPUT_IMAGE:
        raise ValueError(
            "SAVE_OUTPUT_IMAGE must be True (required for Milestone Gate 4)"
        )

    # Validate detection model files only if detection mode is active
    if PIPELINE_MODE == "detection":
        if not os.path.isfile(PROTOTXT_PATH):
            raise ValueError(
                f"Model file not found: {PROTOTXT_PATH}\n"
                "Run: python setup_models.py"
            )
        if not os.path.isfile(CAFFEMODEL_PATH):
            raise ValueError(
                f"Model file not found: {CAFFEMODEL_PATH}\n"
                "Run: python setup_models.py"
            )
        caffemodel_size = os.path.getsize(CAFFEMODEL_PATH)
        if caffemodel_size < 20 * 1024 * 1024:
            raise ValueError(
                f"Caffemodel file appears corrupted (< 20MB): {CAFFEMODEL_PATH}\n"
                f"File size: {caffemodel_size / (1024*1024):.1f} MB. Re-run: python setup_models.py"
            )
