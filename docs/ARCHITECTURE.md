# ARCHITECTURE.md
## Project 4: AI Recognition Pipeline — System Architecture
**DecodeLabs | Batch 2026**

---

## 1. Directory Structure (Canonical)

```
project4-recognition/
│
├── README.md                    # Quick-start guide
├── PRD.md                       # Product Requirements Document
├── ARCHITECTURE.md              # This file
├── DESIGN.md                    # UI/UX and visual design spec
├── SETUP.md                     # Environment setup instructions
├── TROUBLESHOOTING.md           # Known issues and fixes
├── requirements.txt             # Pinned Python dependencies
├── setup_models.py              # One-time model download script
│
├── src/
│   ├── main.py                  # Entry point — path selection and orchestration
│   ├── config.py                # All configurable constants (single source of truth)
│   ├── preprocessor.py          # Image pre-processing pipeline module
│   ├── ocr_pipeline.py          # Path 1: Tesseract OCR logic
│   ├── detection_pipeline.py    # Path 2: MobileNet-SSD object detection logic
│   ├── postprocessor.py         # Confidence filtering, bounding box drawing, output formatting
│   └── utils.py                 # File validation, logging helpers, image I/O
│
├── models/
│   ├── MobileNetSSD_deploy.prototxt    # Network architecture definition
│   ├── MobileNetSSD_deploy.caffemodel  # Pre-trained weights (~23MB)
│   └── coco_labels.txt                 # 91-class COCO label list (one label per line)
│
├── sample_images/
│   ├── ocr_sample.jpg           # Test image for OCR path (printed text, invoice, or sign)
│   └── detection_sample.jpg     # Test image for detection path (people, vehicles, objects)
│
└── output/
    └── .gitkeep                 # Output directory (auto-created; gitignored except .gitkeep)
```

---

## 2. Module Responsibilities

### `config.py` — Single Source of Truth

All constants live here. **No other file defines thresholds, paths, or mode flags.**

```python
# ─────────────────────────────────────────────
# config.py — DecodeLabs Project 4 Configuration
# ─────────────────────────────────────────────

import os

# ── Pipeline Mode ──────────────────────────────────────────────────────────────
# Options: "ocr" | "detection"
# Change this constant to switch between execution paths.
PIPELINE_MODE = "ocr"

# ── Confidence Threshold ───────────────────────────────────────────────────────
# Minimum confidence for a detection/extraction to be included in output.
# MANDATORY: Must remain at 0.80 or above. Lowering this value fails Gate 3.
CONFIDENCE_THRESHOLD = 0.80

# ── Pre-Processing Flags ───────────────────────────────────────────────────────
GAUSSIAN_BLUR_KERNEL = (5, 5)     # Kernel size for noise smoothing
DESKEW_ENABLED = True              # Enable/disable rotation correction
DESKEW_ANGLE_THRESHOLD = 2.0      # Degrees. Only deskew if tilt exceeds this.

# ── OCR Configuration ──────────────────────────────────────────────────────────
# PSM Modes:
#   3  = Auto (default, good for mixed layouts)
#   6  = Single uniform text block (book pages)
#   7  = Single line (number plates, headers)
#   11 = Sparse text (invoices, scattered labels)
TESSERACT_PSM = 3

# ── Model File Paths ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
PROTOTXT_PATH = os.path.join(MODELS_DIR, "MobileNetSSD_deploy.prototxt")
CAFFEMODEL_PATH = os.path.join(MODELS_DIR, "MobileNetSSD_deploy.caffemodel")
LABELS_PATH = os.path.join(MODELS_DIR, "coco_labels.txt")

# ── Output Configuration ───────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "output")
OUTPUT_SUFFIX = "_output"          # Appended to input filename
SAVE_OUTPUT_IMAGE = True           # Always save to disk (required for Gate 4)
SHOW_OUTPUT_WINDOW = True          # Show cv2 window (may be False on headless systems)

# ── Blob Parameters (Object Detection Only) ───────────────────────────────────
BLOB_SIZE = (300, 300)             # Required input size for MobileNet-SSD
BLOB_SCALE_FACTOR = 1 / 127.5     # Normalizes pixel values to [-1, 1]
BLOB_MEAN = (127.5, 127.5, 127.5) # Mean subtraction for RGB channels
```

---

## 3. Data Flow Diagrams

### 3.1 Overall Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        main.py                                  │
│                                                                  │
│  1. Parse CLI args / prompt for input image path                 │
│  2. Validate file path (utils.validate_image_path)               │
│  3. Read PIPELINE_MODE from config                               │
│  4. Branch:                                                      │
│       ├── PIPELINE_MODE == "ocr"       → run_ocr_pipeline()     │
│       └── PIPELINE_MODE == "detection" → run_detection_pipeline()│
│  5. Print summary output                                         │
│  6. Save annotated image                                         │
└─────────────────────────────────────────────────────────────────┘
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
    ┌─────────────────┐           ┌──────────────────────┐
    │  ocr_pipeline   │           │  detection_pipeline   │
    │                 │           │                       │
    │ preprocessor →  │           │ preprocessor →        │
    │ pytesseract →   │           │ blob construction →   │
    │ filter(≥80%) →  │           │ net.forward() →       │
    │ draw boxes →    │           │ filter(≥80%) →        │
    │ return results  │           │ scale coords →        │
    └─────────────────┘           │ draw boxes →          │
                                  │ return results        │
                                  └──────────────────────┘
```

### 3.2 Pre-Processing Pipeline (Detailed)

```
Input: BGR Image (H × W × 3)
         │
         ▼
  ┌─────────────────────────────────┐
  │ cv2.cvtColor(BGR → GRAY)        │  H × W × 1
  │ Removes color noise             │
  └────────────────┬────────────────┘
                   │
                   ▼
  ┌─────────────────────────────────┐
  │ cv2.GaussianBlur(5×5, σ=0)      │  Smooths micro-artifacts
  │ Reduces high-frequency noise    │
  └────────────────┬────────────────┘
                   │
                   ▼ (if DESKEW_ENABLED)
  ┌─────────────────────────────────┐
  │ imutils.correct_skew()          │  Rotates image to horizontal
  │ Corrects tilt > ±2°             │
  └────────────────┬────────────────┘
                   │
                   ▼
  ┌─────────────────────────────────┐
  │ cv2.adaptiveThreshold OR        │  Binary: 0 or 255
  │ cv2.threshold (Otsu's method)   │  Sharp character contours
  └────────────────┬────────────────┘
                   │
                   ▼
         Processed Image (Binary)
       → Feed to OCR or Detection model
```

### 3.3 Object Detection: Coordinate Scaling

```
Model Output (normalized, [0.0 → 1.0]):
  box = [confidence, x_start, y_start, x_end, y_end]
                      [0.0-1.0] [0.0-1.0] [0.0-1.0] [0.0-1.0]

Actual Image Dimensions:
  W = image.shape[1]   (pixel width)
  H = image.shape[0]   (pixel height)

Scaling Formula:
  x_start_px = max(0, int(x_start * W))
  y_start_px = max(0, int(y_start * H))
  x_end_px   = min(W, int(x_end   * W))
  y_end_px   = min(H, int(y_end   * H))

Note: max(0, ...) and min(W/H, ...) prevent coordinates from
      going outside image boundaries.
```

### 3.4 OCR Confidence Extraction

```
pytesseract.image_to_data(img, output_type=Output.DATAFRAME)
  Returns DataFrame with columns:
  [level | page_num | block_num | par_num | line_num | word_num | left | top | width | height | conf | text]
                                                                                              ↑         ↑
                                                                                         -1 = not a word
                                                                                         0-100 = confidence %

Filter:
  df = df[df['conf'] >= 80]   # Keep only high-confidence words
  df = df[df['text'].str.strip() != '']  # Remove empty detections
```

---

## 4. Module Interface Contracts

### `preprocessor.py`

```python
def preprocess(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies the full pre-processing pipeline to a raw BGR image.
    
    Args:
        image: Raw BGR image loaded via cv2.imread()
    
    Returns:
        gray_thresh: Binary thresholded image (for model input)
        gray_only:   Grayscale-only image (for deskew reference)
    
    Raises:
        ValueError: If image is None or has invalid shape
    """
```

### `ocr_pipeline.py`

```python
def run_ocr(preprocessed_image: np.ndarray, original_image: np.ndarray) -> dict:
    """
    Runs Tesseract OCR on the preprocessed image.
    Filters results by CONFIDENCE_THRESHOLD.
    
    Returns:
        {
            "words": [{"text": str, "confidence": float, "bbox": (x, y, w, h)}, ...],
            "full_text": str,         # Reconstructed from high-confidence words only
            "annotated_image": np.ndarray
        }
    """
```

### `detection_pipeline.py`

```python
def run_detection(original_image: np.ndarray, net: cv2.dnn_Net, labels: list) -> dict:
    """
    Runs MobileNet-SSD object detection on the image.
    Filters results by CONFIDENCE_THRESHOLD.
    
    Returns:
        {
            "detections": [{"label": str, "confidence": float, "bbox": (x, y, w, h)}, ...],
            "annotated_image": np.ndarray
        }
    """
```

### `postprocessor.py`

```python
def draw_ocr_boxes(image: np.ndarray, words: list) -> np.ndarray:
    """Draws green bounding boxes around each detected word."""

def draw_detection_boxes(image: np.ndarray, detections: list) -> np.ndarray:
    """Draws colored bounding boxes with labels around each detected object."""

def print_summary(results: dict, mode: str, input_path: str, output_path: str) -> None:
    """Prints the standardized console output block."""

def save_output(image: np.ndarray, input_path: str) -> str:
    """Saves annotated image to output/ directory. Returns saved path."""
```

### `utils.py`

```python
def validate_image_path(path: str) -> bool:
    """
    Validates that the path exists, is readable, and has a valid image extension.
    Valid extensions: .jpg, .jpeg, .png, .bmp, .tiff
    Prints a descriptive error and returns False if invalid.
    """

def load_labels(labels_path: str) -> list:
    """Loads class labels from a .txt file. One label per line."""

def load_model(prototxt: str, caffemodel: str) -> cv2.dnn_Net:
    """Loads MobileNet-SSD model. Raises FileNotFoundError with helpful message if files missing."""
```

---

## 5. State Management

This pipeline is **stateless and single-pass**. There is no persistent state between runs.

- Each invocation of `main.py` is independent.
- No database, no session files, no caching of intermediate results.
- The only persistent output is the annotated image written to `/output/`.

This simplifies debugging: if a run fails, rerunning with the same image should produce identical results (NFR-06: Reproducibility).

---

## 6. Error Handling Strategy

All error handling follows a **fail-fast, fail-loud** principle:

| Stage | Error | Response |
|-------|-------|----------|
| File validation | Invalid path or extension | Print `ERROR: [reason]`. Exit with code 1. |
| Model loading | `.caffemodel` missing | Print `ERROR: Model file not found. Run setup_models.py`. Exit with code 1. |
| Inference | `cv2.error` during forward pass | Print `ERROR: Inference failed — [cv2 message]`. Exit with code 1. |
| Zero results | No detections above threshold | Print `WARNING: No high-confidence detections found. Try a clearer image.` Exit with code 0. |
| Output saving | Permission error on write | Print `WARNING: Could not save output image — [reason]`. Continue (Gate 4 requires the attempt). |

---

## 7. Dependency Graph

```
main.py
  ├── config.py          (no imports from project)
  ├── utils.py           → config.py
  ├── preprocessor.py    → config.py, utils.py
  ├── ocr_pipeline.py    → config.py, preprocessor.py, postprocessor.py
  ├── detection_pipeline.py → config.py, preprocessor.py, postprocessor.py, utils.py
  └── postprocessor.py   → config.py

External:
  ├── cv2 (opencv-python)
  ├── pytesseract
  ├── numpy
  ├── PIL (Pillow)
  └── imutils
```

**No circular imports. `config.py` has zero project-internal imports.**

---

*End of ARCHITECTURE.md*
