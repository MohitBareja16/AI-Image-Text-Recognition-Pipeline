# DATA_SCHEMA.md
## Project 4: Data Schema Specification
**DecodeLabs | Batch 2026**
**Scope:** Every data structure, type contract, file format, and inter-module data exchange in the pipeline — specified precisely enough to catch type mismatches, shape errors, and schema violations before runtime.

---

## Table of Contents
1. [Input Schemas](#1-input-schemas)
2. [Internal Data Structures](#2-internal-data-structures)
3. [Model Output Schemas](#3-model-output-schemas)
4. [OCR DataFrame Schema](#4-ocr-dataframe-schema)
5. [Inter-Module Exchange Schema](#5-inter-module-exchange-schema)
6. [Output File Schemas](#6-output-file-schemas)
7. [Config Schema](#7-config-schema)
8. [Error Object Schema](#8-error-object-schema)
9. [Validation Rules (Cross-Cutting)](#9-validation-rules-cross-cutting)

---

## 1. Input Schemas

### 1.1 CLI / User Input

```
Schema: ImageInputRequest
─────────────────────────────────────────────────────
Field         Type      Required  Validation
─────────────────────────────────────────────────────
image_path    str       YES       - Must not be empty string
                                  - Must point to an existing file
                                  - File extension must be one of:
                                    .jpg, .jpeg, .png, .bmp, .tiff
                                  - File must be readable (no permission error)
                                  - File size must be > 0 bytes
                                  - File size must be < 50 MB
─────────────────────────────────────────────────────
```

**Rejected input examples (must raise ValidationError with descriptive message):**
```
""                        → "Image path cannot be empty."
"image.pdf"               → "Unsupported format: .pdf. Use jpg, jpeg, png, bmp, or tiff."
"/nonexistent/path.jpg"   → "File not found: /nonexistent/path.jpg"
"empty.jpg" (0 bytes)     → "File is empty: empty.jpg"
"huge.jpg" (> 50 MB)      → "File too large (>50 MB). Use a smaller image."
"locked.jpg" (no read)    → "Permission denied: locked.jpg"
```

### 1.2 Raw Image (after cv2.imread)

```
Schema: RawImage
─────────────────────────────────────────────────────
Field         Type             Constraints
─────────────────────────────────────────────────────
array         np.ndarray       - dtype: uint8
                                - ndim: 3
                                - shape: (H, W, 3) where:
                                    H ≥ 32   (minimum height)
                                    W ≥ 32   (minimum width)
                                    C = 3    (BGR channels, exactly)
                                - No NaN or Inf values
                                - All values in [0, 255]
─────────────────────────────────────────────────────
```

**Failure case:** If `cv2.imread()` returns `None`, the array is `None` — not an ndarray. This must be caught explicitly before any `.shape` access.

```python
# CORRECT check:
image = cv2.imread(path)
if image is None:
    raise FileNotFoundError(...)

# WRONG — will raise AttributeError, not a useful message:
if image.shape is None: ...
```

---

## 2. Internal Data Structures

### 2.1 PreprocessedBundle

Returned by `preprocessor.preprocess()`. Contains all intermediate images needed downstream.

```python
@dataclass
class PreprocessedBundle:
    gray:        np.ndarray   # Grayscale image.  shape=(H,W),   dtype=uint8
    blurred:     np.ndarray   # After GaussianBlur. shape=(H,W), dtype=uint8
    binary:      np.ndarray   # After thresholding. shape=(H,W), dtype=uint8, values={0,255} only
    deskewed:    bool         # True if deskew was applied
    angle:       float        # Detected skew angle in degrees. 0.0 if deskew not applied.
    original:    np.ndarray   # Original BGR image (reference, unmodified). shape=(H,W,3), dtype=uint8
```

**Invariants that must hold after preprocessing:**
```
assert gray.ndim == 2
assert blurred.ndim == 2
assert binary.ndim == 2
assert gray.shape == blurred.shape == binary.shape == original.shape[:2]
assert binary.dtype == np.uint8
assert set(np.unique(binary)).issubset({0, 255})   # Binary — no other values
assert original.dtype == np.uint8
assert original.ndim == 3 and original.shape[2] == 3
```

---

### 2.2 WordDetection (OCR Path)

A single high-confidence word extracted by Tesseract.

```python
@dataclass
class WordDetection:
    text:        str     # The recognized word string. Non-empty. No leading/trailing whitespace.
    confidence:  float   # Normalized to [0.0, 1.0]. ALWAYS ≥ CONFIDENCE_THRESHOLD.
    bbox:        BBox    # Pixel-space bounding box (see 2.4)
    line_num:    int     # Line index within the block (for text reconstruction)
    block_num:   int     # Block index (for multi-column layout handling)
    word_num:    int     # Word index within the line (for ordering)
```

**Constraints:**
```
text:        len(text.strip()) > 0       # No empty or whitespace-only words
confidence:  0.80 ≤ confidence ≤ 1.0    # Already filtered — must meet threshold
bbox:        valid BBox (see 2.4)
line_num:    ≥ 0
block_num:   ≥ 0
word_num:    ≥ 1
```

---

### 2.3 ObjectDetection (Detection Path)

A single high-confidence object detection from MobileNet-SSD.

```python
@dataclass
class ObjectDetection:
    label:       str     # Human-readable class name from coco_labels.txt
    class_id:    int     # Integer class index. 0 ≤ class_id < len(labels)
    confidence:  float   # [0.0, 1.0]. ALWAYS ≥ CONFIDENCE_THRESHOLD.
    bbox:        BBox    # Pixel-space bounding box (see 2.4)
```

**Constraints:**
```
label:       len(label) > 0 and label != "background"   # class_id=0 is background; skip it
class_id:    0 < class_id < 91  (for COCO 90-class model)
confidence:  0.80 ≤ confidence ≤ 1.0
bbox:        valid BBox AND non-zero area
```

**Critical:** `class_id = 0` is the background class in MobileNet-SSD. Detections with `class_id = 0` must be skipped even if confidence ≥ 0.80.

---

### 2.4 BBox (Bounding Box)

Shared by both paths. Always in **pixel coordinates** at this level (normalized floats are converted before creating a BBox).

```python
@dataclass
class BBox:
    x1: int   # Left edge (pixels from left). 0 ≤ x1 < image_width
    y1: int   # Top edge (pixels from top).   0 ≤ y1 < image_height
    x2: int   # Right edge (pixels).          x1 < x2 ≤ image_width
    y2: int   # Bottom edge (pixels).         y1 < y2 ≤ image_height
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1   # Always > 0
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1   # Always > 0
    
    @property
    def area(self) -> int:
        return self.width * self.height   # Always > 0
    
    def to_xywh(self) -> tuple:
        """For cv2.dnn.NMSBoxes which expects [x, y, w, h]"""
        return (self.x1, self.y1, self.width, self.height)
```

**Hard invariants (validated on construction):**
```
0   ≤ x1 < x2 ≤ image_width
0   ≤ y1 < y2 ≤ image_height
width  = x2 - x1 > 0
height = y2 - y1 > 0
area   > 0
All values: int (not float)
```

---

### 2.5 PipelineResult

The unified output object from either pipeline path. Returned to `main.py` for printing and saving.

```python
@dataclass
class PipelineResult:
    mode:             str                       # "ocr" | "detection"
    input_path:       str                       # Absolute path to input image
    output_path:      str                       # Absolute path to saved output image
    detections:       list                      # List of WordDetection | ObjectDetection
    annotated_image:  np.ndarray                # BGR image with boxes drawn
    full_text:        Optional[str]             # OCR path only. None for detection.
    total_raw:        int                       # Count before confidence filter
    total_accepted:   int                       # Count after confidence filter (= len(detections))
    total_rejected:   int                       # total_raw - total_accepted
    runtime_seconds:  float                     # Wall-clock inference time
    threshold_used:   float                     # CONFIDENCE_THRESHOLD value used (for audit)
```

**Invariants:**
```
mode in {"ocr", "detection"}
total_accepted == len(detections)
total_accepted + total_rejected == total_raw
total_accepted ≥ 0
total_raw ≥ total_accepted ≥ 0
runtime_seconds > 0.0
threshold_used == 0.80   (auditable)
annotated_image.shape == original_image.shape  (same dimensions as input)
IF mode == "ocr":      full_text is not None
IF mode == "detection": full_text is None
```

---

## 3. Model Output Schemas

### 3.1 MobileNet-SSD Raw Output

Direct output of `net.forward()`.

```
Schema: MobileNetRawOutput
─────────────────────────────────────────────────────────────────────
Field         Type              Shape            Value Range
─────────────────────────────────────────────────────────────────────
detections    np.ndarray        (1, 1, N, 7)     float32

Where N = number of candidate detections (typically 100–300 before NMS)

detections[0, 0, i] layout:
  Index 0: image_id    float32   Always 0.0 for single-image inference
  Index 1: class_id    float32   [0.0, 90.0] — cast to int for label lookup
  Index 2: confidence  float32   [0.0, 1.0]
  Index 3: x_start     float32   [0.0, 1.0] normalized
  Index 4: y_start     float32   [0.0, 1.0] normalized
  Index 5: x_end       float32   [0.0, 1.0] normalized
  Index 6: y_end       float32   [0.0, 1.0] normalized
─────────────────────────────────────────────────────────────────────
```

**Common malformed outputs and how to handle:**
```
x_start > x_end → Box is inverted (rare bug). Skip this detection entirely.
y_start > y_end → Same. Skip.
confidence > 1.0 → Numerical instability. Clamp to 1.0.
class_id = 0    → Background class. Always skip regardless of confidence.
x_start < 0 or x_end > 1 → Out-of-bounds normalized coords. Clamp, then scale.
```

---

## 4. OCR DataFrame Schema

Output of `pytesseract.image_to_data(..., output_type=Output.DATAFRAME)`.

```
Schema: TesseractDataFrame
─────────────────────────────────────────────────────────────────────
Column        dtype     Value Range     Description
─────────────────────────────────────────────────────────────────────
level         int64     1–5             Hierarchy: Page/Block/Para/Line/Word
page_num      int64     ≥ 1             Page number (always 1 for single images)
block_num     int64     ≥ 0             Block index
par_num       int64     ≥ 0             Paragraph index within block
line_num      int64     ≥ 0             Line index within paragraph
word_num      int64     ≥ 0             Word index within line (0=line-level row)
left          int64     ≥ 0             Bounding box left edge (pixels)
top           int64     ≥ 0             Bounding box top edge (pixels)
width         int64     ≥ 0             Bounding box width (pixels)
height        int64     ≥ 0             Bounding box height (pixels)
conf          float64   -1 or [0,100]   -1 = layout row (not a word)
                                         0–100 = word confidence (integer-valued float)
text          object    str             Recognized text. May be empty string "".
─────────────────────────────────────────────────────────────────────
```

**Filtering sequence (must be applied in this exact order):**
```python
# Step 1: Remove non-word rows (layout hierarchy rows return conf=-1)
df = df[df['conf'] != -1].copy()

# Step 2: Remove rows where text is empty or whitespace
df = df[df['text'].str.strip() != ''].copy()

# Step 3: Remove rows where width or height is 0 (degenerate boxes)
df = df[(df['width'] > 0) & (df['height'] > 0)].copy()

# Step 4: Apply confidence threshold (conf column is 0-100, threshold is 0-1 scale → multiply)
df = df[df['conf'] >= (CONFIDENCE_THRESHOLD * 100)].copy()

# Step 5: Reset index
df = df.reset_index(drop=True)
```

**Why `.copy()` after each filter:** Pandas warns about "SettingWithCopyWarning" when chaining operations on slices without copying. The `.copy()` calls prevent this and make each filtered state an independent DataFrame.

---

## 5. Inter-Module Exchange Schema

This defines what each module receives and what it must return. Violating these contracts causes runtime TypeErrors or silent data corruption.

```
┌─────────────────────────────────────────────────────────────────────┐
│  utils.validate_image_path(path: str) → bool                        │
│    IN:  str                                                          │
│    OUT: True (valid) | False (invalid, message already printed)      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  utils.load_image(path: str) → np.ndarray                           │
│    IN:  str (validated path)                                        │
│    OUT: np.ndarray  shape=(H,W,3)  dtype=uint8                      │
│    RAISES: FileNotFoundError if imread returns None                 │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  preprocessor.preprocess(image: np.ndarray) → PreprocessedBundle    │
│    IN:  np.ndarray  shape=(H,W,3)  dtype=uint8                      │
│    OUT: PreprocessedBundle (see §2.1)                               │
│    RAISES: ValueError if image is None or wrong shape/dtype         │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  ocr_pipeline.run_ocr(bundle: PreprocessedBundle) → PipelineResult  │
│    IN:  PreprocessedBundle                                          │
│    OUT: PipelineResult  (mode="ocr")                                │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  detection_pipeline.run_detection(                                  │
│      bundle: PreprocessedBundle,                                    │
│      net: cv2.dnn_Net,                                              │
│      labels: list[str]                                              │
│  ) → PipelineResult                                                 │
│    IN:  PreprocessedBundle + loaded net + labels list               │
│    OUT: PipelineResult  (mode="detection")                          │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  postprocessor.save_output(                                         │
│      image: np.ndarray,                                             │
│      input_path: str                                                │
│  ) → str                                                            │
│    IN:  annotated BGR image + original input path                   │
│    OUT: str — absolute path where image was saved                   │
│    RAISES: IOError (caught internally; warning printed)             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6. Output File Schemas

### 6.1 Annotated Output Image

```
Schema: OutputImageFile
─────────────────────────────────────────────────────────────────────
Filename:    <original_name>_output.<original_extension>
             Example: invoice.jpg → invoice_output.jpg
Location:    /output/  (relative to project root)
Format:      JPEG (cv2.imwrite default) OR PNG if input was PNG
Color space: BGR (OpenCV native — same as input)
Shape:       Identical to input image (H, W, 3)
dtype:       uint8
─────────────────────────────────────────────────────────────────────
```

**Naming collision handling:** If `invoice_output.jpg` already exists, overwrite it. Do not append `_1`, `_2` etc. — deterministic output names (NFR-06).

### 6.2 Console Output (stdout)

```
Schema: ConsoleOutputBlock

Line 1:   "=" × 50
Line 2:   "=== DecodeLabs Project 4 — Recognition Output ==="
Line 3:   "=" × 50
Line 4:   f"Mode:       {mode.upper()}"
Line 5:   f"Input:      {input_path}"
Line 6:   f"Threshold:  {threshold_used:.0%}"
Line 7:   "─" × 50
Lines 8+: FOR EACH detection:
          f"[{index+1:02d}] {label:<20} | {confidence:.1%}"
          Examples:
          "[01] Person               | 91.3%"
          "[02] Car                  | 84.7%"
Line N:   "─" × 50
Line N+1: f"Total High-Confidence Results: {total_accepted}"
Line N+2: f"Total Rejected (below {threshold_used:.0%}): {total_rejected}"
Line N+3: f"Runtime: {runtime_seconds:.2f}s"
Line N+4: f"Output saved to: {output_path}"
Line N+5: "=" × 50
```

**Zero-results variant (replaces lines 8+ through N):**
```
"WARNING: No high-confidence detections found."
"         Tip: Try a clearer image or adjust PSM mode (OCR) / check object visibility (Detection)."
```

### 6.3 Optional JSON Report

```json
{
  "schema_version": "1.0",
  "run_metadata": {
    "mode": "ocr",
    "input_path": "/absolute/path/to/image.jpg",
    "output_path": "/absolute/path/to/output/image_output.jpg",
    "timestamp": "2026-06-24T14:30:00",
    "runtime_seconds": 1.43,
    "confidence_threshold": 0.80,
    "pipeline_version": "1.0.0"
  },
  "summary": {
    "total_raw_detections": 12,
    "total_accepted": 8,
    "total_rejected": 4
  },
  "detections": [
    {
      "index": 1,
      "label": "Invoice",
      "confidence": 0.913,
      "bbox": {
        "x1": 124,
        "y1": 88,
        "x2": 340,
        "y2": 112,
        "width": 216,
        "height": 24
      }
    }
  ],
  "full_text": "Invoice\nDate: 2026-01-15\nTotal: $480.00"
}
```

**JSON field types:**
```
schema_version:       string, semver format "X.Y"
mode:                 string, enum ["ocr", "detection"]
input_path:           string, absolute path
output_path:          string, absolute path
timestamp:            string, ISO 8601 format
runtime_seconds:      number, float, ≥ 0.0
confidence_threshold: number, float, 0.0–1.0
total_*:              integer, ≥ 0
confidence:           number, float, 0.0–1.0  (NOT 0–100 in JSON output)
bbox.*:               integer (pixel coordinates)
full_text:            string | null
```

---

## 7. Config Schema

Every field in `config.py` has a declared type, valid range, and default.

```
Schema: Config
─────────────────────────────────────────────────────────────────────────────
Field                   Type    Default         Valid Values / Range
─────────────────────────────────────────────────────────────────────────────
PIPELINE_MODE           str     "ocr"           {"ocr", "detection"}
CONFIDENCE_THRESHOLD    float   0.80            [0.80, 1.0] — never below 0.80
GAUSSIAN_BLUR_KERNEL    tuple   (5, 5)          Odd integers ≥ 3. Symmetric.
DESKEW_ENABLED          bool    True            True | False
DESKEW_ANGLE_THRESHOLD  float   2.0             [0.0, 45.0] degrees
TESSERACT_PSM           int     3               [0, 13] — see ALGORITHM_SPEC §2.2
MODELS_DIR              str     computed        Must exist as directory
PROTOTXT_PATH           str     computed        Must exist as file
CAFFEMODEL_PATH         str     computed        Must exist as file, size > 20MB
LABELS_PATH             str     computed        Must exist as file, ≥ 1 line
OUTPUT_DIR              str     computed        Created if not exists
OUTPUT_SUFFIX           str     "_output"       Non-empty string
SAVE_OUTPUT_IMAGE       bool    True            True | False (True required for Gate 4)
SHOW_OUTPUT_WINDOW      bool    True            True | False
BLOB_SIZE               tuple   (300, 300)      Fixed — do not change
BLOB_SCALE_FACTOR       float   1/127.5         Fixed — do not change
BLOB_MEAN               tuple   (127.5,127.5,127.5) Fixed — do not change
─────────────────────────────────────────────────────────────────────────────
```

**Config validation function (runs at startup):**
```python
def validate_config() -> None:
    """Validates all config constants at startup. Raises ValueError on any violation."""
    assert PIPELINE_MODE in {"ocr", "detection"}, \
        f"PIPELINE_MODE must be 'ocr' or 'detection', got: '{PIPELINE_MODE}'"
    
    assert CONFIDENCE_THRESHOLD >= 0.80, \
        f"CONFIDENCE_THRESHOLD cannot be below 0.80. Got: {CONFIDENCE_THRESHOLD}"
    
    assert all(k % 2 == 1 and k >= 3 for k in GAUSSIAN_BLUR_KERNEL), \
        "GAUSSIAN_BLUR_KERNEL values must be odd integers ≥ 3"
    
    assert 0 <= TESSERACT_PSM <= 13, \
        f"TESSERACT_PSM must be between 0 and 13. Got: {TESSERACT_PSM}"
    
    assert SAVE_OUTPUT_IMAGE == True, \
        "SAVE_OUTPUT_IMAGE must be True (required for Milestone Gate 4)"
    
    if PIPELINE_MODE == "detection":
        assert os.path.isfile(PROTOTXT_PATH), \
            f"Model file not found: {PROTOTXT_PATH}. Run: python setup_models.py"
        assert os.path.isfile(CAFFEMODEL_PATH), \
            f"Model file not found: {CAFFEMODEL_PATH}. Run: python setup_models.py"
        assert os.path.getsize(CAFFEMODEL_PATH) > 20 * 1024 * 1024, \
            f"Caffemodel file appears corrupted (< 20MB): {CAFFEMODEL_PATH}"
```

---

## 8. Error Object Schema

All pipeline errors are raised as structured exceptions, not bare strings.

```python
class PipelineError(Exception):
    """Base exception for all Project 4 pipeline errors."""
    def __init__(self, stage: str, message: str, recoverable: bool = False):
        self.stage = stage          # Which pipeline stage failed
        self.message = message      # Human-readable description
        self.recoverable = recoverable  # Can the user fix this without code changes?
        super().__init__(f"[{stage}] {message}")

class InputValidationError(PipelineError):
    """Raised when input image fails validation."""
    pass

class ModelLoadError(PipelineError):
    """Raised when model files cannot be loaded."""
    pass

class InferenceError(PipelineError):
    """Raised when model forward pass fails."""
    pass

class OutputError(PipelineError):
    """Raised when output cannot be saved."""
    pass
```

**Error schema fields:**
```
stage:        str   — "input_validation" | "model_load" | "preprocessing" 
                      | "inference" | "postprocessing" | "output"
message:      str   — Non-empty. Must include what went wrong AND what to do.
recoverable:  bool  — True if user can fix without editing code 
                      (e.g., wrong path = True; corrupt model = False)
```

---

## 9. Validation Rules (Cross-Cutting)

These rules apply across all modules and are checked at the boundaries between modules.

### Rule V-01: Image Not Modified In Place
```python
# ENFORCE: Any function that takes an image and returns an annotated version
# must operate on a COPY, not the original.
assert np.array_equal(original, image_passed_to_function)  # After any function call
```

### Rule V-02: Confidence Always Normalized to [0, 1]
```python
# ALL internal confidence values are float in [0.0, 1.0]
# Tesseract's 0–100 integer scale is converted IMMEDIATELY on read
# and never stored in internal structures as the raw integer
assert 0.0 <= detection.confidence <= 1.0
```

### Rule V-03: BBox Always in Pixel Space
```python
# No normalized coordinates in BBox objects — always pixel integers
assert isinstance(bbox.x1, int) and isinstance(bbox.y1, int)
assert isinstance(bbox.x2, int) and isinstance(bbox.y2, int)
```

### Rule V-04: Background Class Never Returned
```python
# MobileNet-SSD class_id=0 is background. Never include in results.
assert all(d.class_id != 0 for d in detections)
```

### Rule V-05: Output Image Same Shape as Input
```python
assert annotated_image.shape == original_image.shape
assert annotated_image.dtype == np.uint8
```

### Rule V-06: Empty String Never in Text Results
```python
assert all(len(w.text.strip()) > 0 for w in word_detections)
```

### Rule V-07: total_accepted == len(detections)
```python
assert result.total_accepted == len(result.detections)
assert result.total_accepted + result.total_rejected == result.total_raw
```

### Rule V-08: threshold_used Must Match Config
```python
assert result.threshold_used == CONFIDENCE_THRESHOLD
```

### Rule V-09: output_path Must Exist After Save
```python
postprocessor.save_output(image, input_path)
assert os.path.isfile(output_path)
assert os.path.getsize(output_path) > 0
```

### Rule V-10: No Detections Below Threshold in Results
```python
assert all(d.confidence >= CONFIDENCE_THRESHOLD for d in result.detections)
```

---

*End of DATA_SCHEMA.md*
