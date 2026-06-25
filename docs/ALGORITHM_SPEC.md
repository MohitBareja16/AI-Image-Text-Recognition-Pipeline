# ALGORITHM_SPEC.md
## Project 4: Algorithm Specification
**DecodeLabs | Batch 2026**
**Scope:** Every algorithm, formula, decision rule, and mathematical operation used in the recognition pipeline — specified precisely enough that two independent implementations produce identical outputs on the same input.

---

## Table of Contents
1. [Pre-Processing Algorithms](#1-pre-processing-algorithms)
2. [Path 1 — OCR Algorithm](#2-path-1--ocr-algorithm)
3. [Path 2 — Object Detection Algorithm](#3-path-2--object-detection-algorithm)
4. [Confidence Filtering Algorithm](#4-confidence-filtering-algorithm)
5. [Non-Maximum Suppression (NMS)](#5-non-maximum-suppression-nms)
6. [Bounding Box Coordinate Scaling](#6-bounding-box-coordinate-scaling)
7. [Output Annotation Algorithm](#7-output-annotation-algorithm)
8. [Deskew Algorithm](#8-deskew-algorithm)
9. [Algorithm Decision Tree (Master Flow)](#9-algorithm-decision-tree-master-flow)
10. [Complexity Analysis](#10-complexity-analysis)

---

## 1. Pre-Processing Algorithms

Pre-processing is mandatory for both paths. The steps must execute **in the exact order specified**. Reordering produces different results and may fail Gate 2.

### 1.1 Grayscale Conversion

**Operation:** Collapse 3-channel BGR image to 1-channel luminance image.

**Formula (OpenCV's implementation of ITU-R BT.601 luma):**
```
Y = 0.114 × B + 0.587 × G + 0.299 × R
```

Where B, G, R are the Blue, Green, Red channel values (0–255) of each pixel.

**Implementation:**
```python
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

**Input:** `np.ndarray` shape `(H, W, 3)`, dtype `uint8`  
**Output:** `np.ndarray` shape `(H, W)`, dtype `uint8`  
**Side effect:** None. Original image unchanged.

**Why this formula:** Human eyes are most sensitive to green (~59%), less to red (~30%), least to blue (~11%). This weighting preserves perceived luminance contrast better than a simple average, making subsequent thresholding more accurate.

---

### 1.2 Gaussian Blur

**Operation:** Convolve the grayscale image with a Gaussian kernel to suppress high-frequency noise.

**Kernel (5×5, σ computed automatically):**
```
When σ=0, OpenCV computes: σ = 0.3 × ((ksize-1) × 0.5 - 1) + 0.8
For ksize=5: σ ≈ 1.1
```

**The 5×5 kernel (normalized):**
```
1/273 × ⎡  1   4   7   4   1 ⎤
         ⎢  4  16  26  16   4 ⎥
         ⎢  7  26  41  26   7 ⎥
         ⎢  4  16  26  16   4 ⎥
         ⎣  1   4   7   4   1 ⎦
```

**Implementation:**
```python
blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)
```

**Input:** `(H, W)` grayscale  
**Output:** `(H, W)` blurred grayscale  
**Constraint:** Kernel must be odd-sized and square. `(5, 5)` is the project standard — do not use `(3, 3)` (insufficient smoothing) or `(9, 9)` (over-blurs fine text strokes).

---

### 1.3 Adaptive Thresholding

**Operation:** Convert grayscale to binary (pure black/white) using a locally-computed threshold per pixel neighborhood. Use adaptive thresholding rather than global Otsu when the image has uneven lighting.

**Algorithm — `cv2.ADAPTIVE_THRESH_GAUSSIAN_C`:**

For each pixel `(x, y)`:
1. Define a neighborhood of size `blockSize × blockSize` centered at `(x, y)`.
2. Compute the weighted mean `μ` of the neighborhood using a Gaussian kernel (same weights as 1.2).
3. Apply threshold: `T(x, y) = μ - C`
4. Decision:
   ```
   IF pixel_value > T(x, y):  output = maxValue (255, White)
   ELSE:                       output = 0 (Black)
   ```

**Implementation:**
```python
thresh = cv2.adaptiveThreshold(
    blurred,
    maxValue=255,
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    thresholdType=cv2.THRESH_BINARY,
    blockSize=11,    # Must be odd. Neighborhood size.
    C=2              # Constant subtracted from mean. Tunes sensitivity.
)
```

**Parameters:**
- `blockSize=11`: Pixel neighborhood for local mean. Increase to 21+ for images with very large lighting gradients.
- `C=2`: Small positive value prevents noise pixels from flipping to white. Valid range: 0–10. Default 2 is robust.

**When to use Otsu's instead:**
```python
# Otsu's — only appropriate when lighting is uniform across the image
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

Otsu's algorithm finds the optimal global threshold `T*` by maximizing inter-class variance:
```
σ²_B(T) = ω₀(T) × ω₁(T) × [μ₀(T) - μ₁(T)]²

T* = argmax_T σ²_B(T)
```
Where `ω₀, ω₁` are the pixel proportions below/above T, and `μ₀, μ₁` are their means.

**Decision rule:**
```
USE adaptive thresholding IF:  image has shadows, uneven lighting, or scanned documents
USE Otsu's IF:                 image has uniform, controlled lighting (studio shots, synthetic images)
DEFAULT:                       adaptive (more robust in real-world conditions)
```

---

## 2. Path 1 — OCR Algorithm

### 2.1 Tesseract LSTM Pipeline (Internal — informational)

Tesseract's internal pipeline (not reimplemented — used as a black box):

```
Preprocessed Binary Image
        │
        ▼
  Connected Component Analysis
  (groups pixels into candidate character blobs)
        │
        ▼
  Line Finding
  (detects text baselines and groups blobs into lines)
        │
        ▼
  Convolutional Feature Extraction
  (extracts spatial features from each character region)
        │
        ▼
  Bi-directional LSTM
  (reads sequence of features left→right AND right→left)
        │
        ▼
  CTC Decoder (Connectionist Temporal Classification)
  (maps LSTM output sequence to character sequence)
        │
        ▼
  Word/Line Confidence Scoring
  (per-word confidence: 0–100 integer)
        │
        ▼
  DataFrame Output (via image_to_data)
```

### 2.2 PSM Mode Selection Algorithm

```python
def select_psm(image: np.ndarray, user_override: int = None) -> int:
    """
    Returns the appropriate PSM mode for the image.
    If user has set TESSERACT_PSM in config, that value is used directly.
    Otherwise, a heuristic auto-selection runs.
    """
    if user_override is not None:
        return user_override  # Config always wins

    h, w = image.shape[:2]
    aspect_ratio = w / h

    if aspect_ratio > 5.0:
        return 7   # Very wide, short image → single line
    elif aspect_ratio < 0.5:
        return 4   # Tall, narrow image → single column
    else:
        return 3   # Default: auto layout analysis
```

PSM Reference Table:
```
PSM 0:  Orientation and script detection only (no OCR)
PSM 1:  Automatic with OSD
PSM 2:  Automatic, no OSD (deprecated)
PSM 3:  Fully automatic (DEFAULT — use this when unsure)
PSM 4:  Single column of variable-size text
PSM 5:  Single uniform block of vertically aligned text
PSM 6:  Single uniform block of text
PSM 7:  Single text line
PSM 8:  Single word
PSM 9:  Single word in circle
PSM 10: Single character
PSM 11: Sparse text, find as much as possible
PSM 12: Sparse text with OSD
PSM 13: Raw line (bypass Tesseract heuristics)
```

### 2.3 Confidence Extraction Algorithm

```python
import pytesseract
from pytesseract import Output
import pandas as pd

def extract_high_confidence_words(image: np.ndarray, psm: int) -> pd.DataFrame:
    """
    Runs Tesseract and returns only words meeting the confidence threshold.
    
    Algorithm:
    1. Call image_to_data with custom_config specifying PSM
    2. Parse output into DataFrame
    3. Filter rows: exclude non-word rows (conf == -1) 
    4. Filter rows: exclude empty text
    5. Filter rows: retain only conf >= CONFIDENCE_THRESHOLD * 100
       (Tesseract returns 0–100 int; CONFIDENCE_THRESHOLD is 0.0–1.0 float)
    6. Return filtered DataFrame
    """
    config = f"--psm {psm}"
    data = pytesseract.image_to_data(image, config=config, output_type=Output.DATAFRAME)
    
    # Step 3: Remove layout-level rows (conf=-1 means block/para/line, not word)
    data = data[data['conf'] != -1]
    
    # Step 4: Remove rows where text is empty or whitespace only
    data = data[data['text'].str.strip() != '']
    
    # Step 5: Apply confidence threshold
    # NOTE: Tesseract conf is 0–100 integer; multiply threshold by 100
    threshold_int = CONFIDENCE_THRESHOLD * 100  # = 80
    data = data[data['conf'] >= threshold_int]
    
    # Reset index for clean iteration
    data = data.reset_index(drop=True)
    
    return data
```

**Critical Note on Confidence Scale:** Tesseract returns confidence as an integer `0–100`. The project `CONFIDENCE_THRESHOLD` is a float `0.0–1.0`. The comparison must use `conf >= 80`, not `conf >= 0.80`. This is a common off-by-100x error.

### 2.4 Text Reconstruction Algorithm

After filtering, reconstruct the extracted text preserving line structure:

```python
def reconstruct_text(df: pd.DataFrame) -> str:
    """
    Reconstructs multi-line text from filtered word DataFrame.
    Groups words by (block_num, par_num, line_num) to preserve structure.
    Words within the same line are joined with a space.
    Lines are joined with a newline character.
    """
    lines = []
    grouped = df.groupby(['block_num', 'par_num', 'line_num'])
    
    for _, group in grouped:
        line_words = group.sort_values('word_num')['text'].tolist()
        lines.append(' '.join(line_words))
    
    return '\n'.join(lines)
```

---

## 3. Path 2 — Object Detection Algorithm

### 3.1 MobileNet-SSD Architecture (Informational)

**MobileNet v1 Depthwise Separable Convolution:**

Standard convolution cost: `D_K² × M × N × D_F²`  
Depthwise separable cost: `D_K² × M × D_F² + M × N × D_F²`

Cost ratio: `1/N + 1/D_K²`

For D_K=3 (3×3 kernel): ratio ≈ `1/N + 1/9`  
Result: ~8–9× fewer computations than standard convolution for similar accuracy.

This is why MobileNet runs on CPU in real-time — it's explicitly designed for edge/mobile inference.

**SSD (Single Shot Detector):**
- Makes predictions at **multiple scales** simultaneously (from 8 feature maps of different sizes)
- No region proposal stage (unlike Faster-RCNN) — one forward pass = detections
- Default boxes (anchor boxes) at each feature map location cover different aspect ratios

### 3.2 Blob Construction Algorithm

Blob construction normalizes the image into the exact format MobileNet-SSD was trained on.

```python
def construct_blob(image: np.ndarray) -> np.ndarray:
    """
    Constructs a 4D blob from the input image.
    
    Steps:
    1. Resize image to (300, 300) — required input size for MobileNet-SSD
    2. Convert BGR to RGB (swapRB=True) — model was trained on RGB
    3. Subtract per-channel mean: (127.5, 127.5, 127.5)
       This centers pixel values around 0: range shifts from [0,255] to [-127.5, 127.5]
    4. Multiply by scalefactor (1/127.5)
       This normalizes to [-1.0, 1.0] — the range the model expects
    5. Add batch dimension: (H, W, C) → (1, C, H, W)
    
    Formula per pixel channel:
        normalized = (pixel_value - mean) × scalefactor
        normalized = (pixel_value - 127.5) × (1/127.5)
        
    Example:
        pixel = 255 → (255 - 127.5) / 127.5 =  1.0
        pixel = 127 → (127 - 127.5) / 127.5 ≈  0.0  
        pixel =   0 → (  0 - 127.5) / 127.5 = -1.0
    """
    blob = cv2.dnn.blobFromImage(
        image=image,
        scalefactor=1 / 127.5,
        size=(300, 300),
        mean=(127.5, 127.5, 127.5),
        swapRB=True,    # OpenCV loads BGR; model trained on RGB
        crop=False      # Resize without cropping (maintains aspect ratio via stretching)
    )
    # blob.shape: (1, 3, 300, 300) — batch, channels, height, width
    return blob
```

### 3.3 Forward Pass & Detection Parsing

```python
def run_forward_pass(net: cv2.dnn_Net, blob: np.ndarray) -> np.ndarray:
    """
    Runs the forward inference pass.
    
    Returns:
        detections: np.ndarray shape (1, 1, N, 7)
        
        For each detection i, detections[0, 0, i] contains:
        Index 0: image_id (always 0 for single-image inference)
        Index 1: class_id (integer, maps to label in coco_labels.txt)
        Index 2: confidence (float, 0.0–1.0)
        Index 3: x_start (normalized, 0.0–1.0)
        Index 4: y_start (normalized, 0.0–1.0)
        Index 5: x_end   (normalized, 0.0–1.0)
        Index 6: y_end   (normalized, 0.0–1.0)
    """
    net.setInput(blob)
    detections = net.forward()
    return detections
```

**Parsing loop:**
```python
def parse_detections(detections: np.ndarray, image_shape: tuple, labels: list) -> list:
    H, W = image_shape[:2]
    results = []
    
    num_detections = detections.shape[2]
    
    for i in range(num_detections):
        detection = detections[0, 0, i]
        
        confidence = float(detection[2])
        
        # Apply confidence filter (see Section 4)
        if confidence < CONFIDENCE_THRESHOLD:
            continue
        
        class_id = int(detection[1])
        label = labels[class_id] if class_id < len(labels) else f"class_{class_id}"
        
        # Scale normalized coordinates to pixel coordinates (see Section 6)
        x_start = max(0, int(detection[3] * W))
        y_start = max(0, int(detection[4] * H))
        x_end   = min(W, int(detection[5] * W))
        y_end   = min(H, int(detection[6] * H))
        
        results.append({
            "label":      label,
            "confidence": confidence,
            "class_id":   class_id,
            "bbox":       (x_start, y_start, x_end, y_end)
        })
    
    return results
```

---

## 4. Confidence Filtering Algorithm

This algorithm is the same for both paths but operates on different data types.

```
ALGORITHM: ConfidenceFilter

INPUT:  candidates  — list of detection/word objects, each with a 'confidence' field
        threshold   — float, CONFIDENCE_THRESHOLD = 0.80

PROCESS:
  accepted = []
  rejected = []

  FOR EACH candidate IN candidates:
    confidence = candidate.confidence

    IF confidence IS NOT a number:
      → log WARNING: "Malformed confidence value — skipping"
      → CONTINUE

    IF confidence < 0.0 OR confidence > 1.0:
      IF candidate is from Tesseract (0–100 int scale):
        → normalize: confidence = confidence / 100.0
      ELSE:
        → log WARNING: "Out-of-range confidence — skipping"
        → CONTINUE

    IF confidence >= threshold:
      → accepted.append(candidate)
      → log: "✓ {label}: {confidence:.1%} — retained"
    ELSE:
      → rejected.append(candidate)
      → log: "✗ {label}: {confidence:.1%} — rejected (below threshold)"

  IF len(accepted) == 0:
    → print: "No high-confidence detections found. Try a clearer image."
    → EXIT with code 0 (not an error — a valid result)

OUTPUT: accepted
```

**The critical normalization check** (index 3 of the above): Tesseract returns `int` 0–100. MobileNet returns `float` 0.0–1.0. The filter must normalize Tesseract confidence before comparison. Failing to do this means `80 >= 0.80` evaluates `True` always — every detection passes, including garbage with `conf=1`.

---

## 5. Non-Maximum Suppression (NMS)

NMS removes duplicate detections of the same object (multiple overlapping boxes).

### 5.1 IoU (Intersection over Union)

```
Given two boxes A and B:

Intersection area:
  x_inter_left   = max(A.x1, B.x1)
  y_inter_top    = max(A.y1, B.y1)
  x_inter_right  = min(A.x2, B.x2)
  y_inter_bottom = min(A.y2, B.y2)
  
  inter_w = max(0, x_inter_right - x_inter_left)
  inter_h = max(0, y_inter_bottom - y_inter_top)
  intersection = inter_w × inter_h

Union area:
  area_A = (A.x2 - A.x1) × (A.y2 - A.y1)
  area_B = (B.x2 - B.x1) × (B.y2 - B.y1)
  union  = area_A + area_B - intersection

IoU = intersection / union
```

### 5.2 NMS Algorithm

```
ALGORITHM: NMS

INPUT:  boxes       — list of (x1, y1, x2, y2) pixel coordinate boxes
        scores      — list of confidence floats (parallel to boxes)
        iou_threshold = 0.4  (project standard)

PROCESS:
  Sort boxes by scores DESCENDING (highest confidence first)
  
  keep = []
  remaining = [0, 1, 2, ..., N-1]  # indices of all boxes
  
  WHILE remaining is not empty:
    i = remaining[0]   # Take the highest-confidence box
    keep.append(i)
    remaining.remove(i)
    
    to_remove = []
    FOR EACH j IN remaining:
      iou = compute_iou(boxes[i], boxes[j])
      IF iou > iou_threshold:
        to_remove.append(j)   # Overlaps too much with kept box → discard
    
    remaining = [x for x in remaining if x not in to_remove]
  
  RETURN keep  # Indices of surviving boxes

OUTPUT: filtered list of (box, score, label) tuples
```

**Implementation:**
```python
indices = cv2.dnn.NMSBoxes(
    bboxes=boxes,              # List of [x, y, w, h] (note: x,y,w,h not x1,y1,x2,y2)
    scores=confidences,
    score_threshold=CONFIDENCE_THRESHOLD,   # 0.80
    nms_threshold=0.4          # IoU threshold for suppression
)
```

**Note on format:** `cv2.dnn.NMSBoxes` expects `[x, y, w, h]` not `[x1, y1, x2, y2]`. Convert:
```python
# Convert from (x1, y1, x2, y2) to (x, y, w, h)
nms_boxes = [[x1, y1, x2-x1, y2-y1] for (x1, y1, x2, y2) in pixel_boxes]
```

---

## 6. Bounding Box Coordinate Scaling

### 6.1 Normalization Context

MobileNet-SSD outputs coordinates normalized to `[0.0, 1.0]` relative to the **300×300 resized blob**, not the original image. Scaling must use the **original image dimensions**.

```
ALGORITHM: ScaleCoordinates

INPUT:
  normalized_coords = (x_n_start, y_n_start, x_n_end, y_n_end)  ← floats in [0, 1]
  image_shape = (H, W, C)  ← original image, not the 300×300 blob

FORMULAS:
  x_start = int(x_n_start × W)
  y_start = int(y_n_start × H)
  x_end   = int(x_n_end   × W)
  y_end   = int(y_n_end   × H)

BOUNDARY CLAMPING (mandatory — prevents drawing outside image):
  x_start = max(0, x_start)
  y_start = max(0, y_start)
  x_end   = min(W, x_end)
  y_end   = min(H, y_end)

VALIDITY CHECK:
  IF x_end <= x_start OR y_end <= y_start:
    → This box has zero or negative area. Skip — do not draw.

OUTPUT: (x_start, y_start, x_end, y_end) in pixel coordinates
```

### 6.2 Label Placement Algorithm

Placing the label text above the bounding box without going off-screen:

```python
def compute_label_position(x_start: int, y_start: int, label: str) -> tuple:
    """
    Computes where to place the label text.
    Moves label inside box if it would render above the image boundary.
    """
    text_size, baseline = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
    text_w, text_h = text_size
    
    # Preferred position: above the top-left corner of the box
    label_y = y_start - 10
    
    # If label would go above image top, place it inside the box instead
    if label_y - text_h < 0:
        label_y = y_start + text_h + 5
    
    label_x = x_start
    
    # Label background rectangle
    bg_x1 = label_x
    bg_y1 = label_y - text_h - LABEL_PADDING
    bg_x2 = label_x + text_w
    bg_y2 = label_y + LABEL_PADDING
    
    return (label_x, label_y), (bg_x1, bg_y1, bg_x2, bg_y2)
```

---

## 7. Output Annotation Algorithm

```
ALGORITHM: AnnotateImage

INPUT:
  image         — original BGR image (unmodified)
  detections    — list of {label, confidence, bbox} dicts
  path_mode     — "ocr" | "detection"

PROCESS:
  annotated = image.copy()   ← ALWAYS copy; never modify the original in place

  FOR EACH detection IN detections:
    bbox = detection["bbox"]
    confidence = detection["confidence"]
    label = detection["label"]
    
    1. Determine box color:
       IF confidence >= 0.90: color = COLOR_HIGH_CONFIDENCE
       ELSE:                  color = COLOR_MED_CONFIDENCE

    2. Draw bounding box rectangle:
       cv2.rectangle(annotated, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, thickness=2)

    3. Format label string:
       label_str = f"{label}: {confidence*100:.1f}%"
       e.g., "Person: 91.3%"

    4. Compute label position (see Section 6.2)

    5. Draw label background rectangle (filled, same color, alpha blend or solid)

    6. Draw label text:
       cv2.putText(annotated, label_str, text_pos, FONT, FONT_SCALE, COLOR_TEXT_ON_BOX, 
                   FONT_THICKNESS, cv2.LINE_AA)

  RETURN annotated

OUTPUT: annotated image (np.ndarray, same shape as input)
```

**Alpha blending for label background (optional but recommended):**
```python
overlay = annotated.copy()
cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), color, cv2.FILLED)
cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
```

---

## 8. Deskew Algorithm

Only runs if `DESKEW_ENABLED = True` and detected angle > `DESKEW_ANGLE_THRESHOLD`.

```
ALGORITHM: Deskew

INPUT: grayscale_image (after blur, before threshold)

STEP 1 — Edge detection:
  edges = cv2.Canny(gray, threshold1=50, threshold2=150)

STEP 2 — Find contours:
  contours = cv2.findContours(edges, RETR_LIST, CHAIN_APPROX_SIMPLE)
  all_points = np.concatenate([c.reshape(-1, 2) for c in contours])

STEP 3 — Fit minimum area rectangle:
  rect = cv2.minAreaRect(all_points)
  angle = rect[2]   ← rotation angle in degrees (-90 to 0)

STEP 4 — Normalize angle:
  IF angle < -45:
    angle = 90 + angle   ← maps (-90,-45) to (0, 45)

STEP 5 — Check threshold:
  IF abs(angle) <= DESKEW_ANGLE_THRESHOLD (2.0°):
    → Skip rotation (not worth the resampling artifacts)
    → Return original image unchanged

STEP 6 — Compute rotation matrix:
  (h, w) = image.shape[:2]
  center = (w // 2, h // 2)
  M = cv2.getRotationMatrix2D(center, angle, scale=1.0)

STEP 7 — Apply rotation:
  deskewed = cv2.warpAffine(
      image, M, (w, h),
      flags=cv2.INTER_CUBIC,
      borderMode=cv2.BORDER_REPLICATE  ← fills corners with edge pixels, not black
  )

OUTPUT: deskewed image (same shape as input)
```

**Why BORDER_REPLICATE:** Using `BORDER_CONSTANT` (default) fills rotation-introduced corners with black pixels (0). Thresholding then treats these as text content, creating spurious detections at image edges. `BORDER_REPLICATE` avoids this.

---

## 9. Algorithm Decision Tree (Master Flow)

```
START
  │
  ▼
validate_image_path(path)
  │
  ├── INVALID → print error, exit(1)
  │
  └── VALID
       │
       ▼
    image = cv2.imread(path)
       │
       ├── None → print error, exit(1)
       │
       └── OK
            │
            ▼
         preprocess(image)
         ├── cvtColor (BGR→GRAY)
         ├── GaussianBlur (5×5)
         ├── [deskew if enabled]
         └── adaptiveThreshold
            │
            ▼
         BRANCH on PIPELINE_MODE
            │
     ┌──────┴──────┐
     │             │
    OCR         DETECTION
     │             │
     ▼             ▼
 image_to_data  blobFromImage
 filter(≥80)    forward()
 reconstruct    parse_detections
 text           filter(≥80)
     │          apply NMS
     │             │
     └──────┬──────┘
            │
            ▼
      annotate_image (copy!)
            │
            ▼
      print_summary
            │
            ▼
      save_output → /output/<name>_output.jpg
            │
            ▼
      [optional] imshow
            │
            ▼
         exit(0)
```

---

## 10. Complexity Analysis

| Algorithm | Time Complexity | Space Complexity | Notes |
|-----------|----------------|-----------------|-------|
| Grayscale conversion | O(H×W) | O(H×W) | Single pass per pixel |
| Gaussian Blur (5×5) | O(H×W×25) ≈ O(H×W) | O(H×W) | Separable kernel; O(H×W×10) in practice |
| Adaptive Threshold | O(H×W×blockSize²) | O(H×W) | blockSize=11 → O(H×W×121) |
| Deskew (warpAffine) | O(H×W) | O(H×W) | Bicubic interpolation at each pixel |
| OCR (Tesseract) | O(H×W) amortized | O(H×W) | LSTM internal — treated as black box |
| Blob construction | O(H×W×C) | O(300×300×3) | Resize + normalize |
| MobileNet-SSD forward | O(1) per image | O(model_size) | Fixed architecture; ~23M params |
| NMS | O(N² ) worst case | O(N) | N = raw detections (typically < 100) |
| Annotation | O(D) | O(H×W×C) | D = number of final detections |

**Total wall-clock target:** ≤ 10 seconds on a 512×512 image on a standard 4-core CPU laptop (NFR-01). Tesseract is the bottleneck for OCR; MobileNet forward pass is the bottleneck for detection (~300ms on CPU).

---

*End of ALGORITHM_SPEC.md*
