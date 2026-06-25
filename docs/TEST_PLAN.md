# TEST_PLAN.md
## Project 4: Test Plan
**DecodeLabs | Batch 2026**
**Philosophy:** Every loophole identified in `PRD.md §12` and every invariant declared in `DATA_SCHEMA.md §9` has at least one explicit test case. Tests are organized by layer (unit → integration → gate validation → adversarial). A submission that passes all Gate tests but fails any adversarial test is **not complete**.

---

## Table of Contents
1. [Testing Layers](#1-testing-layers)
2. [Test Environment](#2-test-environment)
3. [Unit Tests — Input Validation](#3-unit-tests--input-validation)
4. [Unit Tests — Pre-Processing](#4-unit-tests--pre-processing)
5. [Unit Tests — Confidence Filtering](#5-unit-tests--confidence-filtering)
6. [Unit Tests — Coordinate Scaling](#6-unit-tests--coordinate-scaling)
7. [Unit Tests — Bounding Box](#7-unit-tests--bounding-box)
8. [Unit Tests — Text Reconstruction](#8-unit-tests--text-reconstruction)
9. [Unit Tests — Output & Save](#9-unit-tests--output--save)
10. [Unit Tests — Config Validation](#10-unit-tests--config-validation)
11. [Integration Tests — OCR Pipeline](#11-integration-tests--ocr-pipeline)
12. [Integration Tests — Detection Pipeline](#12-integration-tests--detection-pipeline)
13. [Milestone Gate Tests (Evaluator Suite)](#13-milestone-gate-tests-evaluator-suite)
14. [Adversarial Tests (Loophole Coverage)](#14-adversarial-tests-loophole-coverage)
15. [Regression Test Checklist](#15-regression-test-checklist)
16. [Test Image Specifications](#16-test-image-specifications)
17. [Test Execution Guide](#17-test-execution-guide)

---

## 1. Testing Layers

```
Layer 4: Adversarial / Anti-Cheat
  ↑ Tests that specifically probe known loopholes and gaming strategies
Layer 3: Milestone Gate Validation  
  ↑ The four official gates evaluated by DecodeLabs mentors
Layer 2: Integration Tests
  ↑ Full pipeline runs on real images; end-to-end output verification
Layer 1: Unit Tests
  ↑ Individual functions with mocked or synthetic inputs
```

Each layer depends on the one below it passing first. Do not skip to Gate tests without passing unit tests.

---

## 2. Test Environment

### 2.1 Setup

```bash
# Install test dependencies (add to requirements.txt for dev)
pip install pytest==7.4.3
pip install pytest-cov==4.1.0
pip install numpy==1.24.4   # already in requirements

# Run all tests
pytest tests/ -v --tb=short

# Run with coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Run only gate tests (evaluator mode)
pytest tests/test_gates.py -v
```

### 2.2 Directory Structure

```
tests/
├── conftest.py                  # Shared fixtures (sample images, mock data)
├── test_input_validation.py     # §3
├── test_preprocessing.py        # §4
├── test_confidence_filter.py    # §5
├── test_coordinate_scaling.py   # §6
├── test_bounding_box.py         # §7
├── test_text_reconstruction.py  # §8
├── test_output_save.py          # §9
├── test_config_validation.py    # §10
├── test_ocr_integration.py      # §11
├── test_detection_integration.py# §12
├── test_gates.py                # §13 — official milestone gates
├── test_adversarial.py          # §14 — loophole / anti-cheat tests
└── test_images/
    ├── ocr_clean.png            # High-quality printed text
    ├── ocr_noisy.png            # Text with noise and shadow
    ├── ocr_skewed.png           # Text rotated ~5 degrees
    ├── ocr_blank.png            # White image with no text
    ├── detection_people.jpg     # Image with clearly visible people
    ├── detection_vehicles.jpg   # Image with cars/trucks
    ├── detection_empty.jpg      # Outdoor scene with no COCO objects
    ├── detection_crowded.jpg    # Many overlapping objects (NMS test)
    ├── tiny.png                 # 10×10 pixels (minimum size rejection test)
    ├── corrupt.jpg              # Corrupted file (invalid JPEG header)
    ├── zero_bytes.jpg           # Empty file
    └── large.png                # 4000×4000 pixels (performance test)
```

### 2.3 Fixtures (conftest.py)

```python
import pytest
import numpy as np
import cv2
import os

@pytest.fixture
def black_image():
    """Pure black 100×100 RGB image."""
    return np.zeros((100, 100, 3), dtype=np.uint8)

@pytest.fixture
def white_image():
    """Pure white 100×100 RGB image."""
    return np.full((100, 100, 3), 255, dtype=np.uint8)

@pytest.fixture
def synthetic_text_image():
    """White image with 'HELLO' printed in black — guaranteed OCR target."""
    img = np.full((100, 400, 3), 255, dtype=np.uint8)
    cv2.putText(img, 'HELLO', (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)
    return img

@pytest.fixture
def sample_tesseract_dataframe():
    """Realistic mock DataFrame as returned by pytesseract.image_to_data."""
    import pandas as pd
    return pd.DataFrame({
        'level':    [1,    2,    3,    4,    5,    5,    5   ],
        'page_num': [1,    1,    1,    1,    1,    1,    1   ],
        'block_num':[0,    1,    1,    1,    1,    1,    1   ],
        'par_num':  [0,    0,    1,    1,    1,    1,    1   ],
        'line_num': [0,    0,    0,    1,    1,    1,    1   ],
        'word_num': [0,    0,    0,    0,    1,    2,    3   ],
        'left':     [0,    10,   10,   10,   10,   100,  200 ],
        'top':      [0,    10,   10,   10,   10,   10,   10  ],
        'width':    [400,  380,  380,  380,  80,   90,   70  ],
        'height':   [100,  80,   80,   80,   30,   30,   30  ],
        'conf':     [-1,   -1,   -1,   -1,   95.0, 45.0, 82.0],
        'text':     ['',   '',   '',   '',   'Hello', 'world', 'test'],
    })

@pytest.fixture
def mock_detections_raw():
    """Simulated raw MobileNet-SSD output array."""
    # shape: (1, 1, 3, 7) — 3 candidate detections
    detections = np.zeros((1, 1, 3, 7), dtype=np.float32)
    # Detection 0: Person, confidence 0.91 — should be KEPT
    detections[0, 0, 0] = [0, 15, 0.91, 0.1, 0.2, 0.4, 0.8]
    # Detection 1: Car, confidence 0.67 — should be REJECTED
    detections[0, 0, 1] = [0, 7,  0.67, 0.5, 0.1, 0.9, 0.6]
    # Detection 2: Background class 0, confidence 0.95 — should be REJECTED (background)
    detections[0, 0, 2] = [0, 0,  0.95, 0.0, 0.0, 1.0, 1.0]
    return detections
```

---

## 3. Unit Tests — Input Validation

**File:** `tests/test_input_validation.py`

### TC-INV-001: Valid image path accepted
```python
def test_valid_jpg_path(tmp_path):
    """A valid .jpg file path returns True."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    p = tmp_path / "test.jpg"
    cv2.imwrite(str(p), img)
    assert validate_image_path(str(p)) == True
```

### TC-INV-002: Valid PNG path accepted
```python
def test_valid_png_path(tmp_path):
    p = tmp_path / "test.png"
    cv2.imwrite(str(p), np.zeros((100, 100, 3), dtype=np.uint8))
    assert validate_image_path(str(p)) == True
```

### TC-INV-003: Empty string rejected
```python
def test_empty_path_rejected():
    assert validate_image_path("") == False
```

### TC-INV-004: PDF extension rejected
```python
def test_pdf_extension_rejected(tmp_path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"fake pdf content")
    assert validate_image_path(str(p)) == False
```

### TC-INV-005: Non-existent file rejected
```python
def test_nonexistent_file_rejected():
    assert validate_image_path("/definitely/does/not/exist.jpg") == False
```

### TC-INV-006: Zero-byte file rejected
```python
def test_zero_byte_file_rejected(tmp_path):
    p = tmp_path / "empty.jpg"
    p.write_bytes(b"")
    assert validate_image_path(str(p)) == False
```

### TC-INV-007: File over 50MB rejected
```python
def test_oversized_file_rejected(tmp_path):
    p = tmp_path / "huge.jpg"
    p.write_bytes(b"x" * (51 * 1024 * 1024))  # 51 MB
    assert validate_image_path(str(p)) == False
```

### TC-INV-008: .txt extension rejected
```python
def test_txt_extension_rejected(tmp_path):
    p = tmp_path / "data.txt"
    p.write_text("hello")
    assert validate_image_path(str(p)) == False
```

### TC-INV-009: Case-insensitive extension accepted (.JPG)
```python
def test_uppercase_extension_accepted(tmp_path):
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    p = tmp_path / "TEST.JPG"
    cv2.imwrite(str(p), img)
    assert validate_image_path(str(p)) == True
```

### TC-INV-010: cv2.imread returns None → raises FileNotFoundError
```python
def test_load_image_raises_on_none(tmp_path):
    """A file that exists but can't be decoded by OpenCV must raise FileNotFoundError."""
    p = tmp_path / "corrupt.jpg"
    p.write_bytes(b"this is not a real jpeg")
    with pytest.raises(FileNotFoundError):
        load_image(str(p))
```

---

## 4. Unit Tests — Pre-Processing

**File:** `tests/test_preprocessing.py`

### TC-PRE-001: Output grayscale shape is (H, W) not (H, W, 1)
```python
def test_grayscale_shape(black_image):
    bundle = preprocess(black_image)
    assert bundle.gray.ndim == 2
    assert bundle.gray.shape == black_image.shape[:2]
```

### TC-PRE-002: Grayscale dtype is uint8
```python
def test_grayscale_dtype(white_image):
    bundle = preprocess(white_image)
    assert bundle.gray.dtype == np.uint8
```

### TC-PRE-003: Binary image contains only 0 and 255
```python
def test_binary_only_zero_and_255(synthetic_text_image):
    bundle = preprocess(synthetic_text_image)
    unique_vals = set(np.unique(bundle.binary))
    assert unique_vals.issubset({0, 255}), \
        f"Binary image contains unexpected values: {unique_vals - {0, 255}}"
```

### TC-PRE-004: Original image is not modified
```python
def test_original_image_unmodified(synthetic_text_image):
    original_copy = synthetic_text_image.copy()
    preprocess(synthetic_text_image)
    np.testing.assert_array_equal(synthetic_text_image, original_copy)
```

### TC-PRE-005: Output shapes match input shape
```python
def test_all_shapes_match(synthetic_text_image):
    H, W = synthetic_text_image.shape[:2]
    bundle = preprocess(synthetic_text_image)
    assert bundle.gray.shape == (H, W)
    assert bundle.blurred.shape == (H, W)
    assert bundle.binary.shape == (H, W)
```

### TC-PRE-006: None input raises ValueError
```python
def test_none_input_raises():
    with pytest.raises(ValueError, match="image"):
        preprocess(None)
```

### TC-PRE-007: Wrong dtype input raises ValueError
```python
def test_float_image_raises():
    float_image = np.zeros((100, 100, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        preprocess(float_image)
```

### TC-PRE-008: Too-small image raises ValueError
```python
def test_tiny_image_raises():
    tiny = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="too small"):
        preprocess(tiny)
```

### TC-PRE-009: Gaussian blur produces different values than input (not a no-op)
```python
def test_blur_is_not_noop(synthetic_text_image):
    bundle = preprocess(synthetic_text_image)
    # Blurred image must differ from grayscale (except all-uniform images)
    # We use synthetic_text_image which has edges — blur will change them
    assert not np.array_equal(bundle.gray, bundle.blurred)
```

### TC-PRE-010: Deskew disabled — angle is 0.0
```python
def test_deskew_disabled_angle_zero(monkeypatch, synthetic_text_image):
    monkeypatch.setattr("src.config.DESKEW_ENABLED", False)
    bundle = preprocess(synthetic_text_image)
    assert bundle.deskewed == False
    assert bundle.angle == 0.0
```

---

## 5. Unit Tests — Confidence Filtering

**File:** `tests/test_confidence_filter.py`

### TC-CONF-001: Score of exactly 0.80 is ACCEPTED (boundary — inclusive)
```python
def test_boundary_exactly_80_accepted():
    result = confidence_filter([{"label": "test", "confidence": 0.80}])
    assert len(result) == 1
```

### TC-CONF-002: Score of 0.7999 is REJECTED (boundary — exclusive)
```python
def test_boundary_below_80_rejected():
    result = confidence_filter([{"label": "test", "confidence": 0.7999}])
    assert len(result) == 0
```

### TC-CONF-003: Score of 1.0 is ACCEPTED
```python
def test_perfect_score_accepted():
    result = confidence_filter([{"label": "test", "confidence": 1.0}])
    assert len(result) == 1
```

### TC-CONF-004: Score of 0.0 is REJECTED
```python
def test_zero_score_rejected():
    result = confidence_filter([{"label": "test", "confidence": 0.0}])
    assert len(result) == 0
```

### TC-CONF-005: Tesseract integer 80 is treated as 0.80 (ACCEPTED)
```python
def test_tesseract_integer_80_is_accepted():
    """conf=80 in Tesseract's 0-100 scale must normalize to 0.80 and be accepted."""
    result = filter_tesseract_words(conf_value=80)
    assert result == True
```

### TC-CONF-006: Tesseract integer 79 is REJECTED
```python
def test_tesseract_integer_79_is_rejected():
    result = filter_tesseract_words(conf_value=79)
    assert result == False
```

### TC-CONF-007: Tesseract integer -1 (layout row) is ALWAYS REJECTED
```python
def test_tesseract_neg1_always_rejected():
    result = filter_tesseract_words(conf_value=-1)
    assert result == False
```

### TC-CONF-008: Mixed list — only ≥80% pass through
```python
def test_mixed_list_correct_count():
    candidates = [
        {"label": "a", "confidence": 0.91},
        {"label": "b", "confidence": 0.79},
        {"label": "c", "confidence": 0.80},
        {"label": "d", "confidence": 0.50},
        {"label": "e", "confidence": 1.00},
    ]
    result = confidence_filter(candidates)
    assert len(result) == 3
    assert all(r["confidence"] >= 0.80 for r in result)
```

### TC-CONF-009: Empty input list → empty output (no crash)
```python
def test_empty_input_no_crash():
    result = confidence_filter([])
    assert result == []
```

### TC-CONF-010: All-rejected list → zero-results message triggered
```python
def test_all_rejected_triggers_warning(capsys):
    result = confidence_filter([{"label": "x", "confidence": 0.50}])
    assert len(result) == 0
    captured = capsys.readouterr()
    # The zero-results message must appear in stdout
    assert "No high-confidence" in captured.out or len(result) == 0
```

### TC-CONF-011: Background class_id=0 REJECTED even at 0.95 confidence
```python
def test_background_class_always_rejected():
    """MobileNet class_id=0 is background — never a valid detection."""
    detection = ObjectDetection(label="background", class_id=0, confidence=0.95, bbox=...)
    result = filter_object_detections([detection])
    assert len(result) == 0
```

---

## 6. Unit Tests — Coordinate Scaling

**File:** `tests/test_coordinate_scaling.py`

### TC-COORD-001: Normalized (0.0, 0.0, 1.0, 1.0) → full image bounds
```python
def test_full_frame_detection():
    bbox = scale_coordinates((0.0, 0.0, 1.0, 1.0), image_width=640, image_height=480)
    assert bbox == (0, 0, 640, 480)
```

### TC-COORD-002: Normalized (0.5, 0.5, 0.75, 0.75) → correct pixel coords
```python
def test_center_detection():
    bbox = scale_coordinates((0.5, 0.5, 0.75, 0.75), image_width=400, image_height=400)
    assert bbox == (200, 200, 300, 300)
```

### TC-COORD-003: Negative normalized coord clamped to 0
```python
def test_negative_coord_clamped():
    bbox = scale_coordinates((-0.1, 0.0, 0.5, 0.5), image_width=100, image_height=100)
    assert bbox[0] == 0   # x1 clamped to 0
```

### TC-COORD-004: Normalized coord > 1.0 clamped to image boundary
```python
def test_overflow_coord_clamped():
    bbox = scale_coordinates((0.0, 0.0, 1.5, 1.2), image_width=100, image_height=100)
    assert bbox[2] == 100  # x2 clamped to width
    assert bbox[3] == 100  # y2 clamped to height
```

### TC-COORD-005: All output values are int (not float)
```python
def test_output_values_are_int():
    bbox = scale_coordinates((0.1, 0.2, 0.8, 0.9), image_width=300, image_height=200)
    assert all(isinstance(v, int) for v in bbox)
```

### TC-COORD-006: Inverted box (x_start > x_end after scaling) → flagged as invalid
```python
def test_inverted_box_detected():
    """Raw model sometimes outputs x_start > x_end — must be skipped."""
    is_valid = validate_bbox_coords(x1=200, y1=50, x2=100, y2=150)
    assert is_valid == False
```

### TC-COORD-007: Zero-area box (x1==x2 or y1==y2) → flagged as invalid
```python
def test_zero_area_box_invalid():
    assert validate_bbox_coords(x1=100, y1=100, x2=100, y2=200) == False  # zero width
    assert validate_bbox_coords(x1=100, y1=100, x2=200, y2=100) == False  # zero height
```

### TC-COORD-008: Scaling uses original image dimensions, not blob (300×300)
```python
def test_uses_original_not_blob_dimensions():
    """Common bug: developer uses 300 instead of actual image W/H."""
    original_W, original_H = 1920, 1080
    bbox = scale_coordinates((0.5, 0.5, 0.75, 0.75), original_W, original_H)
    assert bbox[2] == int(0.75 * 1920)  # 1440, not int(0.75 * 300) = 225
    assert bbox[3] == int(0.75 * 1080)  # 810, not int(0.75 * 300) = 225
```

---

## 7. Unit Tests — Bounding Box

**File:** `tests/test_bounding_box.py`

### TC-BBOX-001: width property computed correctly
```python
def test_bbox_width():
    bbox = BBox(x1=10, y1=20, x2=110, y2=80)
    assert bbox.width == 100
```

### TC-BBOX-002: height property computed correctly
```python
def test_bbox_height():
    bbox = BBox(x1=10, y1=20, x2=110, y2=80)
    assert bbox.height == 60
```

### TC-BBOX-003: area computed correctly
```python
def test_bbox_area():
    bbox = BBox(x1=0, y1=0, x2=10, y2=10)
    assert bbox.area == 100
```

### TC-BBOX-004: to_xywh converts correctly for cv2.NMSBoxes
```python
def test_bbox_to_xywh():
    bbox = BBox(x1=50, y1=30, x2=150, y2=130)
    assert bbox.to_xywh() == (50, 30, 100, 100)
```

### TC-BBOX-005: Invalid BBox (x1 > x2) raises ValueError on construction
```python
def test_invalid_bbox_raises():
    with pytest.raises(ValueError):
        BBox(x1=200, y1=100, x2=100, y2=200)  # x1 > x2
```

### TC-BBOX-006: All coordinate values must be int
```python
def test_bbox_requires_int():
    with pytest.raises((TypeError, ValueError)):
        BBox(x1=1.5, y1=2.0, x2=3.5, y2=4.0)
```

---

## 8. Unit Tests — Text Reconstruction

**File:** `tests/test_text_reconstruction.py`

### TC-TEXT-001: Single-line reconstruction preserves word order
```python
def test_single_line_word_order(sample_tesseract_dataframe):
    df = filter_dataframe(sample_tesseract_dataframe)
    text = reconstruct_text(df)
    # "Hello" (conf=95) and "test" (conf=82) pass; "world" (conf=45) fails
    assert "Hello" in text
    assert "test" in text
    assert "world" not in text
```

### TC-TEXT-002: Low-confidence word NOT in reconstructed text
```python
def test_low_confidence_word_excluded(sample_tesseract_dataframe):
    df = filter_dataframe(sample_tesseract_dataframe)
    text = reconstruct_text(df)
    assert "world" not in text  # conf=45, below threshold
```

### TC-TEXT-003: Empty DataFrame returns empty string (no crash)
```python
def test_empty_dataframe_returns_empty_string():
    import pandas as pd
    empty_df = pd.DataFrame(columns=['block_num','par_num','line_num','word_num','text','conf'])
    text = reconstruct_text(empty_df)
    assert text == ""
```

### TC-TEXT-004: Multi-line text has newline separators between lines
```python
def test_multiline_has_newlines():
    # Construct a DataFrame with two distinct line groups
    ...  # see full test in test file
    text = reconstruct_text(df)
    assert "\n" in text
```

### TC-TEXT-005: Whitespace-only tokens excluded from output
```python
def test_whitespace_tokens_excluded():
    # conf=95, text="   " (only spaces) must not appear in output
    ...
    assert text.strip() != ""  # Not all whitespace
```

---

## 9. Unit Tests — Output & Save

**File:** `tests/test_output_save.py`

### TC-OUT-001: Output file created on disk
```python
def test_output_file_created(tmp_path, synthetic_text_image, monkeypatch):
    monkeypatch.setattr("src.config.OUTPUT_DIR", str(tmp_path))
    saved_path = save_output(synthetic_text_image, str(tmp_path / "test.jpg"))
    assert os.path.isfile(saved_path)
```

### TC-OUT-002: Output file is not zero bytes
```python
def test_output_file_not_empty(tmp_path, synthetic_text_image, monkeypatch):
    monkeypatch.setattr("src.config.OUTPUT_DIR", str(tmp_path))
    saved_path = save_output(synthetic_text_image, str(tmp_path / "test.jpg"))
    assert os.path.getsize(saved_path) > 0
```

### TC-OUT-003: Output filename has correct suffix
```python
def test_output_filename_suffix(tmp_path, synthetic_text_image, monkeypatch):
    monkeypatch.setattr("src.config.OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr("src.config.OUTPUT_SUFFIX", "_output")
    saved_path = save_output(synthetic_text_image, "/some/path/invoice.jpg")
    assert os.path.basename(saved_path) == "invoice_output.jpg"
```

### TC-OUT-004: Output directory auto-created if not exists
```python
def test_output_dir_created(tmp_path, synthetic_text_image, monkeypatch):
    new_dir = tmp_path / "new_output_dir"
    assert not new_dir.exists()
    monkeypatch.setattr("src.config.OUTPUT_DIR", str(new_dir))
    save_output(synthetic_text_image, str(tmp_path / "test.jpg"))
    assert new_dir.exists()
```

### TC-OUT-005: Annotated image same shape as original
```python
def test_annotated_image_same_shape(synthetic_text_image):
    detections = []  # Empty — draw nothing
    annotated = draw_ocr_boxes(synthetic_text_image, detections)
    assert annotated.shape == synthetic_text_image.shape
```

### TC-OUT-006: Annotation operates on copy — original unchanged
```python
def test_annotation_does_not_modify_original(synthetic_text_image):
    original_copy = synthetic_text_image.copy()
    _ = draw_ocr_boxes(synthetic_text_image, [])
    np.testing.assert_array_equal(synthetic_text_image, original_copy)
```

### TC-OUT-007: Console output contains confidence values
```python
def test_console_output_has_confidence(capsys):
    result = PipelineResult(
        mode="detection",
        detections=[ObjectDetection(label="Person", confidence=0.913, ...)],
        ...
    )
    print_summary(result)
    captured = capsys.readouterr()
    assert "91.3%" in captured.out or "0.913" in captured.out
```

### TC-OUT-008: Console output contains "No high-confidence" when empty
```python
def test_empty_result_message(capsys):
    result = PipelineResult(mode="detection", detections=[], ...)
    print_summary(result)
    captured = capsys.readouterr()
    assert "No high-confidence" in captured.out
```

---

## 10. Unit Tests — Config Validation

**File:** `tests/test_config_validation.py`

### TC-CFG-001: Valid default config passes validation
```python
def test_valid_config_passes():
    validate_config()  # Must not raise
```

### TC-CFG-002: PIPELINE_MODE invalid string fails
```python
def test_invalid_pipeline_mode(monkeypatch):
    monkeypatch.setattr("src.config.PIPELINE_MODE", "video")
    with pytest.raises((AssertionError, ValueError)):
        validate_config()
```

### TC-CFG-003: CONFIDENCE_THRESHOLD below 0.80 fails
```python
def test_threshold_below_80_fails(monkeypatch):
    monkeypatch.setattr("src.config.CONFIDENCE_THRESHOLD", 0.79)
    with pytest.raises((AssertionError, ValueError)):
        validate_config()
```

### TC-CFG-004: CONFIDENCE_THRESHOLD exactly 0.80 passes
```python
def test_threshold_exactly_80_passes(monkeypatch):
    monkeypatch.setattr("src.config.CONFIDENCE_THRESHOLD", 0.80)
    validate_config()  # Must not raise
```

### TC-CFG-005: SAVE_OUTPUT_IMAGE = False fails
```python
def test_save_output_false_fails(monkeypatch):
    monkeypatch.setattr("src.config.SAVE_OUTPUT_IMAGE", False)
    with pytest.raises((AssertionError, ValueError)):
        validate_config()
```

### TC-CFG-006: Even GAUSSIAN_BLUR_KERNEL fails
```python
def test_even_kernel_fails(monkeypatch):
    monkeypatch.setattr("src.config.GAUSSIAN_BLUR_KERNEL", (4, 4))
    with pytest.raises((AssertionError, ValueError)):
        validate_config()
```

### TC-CFG-007: TESSERACT_PSM out of range fails
```python
def test_psm_out_of_range_fails(monkeypatch):
    monkeypatch.setattr("src.config.TESSERACT_PSM", 14)
    with pytest.raises((AssertionError, ValueError)):
        validate_config()
```

---

## 11. Integration Tests — OCR Pipeline

**File:** `tests/test_ocr_integration.py`  
*Requires: Tesseract binary installed. Skip with `pytest -m "not integration"` if not available.*

### TC-OCR-INT-001: Clean text image → at least one high-confidence word
```python
@pytest.mark.integration
def test_clean_image_produces_results():
    result = run_full_pipeline("tests/test_images/ocr_clean.png", mode="ocr")
    assert result.total_accepted >= 1
    assert result.full_text is not None
    assert len(result.full_text.strip()) > 0
```

### TC-OCR-INT-002: All returned words have confidence ≥ 0.80
```python
@pytest.mark.integration
def test_all_returned_words_above_threshold():
    result = run_full_pipeline("tests/test_images/ocr_clean.png", mode="ocr")
    for word in result.detections:
        assert word.confidence >= 0.80, \
            f"Word '{word.text}' has confidence {word.confidence:.3f} below threshold"
```

### TC-OCR-INT-003: Blank image → zero accepted results + message
```python
@pytest.mark.integration
def test_blank_image_zero_results(capsys):
    result = run_full_pipeline("tests/test_images/ocr_blank.png", mode="ocr")
    assert result.total_accepted == 0
    captured = capsys.readouterr()
    assert "No high-confidence" in captured.out
```

### TC-OCR-INT-004: Output image saved to disk
```python
@pytest.mark.integration
def test_output_image_saved(tmp_path, monkeypatch):
    monkeypatch.setattr("src.config.OUTPUT_DIR", str(tmp_path))
    result = run_full_pipeline("tests/test_images/ocr_clean.png", mode="ocr")
    assert os.path.isfile(result.output_path)
    assert os.path.getsize(result.output_path) > 0
```

### TC-OCR-INT-005: PipelineResult invariants hold
```python
@pytest.mark.integration
def test_pipeline_result_invariants():
    result = run_full_pipeline("tests/test_images/ocr_clean.png", mode="ocr")
    assert result.total_accepted == len(result.detections)
    assert result.total_accepted + result.total_rejected == result.total_raw
    assert result.threshold_used == 0.80
    assert result.mode == "ocr"
    assert result.full_text is not None
```

### TC-OCR-INT-006: Runtime under 10 seconds
```python
@pytest.mark.integration
def test_runtime_under_10s():
    result = run_full_pipeline("tests/test_images/ocr_clean.png", mode="ocr")
    assert result.runtime_seconds < 10.0, \
        f"Pipeline too slow: {result.runtime_seconds:.2f}s (limit: 10s)"
```

---

## 12. Integration Tests — Detection Pipeline

**File:** `tests/test_detection_integration.py`  
*Requires: MobileNet-SSD model files. Skip with `pytest -m "not integration"` if not downloaded.*

### TC-DET-INT-001: People image → at least one high-confidence detection
```python
@pytest.mark.integration
def test_people_image_detects_person():
    result = run_full_pipeline("tests/test_images/detection_people.jpg", mode="detection")
    labels = [d.label for d in result.detections]
    assert "person" in labels or "people" in [l.lower() for l in labels]
```

### TC-DET-INT-002: All detections have confidence ≥ 0.80
```python
@pytest.mark.integration
def test_all_detections_above_threshold():
    result = run_full_pipeline("tests/test_images/detection_people.jpg", mode="detection")
    for det in result.detections:
        assert det.confidence >= 0.80
```

### TC-DET-INT-003: No background class in results
```python
@pytest.mark.integration
def test_no_background_class_in_results():
    result = run_full_pipeline("tests/test_images/detection_people.jpg", mode="detection")
    for det in result.detections:
        assert det.class_id != 0, "Background class should never appear in results"
        assert det.label.lower() != "background"
```

### TC-DET-INT-004: Bounding boxes within image bounds
```python
@pytest.mark.integration
def test_bboxes_within_image_bounds():
    result = run_full_pipeline("tests/test_images/detection_people.jpg", mode="detection")
    img = cv2.imread("tests/test_images/detection_people.jpg")
    H, W = img.shape[:2]
    for det in result.detections:
        b = det.bbox
        assert 0 <= b.x1 < b.x2 <= W, f"Invalid x coords: {b.x1}, {b.x2}"
        assert 0 <= b.y1 < b.y2 <= H, f"Invalid y coords: {b.y1}, {b.y2}"
```

### TC-DET-INT-005: Crowded image — NMS reduces duplicate detections
```python
@pytest.mark.integration
def test_nms_reduces_duplicates():
    """
    Crowded image is designed to produce multiple overlapping raw detections.
    After NMS, overlapping duplicates must be removed.
    """
    result = run_full_pipeline("tests/test_images/detection_crowded.jpg", mode="detection")
    # Verify no two accepted boxes have IoU > 0.4 (NMS threshold)
    boxes = [(d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2) for d in result.detections]
    for i in range(len(boxes)):
        for j in range(i+1, len(boxes)):
            iou = compute_iou(boxes[i], boxes[j])
            assert iou <= 0.4, f"Boxes {i} and {j} overlap too much: IoU={iou:.3f}"
```

### TC-DET-INT-006: full_text is None for detection path
```python
@pytest.mark.integration
def test_full_text_none_for_detection():
    result = run_full_pipeline("tests/test_images/detection_people.jpg", mode="detection")
    assert result.full_text is None
```

---

## 13. Milestone Gate Tests (Evaluator Suite)

**File:** `tests/test_gates.py`  
**Usage:** `pytest tests/test_gates.py -v`  
*These tests mirror exactly what the DecodeLabs evaluator checks. All must pass.*

### GATE 1 — Library Integration

#### TC-GATE1-001: pytesseract importable without error
```python
def test_pytesseract_importable():
    try:
        import pytesseract
    except ImportError:
        pytest.fail("pytesseract not installed. Run: pip install pytesseract")
```

#### TC-GATE1-002: cv2 importable without error
```python
def test_cv2_importable():
    try:
        import cv2
    except ImportError:
        pytest.fail("opencv-python not installed. Run: pip install opencv-python")
```

#### TC-GATE1-003: Model files loadable (detection mode)
```python
@pytest.mark.skipif(PIPELINE_MODE != "detection", reason="Detection mode only")
def test_model_files_loadable():
    try:
        net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)
        assert net is not None
    except cv2.error as e:
        pytest.fail(f"Failed to load model: {e}")
```

#### TC-GATE1-004: requirements.txt exists and is non-empty
```python
def test_requirements_txt_exists():
    assert os.path.isfile("requirements.txt")
    assert os.path.getsize("requirements.txt") > 0
```

---

### GATE 2 — Pre-Processing Integrity

#### TC-GATE2-001: Source code contains cvtColor call
```python
def test_source_contains_cvtcolor():
    """Checks that grayscale conversion is actually in the source code."""
    source = _read_source_files()
    assert "cv2.cvtColor" in source and "COLOR_BGR2GRAY" in source, \
        "Grayscale conversion (cv2.cvtColor with COLOR_BGR2GRAY) not found in source."
```

#### TC-GATE2-002: Source code contains GaussianBlur call
```python
def test_source_contains_gaussianblur():
    source = _read_source_files()
    assert "GaussianBlur" in source, \
        "cv2.GaussianBlur not found in source code."
```

#### TC-GATE2-003: Source code contains threshold call
```python
def test_source_contains_threshold():
    source = _read_source_files()
    has_adaptive = "adaptiveThreshold" in source
    has_otsu = "THRESH_OTSU" in source
    assert has_adaptive or has_otsu, \
        "Neither adaptiveThreshold nor THRESH_OTSU found in source code."
```

#### TC-GATE2-004: Pre-processed output visually verifiable (binary saved)
```python
@pytest.mark.integration
def test_preprocessing_output_saved_to_disk(tmp_path, monkeypatch):
    monkeypatch.setattr("src.config.OUTPUT_DIR", str(tmp_path))
    run_full_pipeline("tests/test_images/ocr_clean.png", mode=PIPELINE_MODE)
    # At minimum, the final output image must be on disk
    output_files = list(tmp_path.glob("*_output*"))
    assert len(output_files) > 0, "No output image saved — Gate 2 visual confirmation requires output file."
```

---

### GATE 3 — Accuracy Benchmarking

#### TC-GATE3-001: CONFIDENCE_THRESHOLD constant exists in source and equals 0.80
```python
def test_confidence_threshold_constant_exists():
    source = _read_source_files()
    assert "CONFIDENCE_THRESHOLD" in source, \
        "CONFIDENCE_THRESHOLD constant not found in source code."
    
    # Import and verify actual value
    from src.config import CONFIDENCE_THRESHOLD
    assert CONFIDENCE_THRESHOLD == 0.80, \
        f"CONFIDENCE_THRESHOLD must be exactly 0.80, found: {CONFIDENCE_THRESHOLD}"
```

#### TC-GATE3-002: CONFIDENCE_THRESHOLD is the value used in the actual filter (not a different literal)
```python
def test_threshold_used_in_filter_not_hardcoded():
    """
    Check that the filter uses the named constant, not a hardcoded literal.
    This prevents the 'conf >= 0.5' bypass.
    """
    source = _read_source_files()
    import re
    # Find all numeric comparisons involving confidence-like variables
    # Flag any hardcoded float < 0.80 in a comparison
    numeric_comparisons = re.findall(r'(?:conf|confidence|score)\s*[><=!]+\s*([0-9.]+)', source)
    for val in numeric_comparisons:
        v = float(val)
        assert v >= 0.80 or v == 0.0, \
            f"Found hardcoded confidence comparison value {v} < 0.80. Use CONFIDENCE_THRESHOLD."
```

#### TC-GATE3-003: Sample test image produces ≥1 detection at 80%
```python
@pytest.mark.integration
def test_sample_image_min_one_detection():
    result = run_full_pipeline(SAMPLE_IMAGE_PATH, mode=PIPELINE_MODE)
    assert result.total_accepted >= 1, \
        f"Sample image produced zero high-confidence results. Check image quality or pipeline."
```

#### TC-GATE3-004: No output contains confidence below threshold
```python
@pytest.mark.integration
def test_no_output_below_threshold():
    result = run_full_pipeline(SAMPLE_IMAGE_PATH, mode=PIPELINE_MODE)
    for det in result.detections:
        assert det.confidence >= 0.80, \
            f"Detection in output has confidence {det.confidence:.3f} — below threshold!"
```

---

### GATE 4 — Visual Confirmation

#### TC-GATE4-001: Output image file exists after run
```python
@pytest.mark.integration
def test_output_image_file_exists(tmp_path, monkeypatch):
    monkeypatch.setattr("src.config.OUTPUT_DIR", str(tmp_path))
    result = run_full_pipeline(SAMPLE_IMAGE_PATH, mode=PIPELINE_MODE)
    assert os.path.isfile(result.output_path), \
        f"Output image not found at: {result.output_path}"
```

#### TC-GATE4-002: Output image is non-zero bytes
```python
@pytest.mark.integration
def test_output_image_non_zero_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr("src.config.OUTPUT_DIR", str(tmp_path))
    result = run_full_pipeline(SAMPLE_IMAGE_PATH, mode=PIPELINE_MODE)
    assert os.path.getsize(result.output_path) > 0
```

#### TC-GATE4-003: Output image is readable by OpenCV (valid format)
```python
@pytest.mark.integration
def test_output_image_readable(tmp_path, monkeypatch):
    monkeypatch.setattr("src.config.OUTPUT_DIR", str(tmp_path))
    result = run_full_pipeline(SAMPLE_IMAGE_PATH, mode=PIPELINE_MODE)
    loaded = cv2.imread(result.output_path)
    assert loaded is not None, "Output image cannot be re-loaded by OpenCV — may be corrupt."
```

#### TC-GATE4-004: Output image same dimensions as input
```python
@pytest.mark.integration
def test_output_image_same_dimensions(tmp_path, monkeypatch):
    monkeypatch.setattr("src.config.OUTPUT_DIR", str(tmp_path))
    original = cv2.imread(SAMPLE_IMAGE_PATH)
    result = run_full_pipeline(SAMPLE_IMAGE_PATH, mode=PIPELINE_MODE)
    output = cv2.imread(result.output_path)
    assert original.shape == output.shape, \
        f"Output image shape {output.shape} differs from input {original.shape}"
```

#### TC-GATE4-005: Console output contains label and confidence percentage
```python
@pytest.mark.integration
def test_console_has_label_and_percentage(capsys):
    result = run_full_pipeline(SAMPLE_IMAGE_PATH, mode=PIPELINE_MODE)
    captured = capsys.readouterr()
    assert "%" in captured.out, "Console output missing confidence percentage"
```

---

## 14. Adversarial Tests (Loophole Coverage)

**File:** `tests/test_adversarial.py`  
*These tests specifically target the 7 loopholes identified in PRD.md §12 plus additional gaming strategies.*

### Loophole 1: image_to_string() bypass

#### TC-ADV-001: image_to_string usage detected and rejected
```python
def test_image_to_string_not_used():
    """
    image_to_string() returns no confidence data.
    If used, the trainee cannot actually filter by confidence.
    """
    source = _read_source_files()
    assert "image_to_string" not in source, \
        "image_to_string() detected. Must use image_to_data() to access per-word confidence scores."
```

### Loophole 2: Output image only shown, not saved

#### TC-ADV-002: imwrite must be called, not only imshow
```python
def test_imwrite_in_source():
    source = _read_source_files()
    assert "imwrite" in source, \
        "cv2.imwrite not found in source. Output image must be saved to disk (Gate 4)."
```

### Loophole 3: Bounding boxes not scaled

#### TC-ADV-003: Scaled coordinates differ from raw normalized values
```python
def test_coordinate_scaling_actually_multiplies():
    """
    If coordinates are not scaled, they remain near 0.0–1.0 as floats.
    Scaled pixel coordinates for a 640×480 image should be 100–600 range.
    """
    raw = (0.2, 0.3, 0.7, 0.8)  # Normalized
    W, H = 640, 480
    bbox = scale_coordinates(raw, W, H)
    assert bbox[0] > 1, f"x1={bbox[0]} looks unscaled (still near 0–1 float range)"
    assert bbox[2] > 1, f"x2={bbox[2]} looks unscaled"
```

### Loophole 4: No input file type validation

#### TC-ADV-004: Non-image file rejected before processing
```python
def test_text_file_rejected_cleanly(tmp_path):
    """A .txt file must be caught at validation, not mid-pipeline with a cryptic cv2.error."""
    p = tmp_path / "fake_image.txt"
    p.write_text("this is not an image")
    with pytest.raises((ValidationError, SystemExit)):
        run_full_pipeline(str(p), mode="ocr")
```

### Loophole 5: Pre-processing result not used as model input

#### TC-ADV-005: OCR receives binary image, not raw BGR
```python
def test_ocr_receives_preprocessed_not_raw(monkeypatch):
    """
    Intercept the pytesseract call and verify the image passed to it
    is grayscale/binary, not the original 3-channel BGR.
    """
    captured_inputs = []
    original_fn = pytesseract.image_to_data
    
    def spy_fn(img, **kwargs):
        captured_inputs.append(img)
        return original_fn(img, **kwargs)
    
    monkeypatch.setattr("pytesseract.image_to_data", spy_fn)
    run_full_pipeline("tests/test_images/ocr_clean.png", mode="ocr")
    
    assert len(captured_inputs) == 1
    img_passed = captured_inputs[0]
    # Must be 2D (grayscale/binary), not 3D (BGR)
    assert img_passed.ndim == 2, \
        f"Tesseract received a {img_passed.ndim}D image — must be preprocessed grayscale/binary."
```

### Loophole 6: Zero detections silent exit

#### TC-ADV-006: Blank image produces explicit message, exit code 0
```python
@pytest.mark.integration
def test_zero_results_produces_message(capsys):
    result = run_full_pipeline("tests/test_images/ocr_blank.png", mode="ocr")
    assert result.total_accepted == 0
    captured = capsys.readouterr()
    assert "No high-confidence" in captured.out, \
        "Silent exit on zero results — must print explicit message per FR-09."
```

### Loophole 7: Threshold lowered in source

#### TC-ADV-007: CONFIDENCE_THRESHOLD value verified in source code
```python
def test_confidence_threshold_value_in_source():
    """
    A trainee might set CONFIDENCE_THRESHOLD = 0.80 visibly but then use
    a different literal in the actual if-statement.
    This test checks the named constant value.
    """
    from src.config import CONFIDENCE_THRESHOLD
    assert CONFIDENCE_THRESHOLD >= 0.80, \
        f"CONFIDENCE_THRESHOLD = {CONFIDENCE_THRESHOLD} is below the required 0.80."
    assert CONFIDENCE_THRESHOLD <= 1.0, \
        f"CONFIDENCE_THRESHOLD = {CONFIDENCE_THRESHOLD} is above 1.0 — nonsensical value."
```

### Additional Gaming Strategies

#### TC-ADV-008: Model not loaded at runtime (lazy import bypass)
```python
def test_model_loaded_before_inference():
    """Verify the model load happens in the pipeline, not in a no-op stub."""
    source = _read_source_files()
    assert "readNetFromCaffe" in source or "pytesseract" in source, \
        "No model loading call found in source — pipeline appears to be a stub."
```

#### TC-ADV-009: Tesseract conf=-1 rows not included in output
```python
def test_conf_neg1_rows_not_in_output(sample_tesseract_dataframe):
    """Layout rows (conf=-1) must never appear as word detections."""
    df = filter_dataframe(sample_tesseract_dataframe)
    assert -1 not in df['conf'].values, \
        "conf=-1 layout rows found in filtered DataFrame — they must be removed."
```

#### TC-ADV-010: Empty text strings not in output
```python
def test_empty_strings_not_in_word_output(sample_tesseract_dataframe):
    """Empty or whitespace-only text tokens must never appear in WordDetection results."""
    df = filter_dataframe(sample_tesseract_dataframe)
    for text in df['text']:
        assert len(text.strip()) > 0, f"Empty text token in output: '{text}'"
```

#### TC-ADV-011: Confidence 0.7999 is not rounded up to 0.80
```python
def test_confidence_not_rounded_up():
    """Floating point rounding must not cause 0.7999 to be accepted."""
    result = confidence_filter([{"label": "test", "confidence": 0.7999}])
    assert len(result) == 0, \
        "0.7999 must be rejected — boundary is exclusive below 0.80."
```

#### TC-ADV-012: Detection with inverted coordinates (x1>x2) is silently skipped
```python
def test_inverted_detection_skipped():
    raw = np.zeros((1, 1, 1, 7), dtype=np.float32)
    raw[0, 0, 0] = [0, 15, 0.95, 0.8, 0.2, 0.3, 0.9]  # x_start(0.8) > x_end(0.3)
    result = parse_detections(raw, image_shape=(480, 640, 3), labels=LABELS)
    assert len(result) == 0, "Inverted bounding box should be silently discarded."
```

---

## 15. Regression Test Checklist

Run this checklist manually before any submission or after any code change:

```
[ ] Pipeline runs to completion on ocr_clean.png without errors
[ ] Pipeline runs to completion on detection_people.jpg without errors
[ ] Output file appears in /output/ directory after each run
[ ] Console shows formatted output block with confidence percentages
[ ] Changing PIPELINE_MODE in config.py switches behavior correctly
[ ] Changing TESSERACT_PSM in config.py changes Tesseract behavior (OCR)
[ ] No detections below 80% appear in console output or output image
[ ] Zero-results message appears when running on ocr_blank.png
[ ] Corrupt input file produces a readable error message (not a stack trace)
[ ] Two runs of the same image produce identical output files (determinism)
[ ] Script works from a fresh terminal session (no dependency on previous state)
```

---

## 16. Test Image Specifications

Test images must meet these specifications for test results to be reproducible:

| Image | Dimensions | Content | Expected Result |
|-------|-----------|---------|----------------|
| `ocr_clean.png` | ≥ 400×200 | Black text on white, ≥ 3 words, standard font | ≥ 3 words at ≥ 80% confidence |
| `ocr_noisy.png` | ≥ 400×200 | Same text, Gaussian noise added (σ=15) | ≥ 1 word at ≥ 80% confidence |
| `ocr_skewed.png` | ≥ 400×200 | Same text, rotated 5° | ≥ 1 word (tests deskew) |
| `ocr_blank.png` | ≥ 400×200 | Pure white (255,255,255) | 0 detections |
| `detection_people.jpg` | ≥ 640×480 | 1–3 clearly visible standing people | ≥ 1 "person" at ≥ 80% |
| `detection_vehicles.jpg` | ≥ 640×480 | 1–2 clearly visible cars | ≥ 1 vehicle class at ≥ 80% |
| `detection_empty.jpg` | ≥ 640×480 | Sky or solid color — no COCO classes | 0 detections |
| `detection_crowded.jpg` | ≥ 640×480 | 5+ people, some overlapping | Tests NMS — no overlapping boxes |
| `tiny.png` | 10×10 | Any content | Rejected by size validation |
| `corrupt.jpg` | N/A | Truncated JPEG header | Rejected by imread returning None |
| `zero_bytes.jpg` | N/A | Empty file | Rejected by file size validation |

---

## 17. Test Execution Guide

### Running All Tests
```bash
pytest tests/ -v --tb=short 2>&1 | tee test_results.txt
```

### Running Only Unit Tests (no model required)
```bash
pytest tests/ -v -m "not integration"
```

### Running Gate Tests Only (evaluator mode)
```bash
pytest tests/test_gates.py -v
```

### Running Adversarial Tests
```bash
pytest tests/test_adversarial.py -v
```

### Coverage Report
```bash
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

**Minimum acceptable coverage: 80% line coverage on `src/`**

### Interpreting Results

```
PASSED  — Test passes. No action required.
FAILED  — Test fails. Fix the indicated code before submission.
ERROR   — Test errored (setup issue, missing file). Fix environment first.
SKIPPED — Test skipped (integration test, model not installed). Acceptable if labeled.
XFAIL   — Expected failure. Fine to leave.
```

### Pre-submission Gate Check

```bash
# This is the exact command the evaluator runs.
# All 5 lines must show PASSED.
pytest tests/test_gates.py::TC-GATE1-001 \
       tests/test_gates.py::TC-GATE2-001 \
       tests/test_gates.py::TC-GATE3-001 \
       tests/test_gates.py::TC-GATE4-001 \
       tests/test_adversarial.py -v
```

---

*End of TEST_PLAN.md*
