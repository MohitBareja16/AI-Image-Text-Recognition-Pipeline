# ─────────────────────────────────────────────────────────────────────────────
# tests/test_output_save.py — Unit Tests: Output & Save (TEST_PLAN §9)
# ─────────────────────────────────────────────────────────────────────────────

import os
import pytest
import numpy as np
import cv2

from src.postprocessor import draw_ocr_boxes, draw_detection_boxes, save_output, print_summary
from src.schemas import BBox, WordDetection, ObjectDetection, PipelineResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_pipeline_result(mode="ocr", num_detections=2, total_raw=5):
    """Creates a minimal PipelineResult for print_summary testing."""
    if mode == "ocr":
        dets = [
            WordDetection(
                text=f"Word{i}",
                confidence=0.90 + i * 0.01,
                bbox=BBox(x1=10*i+1, y1=10, x2=10*i+50, y2=40),
                line_num=1, block_num=1, word_num=i + 1,
            )
            for i in range(num_detections)
        ]
        full_text = " ".join(d.text for d in dets)
        full_text_val = full_text
    else:
        dets = [
            ObjectDetection(
                label="person",
                class_id=15,
                confidence=0.91,
                bbox=BBox(x1=10, y1=10, x2=100, y2=100),
            )
            for _ in range(num_detections)
        ]
        full_text_val = None

    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    return PipelineResult(
        mode=mode,
        input_path="/some/path/test.jpg",
        output_path="/some/path/output/test_output.jpg",
        detections=dets,
        annotated_image=img,
        full_text=full_text_val,
        total_raw=total_raw,
        total_accepted=num_detections,
        total_rejected=total_raw - num_detections,
        runtime_seconds=1.43,
        threshold_used=0.80,
    )


# ──────────────────────────────────────────────────────────────────────────────

# TC-OUT-001: Output file created on disk
def test_output_file_created(tmp_path, synthetic_text_image, monkeypatch):
    monkeypatch.setattr("src.config.OUTPUT_DIR", str(tmp_path))
    saved_path = save_output(synthetic_text_image, str(tmp_path / "test.jpg"))
    assert os.path.isfile(saved_path), f"Output file not found: {saved_path}"


# TC-OUT-002: Output file is not zero bytes
def test_output_file_not_empty(tmp_path, synthetic_text_image, monkeypatch):
    monkeypatch.setattr("src.config.OUTPUT_DIR", str(tmp_path))
    saved_path = save_output(synthetic_text_image, str(tmp_path / "test.jpg"))
    assert os.path.getsize(saved_path) > 0


# TC-OUT-003: Output filename has correct "_output" suffix
def test_output_filename_suffix(tmp_path, synthetic_text_image, monkeypatch):
    monkeypatch.setattr("src.config.OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr("src.config.OUTPUT_SUFFIX", "_output")
    saved_path = save_output(synthetic_text_image, "/some/path/invoice.jpg")
    assert os.path.basename(saved_path) == "invoice_output.jpg", \
        f"Expected 'invoice_output.jpg', got '{os.path.basename(saved_path)}'"


# TC-OUT-004: Output directory auto-created if not exists
def test_output_dir_created(tmp_path, synthetic_text_image, monkeypatch):
    new_dir = tmp_path / "new_output_dir"
    assert not new_dir.exists()
    monkeypatch.setattr("src.config.OUTPUT_DIR", str(new_dir))
    save_output(synthetic_text_image, str(tmp_path / "test.jpg"))
    assert new_dir.exists(), "Output directory should have been auto-created"


# TC-OUT-005: Annotated image same shape as original
def test_annotated_image_same_shape_ocr(synthetic_text_image):
    annotated = draw_ocr_boxes(synthetic_text_image, [])
    assert annotated.shape == synthetic_text_image.shape


def test_annotated_image_same_shape_detection(synthetic_text_image):
    annotated = draw_detection_boxes(synthetic_text_image, [])
    assert annotated.shape == synthetic_text_image.shape


# TC-OUT-006: Annotation operates on copy — original unchanged
def test_annotation_does_not_modify_original_ocr(synthetic_text_image):
    original_copy = synthetic_text_image.copy()
    _ = draw_ocr_boxes(synthetic_text_image, [])
    np.testing.assert_array_equal(synthetic_text_image, original_copy)


def test_annotation_does_not_modify_original_detection(synthetic_text_image):
    original_copy = synthetic_text_image.copy()
    _ = draw_detection_boxes(synthetic_text_image, [])
    np.testing.assert_array_equal(synthetic_text_image, original_copy)


# TC-OUT-007: Console output contains confidence values
def test_console_output_has_confidence(capsys):
    result = _make_pipeline_result(mode="detection", num_detections=1)
    print_summary(result)
    captured = capsys.readouterr()
    assert "%" in captured.out, "Console output missing confidence percentage sign"


# TC-OUT-008: Console output contains "No high-confidence" when empty
def test_empty_result_message(capsys):
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    result = PipelineResult(
        mode="detection",
        input_path="/test.jpg",
        output_path="/output/test_output.jpg",
        detections=[],
        annotated_image=img,
        full_text=None,
        total_raw=5,
        total_accepted=0,
        total_rejected=5,
        runtime_seconds=0.5,
        threshold_used=0.80,
    )
    print_summary(result)
    captured = capsys.readouterr()
    assert "No high-confidence" in captured.out


# TC-OUT-009: Output image is readable by OpenCV after save
def test_output_image_readable_by_opencv(tmp_path, synthetic_text_image, monkeypatch):
    monkeypatch.setattr("src.config.OUTPUT_DIR", str(tmp_path))
    saved_path = save_output(synthetic_text_image, str(tmp_path / "test.jpg"))
    reloaded = cv2.imread(saved_path)
    assert reloaded is not None, "Output image cannot be re-loaded by OpenCV"


# TC-OUT-010: draw_ocr_boxes with actual WordDetection draws without crash
def test_draw_ocr_with_detections(synthetic_text_image):
    words = [
        WordDetection(
            text="HELLO",
            confidence=0.95,
            bbox=BBox(x1=50, y1=30, x2=250, y2=90),
            line_num=1, block_num=1, word_num=1,
        )
    ]
    annotated = draw_ocr_boxes(synthetic_text_image, words)
    assert annotated.shape == synthetic_text_image.shape
    # Annotated should differ from original (box was drawn)
    assert not np.array_equal(annotated, synthetic_text_image)


# TC-OUT-011: draw_detection_boxes with actual ObjectDetection draws without crash
def test_draw_detection_with_detections(synthetic_text_image):
    detections = [
        ObjectDetection(
            label="person",
            class_id=15,
            confidence=0.91,
            bbox=BBox(x1=10, y1=10, x2=80, y2=90),
        )
    ]
    annotated = draw_detection_boxes(synthetic_text_image, detections)
    assert annotated.shape == synthetic_text_image.shape
