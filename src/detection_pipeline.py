# ─────────────────────────────────────────────────────────────────────────────
# detection_pipeline.py — Path 2: MobileNet-SSD Object Detection Logic
# Implements ARCHITECTURE.md §4 detection_pipeline contract + ALGORITHM_SPEC.md §3
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import time
import logging
import numpy as np
import cv2

from src import config
from src.schemas import (
    PreprocessedBundle,
    PipelineResult,
    ObjectDetection,
    BBox,
    InferenceError,
)
from src.utils import scale_coordinates, validate_bbox_coords, compute_iou
from src.postprocessor import draw_detection_boxes, save_output

logger = logging.getLogger("decodelabs.project4.detection_pipeline")


def run_detection(
    bundle: PreprocessedBundle,
    net: "cv2.dnn_Net",
    labels: list,
) -> PipelineResult:
    """
    Runs the full MobileNet-SSD object detection pipeline.

    Pipeline steps (ALGO §3, §4, §5, §6, §7):
        1. Blob construction (ALGO §3.2)
        2. Forward pass (ALGO §3.3)
        3. Parse & filter raw detections (confidence + background class skip)
        4. Apply Non-Maximum Suppression (ALGO §5)
        5. Build ObjectDetection objects
        6. Annotate image with bounding boxes
        7. Save output image

    Args:
        bundle: PreprocessedBundle from preprocessor.preprocess()
        net:    Loaded cv2.dnn_Net (MobileNet-SSD)
        labels: List of class label strings from coco_labels.txt

    Returns:
        PipelineResult with mode="detection"

    Raises:
        InferenceError: If the forward pass fails with a cv2.error.
    """
    start_time = time.perf_counter()

    image = bundle.original   # Detect on the original BGR image
    H, W = image.shape[:2]

    # ── Step 1: Blob construction (ALGO §3.2, FR-OBJ-02) ─────────────────────
    # Resizes to 300×300, swaps BGR→RGB (swapRB=True), mean-subtracts (127.5),
    # then scales to [-1, 1] via scalefactor=1/127.5.
    logger.info("Constructing 4D blob (size=%s)...", config.BLOB_SIZE)
    blob = cv2.dnn.blobFromImage(
        image=image,
        scalefactor=config.BLOB_SCALE_FACTOR,   # 1 / 127.5
        size=config.BLOB_SIZE,                   # (300, 300)
        mean=config.BLOB_MEAN,                   # (127.5, 127.5, 127.5)
        swapRB=True,    # OpenCV loads BGR; MobileNet was trained on RGB
        crop=False,     # Stretch to 300×300, don't crop (keeps all content)
    )
    logger.debug("Blob shape: %s", blob.shape)  # Expected: (1, 3, 300, 300)

    # ── Step 2: Forward pass (ALGO §3.3) ─────────────────────────────────────
    logger.info("Running MobileNet-SSD forward pass...")
    try:
        net.setInput(blob)
        raw_detections = net.forward()
    except cv2.error as e:
        raise InferenceError(
            stage="inference",
            message=f"Inference failed — cv2 error: {e}",
            recoverable=False,
        )

    # raw_detections shape: (1, 1, N, 7) where N = number of candidate detections
    logger.debug("Forward pass complete. Raw detections array shape: %s", raw_detections.shape)

    # ── Step 3: Parse and filter raw detections ───────────────────────────────
    # Filters by confidence AND skips background class (class_id == 0).
    candidates = _parse_detections(raw_detections, W, H, labels)
    total_raw = raw_detections.shape[2]
    logger.info(
        "After confidence+background filter: %d/%d detections retained",
        len(candidates), total_raw,
    )

    # ── Step 4: Non-Maximum Suppression (ALGO §5) ─────────────────────────────
    # Removes duplicate overlapping boxes for the same object.
    if candidates:
        candidates = _apply_nms(candidates)
        logger.info("After NMS: %d final detections", len(candidates))

    total_accepted = len(candidates)
    total_rejected = total_raw - total_accepted

    # ── Step 5: Build ObjectDetection objects ─────────────────────────────────
    object_detections = []
    for cand in candidates:
        try:
            obj_det = ObjectDetection(
                label=cand["label"],
                class_id=cand["class_id"],
                confidence=cand["confidence"],
                bbox=cand["bbox"],
            )
            object_detections.append(obj_det)
        except (ValueError, TypeError) as e:
            logger.warning("Skipping invalid detection: %s", e)

    # ── Step 6: Annotate image ────────────────────────────────────────────────
    # Boxes drawn on original BGR image for color fidelity.
    annotated = draw_detection_boxes(image, object_detections)

    runtime = time.perf_counter() - start_time
    logger.info("Detection pipeline complete. Runtime: %.2fs", runtime)

    # Zero-results logging (FR-09)
    if total_accepted == 0:
        logger.warning("No high-confidence detections found. Try a clearer image.")

    # ── Step 7: Save output image ─────────────────────────────────────────────
    output_path = _attempt_save(annotated, bundle)

    # ── Construct unified PipelineResult ──────────────────────────────────────
    return PipelineResult(
        mode="detection",
        input_path=getattr(bundle, "original_path", ""),
        output_path=output_path,
        detections=object_detections,
        annotated_image=annotated,
        full_text=None,               # Detection mode never returns text (V-05)
        total_raw=total_raw,
        total_accepted=total_accepted,
        total_rejected=total_rejected,
        runtime_seconds=runtime,
        threshold_used=config.CONFIDENCE_THRESHOLD,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Private helpers
# ══════════════════════════════════════════════════════════════════════════════

def _parse_detections(
    raw: np.ndarray,
    image_width: int,
    image_height: int,
    labels: list,
) -> list:
    """
    Parses the (1, 1, N, 7) raw MobileNet-SSD output array.
    Filters by CONFIDENCE_THRESHOLD and skips background class (class_id == 0).
    Scales normalized coordinates to pixel coordinates (ALGO §6.1).

    Returns list of dicts: [{label, class_id, confidence, bbox}, ...]
    Inverted/zero-area boxes are silently skipped (DATA_SCHEMA.md §3.1).
    """
    results = []
    num_detections = raw.shape[2]

    for i in range(num_detections):
        detection = raw[0, 0, i]

        # Index 2: confidence (float 0.0–1.0)
        confidence = float(detection[2])

        # Clamp confidence to [0, 1] for numerical stability
        confidence = min(1.0, max(0.0, confidence))

        # Apply confidence threshold (FR-OBJ-03, Gate 3)
        if confidence < config.CONFIDENCE_THRESHOLD:
            logger.debug("✗ class_%d: %.1f%% — rejected (below threshold)", int(detection[1]), confidence * 100)
            continue

        # Index 1: class_id — must NOT be 0 (background, DATA_SCHEMA.md V-04)
        class_id = int(detection[1])
        if class_id == 0:
            logger.debug("✗ Background (class_id=0) at %.1f%% — always skipped", confidence * 100)
            continue

        # Resolve label (FR-OBJ-06: from external file, not hard-coded)
        label = labels[class_id] if class_id < len(labels) else f"class_{class_id}"

        # Indices 3–6: normalized bbox coordinates [0, 1]
        # Scale to pixel coordinates using ORIGINAL dimensions (not blob 300×300)
        x1, y1, x2, y2 = scale_coordinates(
            (float(detection[3]), float(detection[4]),
             float(detection[5]), float(detection[6])),
            image_width,
            image_height,
        )

        # Validate bbox geometry (skip inverted/zero-area, DATA_SCHEMA.md §3.1)
        if not validate_bbox_coords(x1, y1, x2, y2):
            logger.warning(
                "Skipping inverted/zero-area box for '%s': (%d,%d,%d,%d)",
                label, x1, y1, x2, y2,
            )
            continue

        try:
            bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2)
        except (ValueError, TypeError) as e:
            logger.warning("Invalid BBox for '%s': %s", label, e)
            continue

        logger.debug("✓ %s: %.1f%% — retained", label, confidence * 100)
        results.append({
            "label":      label,
            "class_id":   class_id,
            "confidence": confidence,
            "bbox":       bbox,
        })

    return results


def _apply_nms(candidates: list) -> list:
    """
    Applies Non-Maximum Suppression to remove duplicate overlapping detections.
    Uses cv2.dnn.NMSBoxes with [x, y, w, h] format.
    (ALGORITHM_SPEC.md §5.2, iou_threshold=0.4)

    Args:
        candidates: List of detection dicts from _parse_detections.

    Returns:
        Filtered list of detection dicts (kept boxes only).
    """
    if not candidates:
        return []

    # Build parallel lists for NMSBoxes
    # cv2.dnn.NMSBoxes requires [x, y, w, h] not [x1, y1, x2, y2] (ALGO §5)
    boxes = [list(det["bbox"].to_xywh()) for det in candidates]
    confidences = [float(det["confidence"]) for det in candidates]

    # Run NMS — score_threshold is applied first, then IoU suppression
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes,
        scores=confidences,
        score_threshold=config.CONFIDENCE_THRESHOLD,   # 0.80
        nms_threshold=0.4,                             # IoU threshold (ALGO §5.2)
    )

    if len(indices) == 0:
        return []

    # cv2.dnn.NMSBoxes returns an array; flatten to a plain list of indices
    kept_indices = indices.flatten().tolist()
    return [candidates[i] for i in kept_indices]


def _attempt_save(annotated: np.ndarray, bundle: PreprocessedBundle) -> str:
    """
    Attempts to save the annotated image. Returns the output path or empty string.
    """
    try:
        input_path = getattr(bundle, "original_path", "unknown.jpg")
        return save_output(annotated, input_path)
    except Exception as e:
        logger.warning("Could not save output image: %s", e)
        return ""
