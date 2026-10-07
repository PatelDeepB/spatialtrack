"""Post-processing utilities for decoding model outputs, filtering, and rescaling detections."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from spatialtrack.core.constants import COCO_CLASSES
from spatialtrack.core.types import (
    BoundingBox,
    Confidence,
    Detection,
    FrameIndex,
    ObjectClass,
)
from spatialtrack.detection.preprocessing import PreprocessMetadata


def calculate_iou_vectorized(
    box: NDArray[np.float32], boxes: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Calculate Intersection over Union (IoU) between a single box and an array of boxes.

    Args:
        box: Single bounding box [x1, y1, x2, y2].
        boxes: Array of bounding boxes with shape (N, 4).

    Returns:
        1D array of IoU values between box and each candidate box.
    """
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    box_area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )

    union = box_area + boxes_area - intersection
    iou_result: NDArray[np.float32] = (intersection / union).astype(np.float32)
    return iou_result


def apply_non_maximum_suppression(
    boxes: NDArray[np.float32],
    scores: NDArray[np.float32],
    iou_threshold: float,
) -> list[int]:
    """Perform Non-Maximum Suppression (NMS) to eliminate overlapping redundant proposals.

    Args:
        boxes: Bounding boxes in format (N, 4) with [x1, y1, x2, y2].
        scores: Detection confidence scores of shape (N,).
        iou_threshold: Maximum allowable IoU overlap threshold before suppression.

    Returns:
        List of integer indices corresponding to retained detections.
    """
    if len(boxes) == 0:
        return []

    order = np.argsort(scores)[::-1]
    keep_indices: list[int] = []

    while len(order) > 0:
        current_idx = int(order[0])
        keep_indices.append(current_idx)

        if len(order) == 1:
            break

        remaining_indices = order[1:]
        ious = calculate_iou_vectorized(boxes[current_idx], boxes[remaining_indices])
        valid_mask = ious <= iou_threshold
        order = remaining_indices[valid_mask]

    return keep_indices


def rescale_box_to_original(
    box: tuple[float, float, float, float],
    metadata: PreprocessMetadata,
) -> BoundingBox:
    """Transform bounding box from letterbox canvas space to original unpadded image coordinates.

    Args:
        box: Coordinates (x1, y1, x2, y2) in letterbox canvas space.
        metadata: Preprocessing metadata holding scale and padding values.

    Returns:
        BoundingBox clamped to original image dimensions.
    """
    scale = metadata.scale_ratio
    orig_x1 = (box[0] - metadata.pad_left) / scale
    orig_y1 = (box[1] - metadata.pad_top) / scale
    orig_x2 = (box[2] - metadata.pad_left) / scale
    orig_y2 = (box[3] - metadata.pad_top) / scale

    # Clamp coordinates to original image bounds
    clamped_x1 = max(0.0, min(orig_x1, float(metadata.original_width)))
    clamped_y1 = max(0.0, min(orig_y1, float(metadata.original_height)))
    clamped_x2 = max(clamped_x1, min(orig_x2, float(metadata.original_width)))
    clamped_y2 = max(clamped_y1, min(orig_y2, float(metadata.original_height)))

    return BoundingBox(
        x1=float(clamped_x1),
        y1=float(clamped_y1),
        x2=float(clamped_x2),
        y2=float(clamped_y2),
    )


def decode_detections(
    raw_proposals: NDArray[np.float32],
    metadata: PreprocessMetadata,
    confidence_threshold: float,
    target_classes: Sequence[str],
    frame_index: FrameIndex,
    iou_threshold: float = 0.45,
) -> list[Detection]:
    """Decode, filter, suppress, and rescale model predictions into typed Detection instances.

    Supports candidate formats:
      - (N, 6) format: [x1, y1, x2, y2, score, class_id] (YOLOv10 / NMS-free)
      - (N, 4 + num_classes) format: [cx, cy, w, h, class_scores...] (YOLOv8 standard)

    Args:
        raw_proposals: Raw output array from inference engine.
        metadata: Geometry transformation metadata from preprocessing.
        confidence_threshold: Minimum confidence threshold.
        target_classes: Sequence of accepted class label strings.
        frame_index: Sequential frame number.
        iou_threshold: IoU threshold for non-maximum suppression.

    Returns:
        List of finalized, filtered Detection instances.
    """
    proposals = np.squeeze(raw_proposals)
    if proposals.ndim == 1:
        proposals = np.expand_dims(proposals, axis=0)

    if proposals.shape[0] == 0:
        return []

    # Handle shape (C, N) where C is channels and N is candidates
    if proposals.shape[0] < proposals.shape[1] and proposals.shape[0] in (6, 84, 85):
        proposals = proposals.T

    boxes_list: list[tuple[float, float, float, float]] = []
    scores_list: list[float] = []
    classes_list: list[ObjectClass] = []

    target_class_set = {cls.lower().strip() for cls in target_classes}

    if proposals.shape[1] == 6:
        _decode_yolov10_format(
            proposals,
            confidence_threshold,
            target_class_set,
            boxes_list,
            scores_list,
            classes_list,
        )
    else:
        _decode_standard_yolo_format(
            proposals,
            confidence_threshold,
            target_class_set,
            boxes_list,
            scores_list,
            classes_list,
        )

    if not boxes_list:
        return []

    boxes_array = np.array(boxes_list, dtype=np.float32)
    scores_array = np.array(scores_list, dtype=np.float32)

    keep_indices = apply_non_maximum_suppression(boxes_array, scores_array, iou_threshold)

    final_detections: list[Detection] = []
    for idx in keep_indices:
        rescaled_bbox = rescale_box_to_original(boxes_list[idx], metadata)
        if rescaled_bbox.area > 0:
            final_detections.append(
                Detection(
                    bbox=rescaled_bbox,
                    confidence=Confidence(scores_list[idx]),
                    object_class=classes_list[idx],
                    frame_index=frame_index,
                )
            )

    return final_detections


def _decode_yolov10_format(
    proposals: NDArray[np.float32],
    confidence_threshold: float,
    target_class_set: set[str],
    boxes_out: list[tuple[float, float, float, float]],
    scores_out: list[float],
    classes_out: list[ObjectClass],
) -> None:
    """Parse (N, 6) detections in format [x1, y1, x2, y2, score, class_id]."""
    for row in proposals:
        score = float(row[4])
        if score < confidence_threshold:
            continue
        class_id = int(row[5])
        if class_id < 0 or class_id >= len(COCO_CLASSES):
            continue
        label = COCO_CLASSES[class_id]
        if label not in target_class_set:
            continue

        boxes_out.append((float(row[0]), float(row[1]), float(row[2]), float(row[3])))
        scores_out.append(score)
        classes_out.append(ObjectClass.from_label(label))


def _decode_standard_yolo_format(
    proposals: NDArray[np.float32],
    confidence_threshold: float,
    target_class_set: set[str],
    boxes_out: list[tuple[float, float, float, float]],
    scores_out: list[float],
    classes_out: list[ObjectClass],
) -> None:
    """Parse (N, 4 + classes) proposals in format [cx, cy, w, h, class_scores...]."""
    for row in proposals:
        class_scores = row[4:]
        best_class_id = int(np.argmax(class_scores))
        best_score = float(class_scores[best_class_id])
        if best_score < confidence_threshold:
            continue
        if best_class_id >= len(COCO_CLASSES):
            continue
        label = COCO_CLASSES[best_class_id]
        if label not in target_class_set:
            continue

        cx, cy, w, h = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0

        boxes_out.append((x1, y1, x2, y2))
        scores_out.append(best_score)
        classes_out.append(ObjectClass.from_label(label))
