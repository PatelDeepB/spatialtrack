"""Unit tests for the post-processing module."""

import numpy as np

from spatialtrack.core.types import FrameIndex, ObjectClass
from spatialtrack.detection.postprocessing import (
    apply_non_maximum_suppression,
    calculate_iou_vectorized,
    decode_detections,
    rescale_box_to_original,
)
from spatialtrack.detection.preprocessing import PreprocessMetadata


def test_calculate_iou_vectorized_exact_and_disjoint() -> None:
    """Verify IoU calculation on identical and disjoint boxes."""
    # Arrange
    query_box = np.array([10.0, 10.0, 50.0, 50.0], dtype=np.float32)
    candidate_boxes = np.array(
        [
            [10.0, 10.0, 50.0, 50.0],  # Exact overlap -> IoU = 1.0
            [100.0, 100.0, 150.0, 150.0],  # Disjoint -> IoU = 0.0
            [10.0, 10.0, 30.0, 50.0],  # Half overlap
        ],
        dtype=np.float32,
    )

    # Act
    ious = calculate_iou_vectorized(query_box, candidate_boxes)

    # Assert
    assert pytest_approx(ious[0], 1.0)
    assert pytest_approx(ious[1], 0.0)
    assert ious[2] > 0.4 and ious[2] < 0.6


def test_apply_non_maximum_suppression() -> None:
    """Verify NMS suppresses overlapping redundant proposals."""
    # Arrange
    boxes = np.array(
        [
            [10.0, 10.0, 50.0, 50.0],
            [12.0, 12.0, 52.0, 52.0],  # High overlap with box 0
            [200.0, 200.0, 250.0, 250.0],  # Distinct object
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.75, 0.85], dtype=np.float32)

    # Act
    keep_indices = apply_non_maximum_suppression(boxes, scores, iou_threshold=0.45)

    # Assert
    assert 0 in keep_indices
    assert 2 in keep_indices
    assert 1 not in keep_indices  # Box 1 suppressed


def test_rescale_box_to_original() -> None:
    """Verify coordinate rescaling from letterbox padding back to original image dimensions."""
    # Arrange
    # Original image 1280x720 scaled by 0.5 to 640x360 with pad_top=140
    metadata = PreprocessMetadata(
        scale_ratio=0.5,
        pad_left=0.0,
        pad_top=140.0,
        original_height=720,
        original_width=1280,
    )
    # Box on letterbox canvas at y in [140, 320]
    canvas_box = (100.0, 140.0, 200.0, 320.0)

    # Act
    rescaled_bbox = rescale_box_to_original(canvas_box, metadata)

    # Assert
    assert rescaled_bbox.x1 == 200.0  # 100 / 0.5
    assert rescaled_bbox.x2 == 400.0  # 200 / 0.5
    assert rescaled_bbox.y1 == 0.0  # (140 - 140) / 0.5
    assert rescaled_bbox.y2 == 360.0  # (320 - 140) / 0.5


def test_decode_detections_yolov10_format() -> None:
    """Verify end-to-end decoding with (N, 6) YOLOv10 format."""
    # Arrange
    metadata = PreprocessMetadata(
        scale_ratio=1.0,
        pad_left=0.0,
        pad_top=0.0,
        original_height=640,
        original_width=640,
    )
    # Class 2 in COCO is "car"
    proposals = np.array(
        [
            [50.0, 60.0, 150.0, 180.0, 0.85, 2.0],  # Valid car
            [10.0, 10.0, 40.0, 40.0, 0.15, 2.0],  # Below confidence threshold
            [20.0, 20.0, 80.0, 80.0, 0.90, 14.0],  # Class 14 is "bird" (not in target)
        ],
        dtype=np.float32,
    )

    # Act
    detections = decode_detections(
        raw_proposals=proposals,
        metadata=metadata,
        confidence_threshold=0.25,
        target_classes=["car", "truck"],
        frame_index=FrameIndex(1),
    )

    # Assert
    assert len(detections) == 1
    assert detections[0].object_class == ObjectClass.CAR
    assert pytest_approx(float(detections[0].confidence), 0.85)
    assert detections[0].bbox.x1 == 50.0


def pytest_approx(a: float, b: float, tol: float = 1e-4) -> bool:
    """Helper for floating point approximate equality."""
    return abs(a - b) <= tol
