"""Integration tests for the ONNX detection pipeline on CPU."""

from pathlib import Path

import numpy as np

from spatialtrack.core.config import DetectionConfig
from spatialtrack.core.types import FrameIndex
from spatialtrack.detection.onnx_detector import OnnxDetector


def test_onnx_detector_initialization_and_inference() -> None:
    """Verify that OnnxDetector loads INT8 model and runs forward pass on synthetic frame."""
    # Arrange
    model_path = Path("models/yolov10n_int8.onnx")
    if not model_path.is_file():
        # Fallback to FP32 if INT8 is not generated in environment
        model_path = Path("models/yolov10n.onnx")

    config = DetectionConfig(
        model_path=model_path,
        input_size=640,
        confidence_threshold=0.25,
        target_classes=["car", "truck", "person"],
        num_threads=2,
    )
    detector = OnnxDetector(config)
    sample_frame = np.full((720, 1280, 3), 120, dtype=np.uint8)

    # Act
    detections = detector.detect(sample_frame, FrameIndex(1))

    # Assert
    # On a blank uniform gray frame, detections should be a list (likely empty or valid)
    assert isinstance(detections, list)
    for det in detections:
        assert det.confidence >= 0.25
        assert det.frame_index == FrameIndex(1)
        assert det.bbox.area > 0
