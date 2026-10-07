"""Detection package exposing detector interfaces, preprocessing, and postprocessing."""

from spatialtrack.detection.base_detector import BaseDetector
from spatialtrack.detection.onnx_detector import OnnxDetector
from spatialtrack.detection.postprocessing import (
    apply_non_maximum_suppression,
    calculate_iou_vectorized,
    decode_detections,
    rescale_box_to_original,
)
from spatialtrack.detection.preprocessing import (
    PreprocessMetadata,
    apply_letterbox,
    calculate_letterbox_geometry,
    preprocess_frame,
)

__all__ = [
    "BaseDetector",
    "OnnxDetector",
    "PreprocessMetadata",
    "apply_letterbox",
    "apply_non_maximum_suppression",
    "calculate_iou_vectorized",
    "calculate_letterbox_geometry",
    "decode_detections",
    "preprocess_frame",
    "rescale_box_to_original",
]
