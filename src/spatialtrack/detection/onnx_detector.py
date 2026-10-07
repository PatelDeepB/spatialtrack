"""ONNX Runtime detector implementation with CPU optimization."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from spatialtrack.core.config import DetectionConfig
from spatialtrack.core.exceptions import InferenceError, ModelLoadError
from spatialtrack.core.types import Detection, FrameIndex
from spatialtrack.detection.base_detector import BaseDetector
from spatialtrack.detection.postprocessing import decode_detections
from spatialtrack.detection.preprocessing import preprocess_frame


class OnnxDetector(BaseDetector):
    """Object detector powered by ONNX Runtime optimized for CPU inference."""

    def __init__(self, config: DetectionConfig) -> None:
        """Initialize the ONNX Runtime session with thread and execution provider configuration.

        Args:
            config: DetectionConfig specifying model path, input size, and thresholds.

        Raises:
            ModelLoadError: If model file cannot be found or initialized by ONNX Runtime.
        """
        self.config = config
        self._session = self._create_inference_session(config.model_path, config.num_threads)
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

    @staticmethod
    def _create_inference_session(model_path: Path, num_threads: int) -> ort.InferenceSession:
        """Create and configure ONNX Runtime InferenceSession on CPU."""
        if not model_path.is_file():
            raise ModelLoadError(f"ONNX model file not found at: {model_path}")

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = num_threads
        session_options.inter_op_num_threads = 1
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        try:
            return ort.InferenceSession(
                str(model_path),
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )
        except Exception as err:
            raise ModelLoadError(f"Failed to load ONNX model from {model_path}: {err}") from err

    def detect(self, frame: NDArray[np.uint8], frame_index: FrameIndex) -> list[Detection]:
        """Execute detection pipeline on a single image frame.

        Args:
            frame: Input BGR image array with shape (height, width, 3).
            frame_index: Current sequential frame index.

        Returns:
            List of detected objects with bounding boxes and confidence scores.

        Raises:
            InferenceError: If model forward pass fails.
        """
        tensor, metadata = preprocess_frame(frame, target_size=self.config.input_size)

        try:
            raw_outputs = self._session.run(
                [self._output_name],
                {self._input_name: tensor},
            )
        except Exception as err:
            raise InferenceError(f"ONNX inference failed on frame {frame_index}: {err}") from err

        return decode_detections(
            raw_proposals=raw_outputs[0],
            metadata=metadata,
            confidence_threshold=self.config.confidence_threshold,
            target_classes=self.config.target_classes,
            frame_index=frame_index,
            iou_threshold=self.config.nms_iou_threshold,
        )
