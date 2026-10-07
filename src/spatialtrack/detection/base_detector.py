"""Abstract base protocol for object detection models."""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from spatialtrack.core.types import Detection, FrameIndex


class BaseDetector(Protocol):
    """Protocol defining the interface for all object detection backends."""

    def detect(self, frame: NDArray[np.uint8], frame_index: FrameIndex) -> list[Detection]:
        """Detect objects in a single image frame.

        Args:
            frame: Raw BGR input frame with shape (height, width, 3).
            frame_index: Sequential index of the processed frame.

        Returns:
            List of detected objects with bounding boxes and confidence scores.
        """
        ...
