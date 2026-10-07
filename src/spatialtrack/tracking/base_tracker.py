"""Abstract protocol for multi-object tracking algorithms."""

from collections.abc import Sequence
from typing import Protocol

from spatialtrack.core.types import Detection, FrameIndex, Track


class BaseTracker(Protocol):
    """Protocol defining the interface for multi-object trackers."""

    def update(
        self,
        detections: Sequence[Detection],
        frame_index: FrameIndex,
    ) -> list[Track]:
        """Update tracker state with new detections from the current frame.

        Args:
            detections: Sequence of detections emitted by object detector.
            frame_index: Sequential index of the processed frame.

        Returns:
            List of confirmed active tracks with persistent IDs and trajectory trails.
        """
        ...

    def reset(self) -> None:
        """Reset internal tracker state, clearing all active and historic tracks."""
        ...
