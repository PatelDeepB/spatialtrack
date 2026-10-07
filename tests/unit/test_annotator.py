"""Unit tests for the visual frame annotator."""

import numpy as np

from spatialtrack.core.config import VisualizationConfig
from spatialtrack.core.types import (
    BoundingBox,
    Confidence,
    Detection,
    FrameIndex,
    ObjectClass,
    PixelCoord,
    Track,
    TrackId,
)
from spatialtrack.visualization.annotator import FrameAnnotator


def test_annotator_draws_without_mutating_original_frame() -> None:
    """Verify that drawing annotations creates a new copy and does not mutate source image."""
    # Arrange
    config = VisualizationConfig(line_thickness=2)
    annotator = FrameAnnotator(config)
    original_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    original_copy = original_frame.copy()

    detection = Detection(
        bbox=BoundingBox(x1=50.0, y1=50.0, x2=150.0, y2=150.0),
        confidence=Confidence(0.88),
        object_class=ObjectClass.CAR,
        frame_index=FrameIndex(1),
    )

    # Act
    annotated = annotator.draw_detections(original_frame, [detection])

    # Assert
    np.testing.assert_array_equal(original_frame, original_copy)
    assert np.any(annotated > 0)
    assert annotated.shape == original_frame.shape


def test_annotator_draws_tracks_with_trails() -> None:
    """Verify that drawing tracks annotates bounding box, badge, and trajectory trails."""
    # Arrange
    config = VisualizationConfig(show_trails=True, show_bboxes=True)
    annotator = FrameAnnotator(config)
    original_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    track = Track(
        track_id=TrackId(7),
        bbox=BoundingBox(x1=100.0, y1=100.0, x2=200.0, y2=200.0),
        object_class=ObjectClass.CAR,
        confidence=Confidence(0.92),
        age=5,
        time_since_update=0,
        pixel_trail=[
            PixelCoord((150.0, 180.0)),
            PixelCoord((150.0, 190.0)),
            PixelCoord((150.0, 200.0)),
        ],
        velocity_px=(0.0, 10.0),
        is_confirmed=True,
    )

    # Act
    annotated = annotator.draw_tracks(original_frame, [track])

    # Assert
    assert np.any(annotated > 0)
    assert annotated.shape == original_frame.shape
