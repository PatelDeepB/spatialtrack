"""Unit tests for the ByteTracker implementation."""

from spatialtrack.core.config import TrackingConfig
from spatialtrack.core.types import BoundingBox, Confidence, Detection, FrameIndex, ObjectClass
from spatialtrack.tracking.bytetrack import ByteTracker


def test_bytetrack_initiates_and_tracks_object() -> None:
    """Verify that ByteTracker maintains persistent track ID across successive frames."""
    # Arrange
    config = TrackingConfig(min_hits=1, max_age=5)
    tracker = ByteTracker(config)

    det_frame1 = Detection(
        bbox=BoundingBox(x1=100.0, y1=100.0, x2=150.0, y2=150.0),
        confidence=Confidence(0.9),
        object_class=ObjectClass.CAR,
        frame_index=FrameIndex(1),
    )
    det_frame2 = Detection(
        bbox=BoundingBox(x1=105.0, y1=105.0, x2=155.0, y2=155.0),
        confidence=Confidence(0.85),
        object_class=ObjectClass.CAR,
        frame_index=FrameIndex(2),
    )

    # Act
    tracks_f1 = tracker.update([det_frame1], FrameIndex(1))
    tracks_f2 = tracker.update([det_frame2], FrameIndex(2))

    # Assert
    assert len(tracks_f1) == 1
    assert len(tracks_f2) == 1
    # Track ID must be persistent
    assert tracks_f1[0].track_id == tracks_f2[0].track_id
    # Pixel trail should have accumulated 2 positions
    assert len(tracks_f2[0].pixel_trail) == 2


def test_bytetrack_deletes_lost_tracks_after_max_age() -> None:
    """Verify that tracks without observations are removed after max_age frames."""
    # Arrange
    config = TrackingConfig(min_hits=1, max_age=3)
    tracker = ByteTracker(config)

    det = Detection(
        bbox=BoundingBox(x1=50.0, y1=50.0, x2=90.0, y2=90.0),
        confidence=Confidence(0.9),
        object_class=ObjectClass.CAR,
        frame_index=FrameIndex(1),
    )

    # Act
    tracker.update([det], FrameIndex(1))

    # Send 4 empty frames (exceeds max_age=3)
    for i in range(2, 6):
        active_tracks = tracker.update([], FrameIndex(i))

    # Assert
    assert len(active_tracks) == 0
    # Both active and lost pools should be empty
    assert len(tracker._tracked_tracks) == 0
    assert len(tracker._lost_tracks) == 0
