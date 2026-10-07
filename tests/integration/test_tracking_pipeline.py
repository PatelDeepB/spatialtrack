"""Integration tests for the tracking pipeline across multi-frame sequences."""

from spatialtrack.core.config import TrackingConfig
from spatialtrack.core.types import BoundingBox, Confidence, Detection, FrameIndex, ObjectClass
from spatialtrack.tracking.bytetrack import ByteTracker


def test_moving_object_trajectory_consistency() -> None:
    """Verify that a moving object across 10 frames maintains a single consistent track ID."""
    # Arrange
    config = TrackingConfig(min_hits=2, max_age=10)
    tracker = ByteTracker(config)

    # Simulate a car moving horizontally from x=100 to x=280 at 20 px/frame
    frames_count = 10
    observed_track_ids: list[int] = []

    # Act
    for i in range(frames_count):
        x_offset = 100.0 + i * 20.0
        detection = Detection(
            bbox=BoundingBox(x1=x_offset, y1=150.0, x2=x_offset + 50.0, y2=200.0),
            confidence=Confidence(0.88),
            object_class=ObjectClass.CAR,
            frame_index=FrameIndex(i + 1),
        )
        tracks = tracker.update([detection], FrameIndex(i + 1))
        if tracks:
            observed_track_ids.append(int(tracks[0].track_id))

    # Assert
    # After min_hits=2, the track is confirmed on every frame
    assert len(observed_track_ids) == frames_count - 1
    # All observed track IDs must be identical (no identity switching)
    assert len(set(observed_track_ids)) == 1
    assert observed_track_ids[0] == 1
