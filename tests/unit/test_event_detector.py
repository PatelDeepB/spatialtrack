"""Unit tests for the SpeedViolationDetector debouncing logic."""

from spatialtrack.analytics.event_detector import SpeedViolationDetector
from spatialtrack.core.config import AnalyticsConfig
from spatialtrack.core.types import (
    EventType,
    FrameIndex,
    KilometersPerHour,
    MetersPerSecond,
    ObjectClass,
    PixelCoord,
    SpatialRecord,
    TrackId,
    WorldCoord,
)


def make_record(track_id: int, frame: int, speed: float, is_violation: bool) -> SpatialRecord:
    """Helper to instantiate test SpatialRecord."""
    return SpatialRecord(
        track_id=TrackId(track_id),
        frame_index=FrameIndex(frame),
        pixel_position=PixelCoord((100.0, 100.0)),
        world_position=WorldCoord((5.0, 10.0)),
        speed_ms=MetersPerSecond(speed / 3.6),
        speed_kmh=KilometersPerHour(speed),
        object_class=ObjectClass.CAR,
        is_speed_violation=is_violation,
    )


def test_speed_violation_debounce_and_deduplication() -> None:
    """Verify that violation emits exactly once after debounce_frames threshold."""
    # Arrange: debounce = 3 frames
    config = AnalyticsConfig(speed_violation_debounce_frames=3, speed_limit_kmh=60.0)
    detector = SpeedViolationDetector(config, fps=30.0)

    # Frame 1: speeding (count = 1) -> no event yet
    events1 = detector.update([make_record(1, 1, 75.0, True)])
    assert len(events1) == 0

    # Frame 2: speeding (count = 2) -> no event yet
    events2 = detector.update([make_record(1, 2, 76.0, True)])
    assert len(events2) == 0

    # Frame 3: speeding (count = 3 >= debounce) -> emits event!
    events3 = detector.update([make_record(1, 3, 77.0, True)])
    assert len(events3) == 1
    assert events3[0].event_type == EventType.SPEED_VIOLATION
    assert events3[0].track_id == TrackId(1)
    assert events3[0].metadata["speed_kmh"] == 77.0

    # Frame 4: still speeding -> deduplicated (no duplicate event)
    events4 = detector.update([make_record(1, 4, 78.0, True)])
    assert len(events4) == 0

    # Frame 5: slows down below speed limit -> resets state
    events5 = detector.update([make_record(1, 5, 55.0, False)])
    assert len(events5) == 0
