"""Unit tests for the SpeedEstimator module."""

from spatialtrack.analytics.speed_estimator import SpeedEstimator
from spatialtrack.core.types import (
    BoundingBox,
    Confidence,
    FrameIndex,
    ObjectClass,
    Track,
    TrackId,
    WorldCoord,
)


def test_speed_estimator_calculation_and_smoothing() -> None:
    """Verify speed calculation on known displacement over time."""
    # Arrange: 30 FPS, alpha=1.0 (no smoothing, instant speed for exact math)
    estimator = SpeedEstimator(fps=30.0, smoothing_alpha=1.0, min_displacement_m=0.01)

    track = Track(
        track_id=TrackId(1),
        bbox=BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0),
        object_class=ObjectClass.CAR,
        confidence=Confidence(0.9),
        age=1,
        time_since_update=0,
    )

    # Frame 1: at (0, 0)
    record1 = estimator.estimate_speed(track, WorldCoord((0.0, 0.0)), FrameIndex(1))
    assert record1.speed_kmh == 0.0

    # Frame 31 (1 second later): at (0, 10.0) -> 10 meters in 1 second = 10 m/s = 36.0 km/h
    record2 = estimator.estimate_speed(track, WorldCoord((0.0, 10.0)), FrameIndex(31))

    # Assert
    assert abs(record2.speed_ms - 10.0) < 1e-2
    assert abs(record2.speed_kmh - 36.0) < 1e-2
    assert not record2.is_speed_violation


def test_speed_estimator_violation_flag() -> None:
    """Verify that speed exceeding threshold sets is_speed_violation to True."""
    # Arrange: speed limit = 50 km/h, moving at 20 m/s = 72 km/h
    estimator = SpeedEstimator(
        fps=30.0,
        smoothing_alpha=1.0,
        min_displacement_m=0.01,
        speed_limit_kmh=50.0,
    )
    track = Track(
        track_id=TrackId(2),
        bbox=BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0),
        object_class=ObjectClass.CAR,
        confidence=Confidence(0.9),
        age=1,
        time_since_update=0,
    )

    estimator.estimate_speed(track, WorldCoord((0.0, 0.0)), FrameIndex(1))
    # 20 meters in 30 frames (1 second) = 72 km/h (> 50 km/h limit)
    record = estimator.estimate_speed(track, WorldCoord((0.0, 20.0)), FrameIndex(31))

    # Assert
    assert record.is_speed_violation
    assert record.speed_kmh > 50.0


def test_speed_estimator_micro_jitter_suppression() -> None:
    """Verify that tiny displacement below threshold is treated as 0 speed."""
    # Arrange
    estimator = SpeedEstimator(fps=30.0, smoothing_alpha=1.0, min_displacement_m=0.1)
    track = Track(
        track_id=TrackId(3),
        bbox=BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0),
        object_class=ObjectClass.CAR,
        confidence=Confidence(0.9),
        age=1,
        time_since_update=0,
    )

    estimator.estimate_speed(track, WorldCoord((0.0, 0.0)), FrameIndex(1))
    # 0.02 meters displacement (< 0.1m min threshold)
    record = estimator.estimate_speed(track, WorldCoord((0.0, 0.02)), FrameIndex(2))

    # Assert
    assert record.speed_kmh == 0.0
