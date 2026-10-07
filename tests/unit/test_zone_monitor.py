"""Unit tests for the ZoneMonitor polygonal boundary tracking."""

from spatialtrack.analytics.zone_monitor import MonitoredZone, ZoneMonitor
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


def make_zone_record(x: float, y: float, frame: int) -> SpatialRecord:
    """Helper to instantiate test SpatialRecord with variable coordinates."""
    return SpatialRecord(
        track_id=TrackId(1),
        frame_index=FrameIndex(frame),
        pixel_position=PixelCoord((100.0, 100.0)),
        world_position=WorldCoord((x, y)),
        speed_ms=MetersPerSecond(10.0),
        speed_kmh=KilometersPerHour(36.0),
        object_class=ObjectClass.CAR,
        is_speed_violation=False,
    )


def test_zone_monitor_entry_dwell_exit_lifecycle() -> None:
    """Verify zone monitor emits entry, dwell, and exit events across object trajectory."""
    # Arrange: 10m x 10m zone from (0, 0) to (10, 10), max_dwell = 1.0 second (30 frames)
    zone = MonitoredZone(
        name="secure_zone",
        polygon_m=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
        zone_type="restricted",
        max_dwell_sec=1.0,
    )
    monitor = ZoneMonitor([zone], fps=30.0)

    # Frame 1: outside at (-5, 5) -> no events
    events1 = monitor.update([make_zone_record(-5.0, 5.0, 1)])
    assert len(events1) == 0

    # Frame 2: enters zone at (5, 5) -> emits ZONE_ENTRY!
    events2 = monitor.update([make_zone_record(5.0, 5.0, 2)])
    assert len(events2) == 1
    assert events2[0].event_type == EventType.ZONE_ENTRY
    assert events2[0].metadata["zone_name"] == "secure_zone"

    # Frame 35 (33 frames later = 1.1s > 1.0s dwell) -> emits ZONE_DWELL!
    events3 = monitor.update([make_zone_record(5.0, 5.0, 35)])
    assert len(events3) == 1
    assert events3[0].event_type == EventType.ZONE_DWELL
    assert float(events3[0].metadata["dwell_duration_sec"]) >= 1.0

    # Frame 40: exits zone to (15, 5) -> emits ZONE_EXIT!
    events4 = monitor.update([make_zone_record(15.0, 5.0, 40)])
    assert len(events4) == 1
    assert events4[0].event_type == EventType.ZONE_EXIT
