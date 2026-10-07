"""Unit tests for structured JSON Lines EventLogger."""

import json
from pathlib import Path

from spatialtrack.core.types import EventType, FrameIndex, SpatialEvent, TrackId, WorldCoord
from spatialtrack.io.event_logger import EventLogger


def test_event_logger_writes_json_lines(tmp_path: Path) -> None:
    """Verify that EventLogger appends valid JSON Lines records."""
    # Arrange
    log_file = tmp_path / "events.jsonl"
    event = SpatialEvent(
        event_type=EventType.SPEED_VIOLATION,
        track_id=TrackId(42),
        frame_index=FrameIndex(100),
        timestamp_sec=3.333,
        world_position=WorldCoord((10.5, 25.2)),
        metadata={"speed_kmh": 85.4},
    )

    # Act
    with EventLogger(log_file) as logger:
        logger.log_event(event)

    # Assert
    assert log_file.is_file()
    lines = log_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1

    parsed = json.loads(lines[0])
    assert parsed["event_type"] == "speed_violation"
    assert parsed["track_id"] == 42
    assert parsed["metadata"]["speed_kmh"] == 85.4
