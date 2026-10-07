"""Unit tests for the TelemetryExporter CSV writer."""

import csv
from pathlib import Path

from spatialtrack.core.types import (
    FrameIndex,
    KilometersPerHour,
    MetersPerSecond,
    ObjectClass,
    PixelCoord,
    SpatialRecord,
    TrackId,
    WorldCoord,
)
from spatialtrack.io.telemetry_exporter import TelemetryExporter


def test_telemetry_exporter_writes_csv(tmp_path: Path) -> None:
    """Verify that TelemetryExporter generates valid CSV with correct headers and data."""
    # Arrange
    csv_file = tmp_path / "telemetry.csv"
    record = SpatialRecord(
        track_id=TrackId(5),
        frame_index=FrameIndex(12),
        pixel_position=PixelCoord((200.0, 350.0)),
        world_position=WorldCoord((6.5, 18.2)),
        speed_ms=MetersPerSecond(15.0),
        speed_kmh=KilometersPerHour(54.0),
        object_class=ObjectClass.CAR,
        is_speed_violation=False,
    )

    # Act
    with TelemetryExporter(csv_file) as exporter:
        exporter.export_records([record])

    # Assert
    assert csv_file.is_file()
    with open(csv_file, encoding="utf-8") as f:
        reader = list(csv.reader(f))

    assert len(reader) == 2  # Header + 1 record row
    assert reader[0] == list(TelemetryExporter.CSV_HEADERS)
    assert reader[1][0] == "12"  # frame_index
    assert reader[1][1] == "5"  # track_id
    assert reader[1][7] == "54.0"  # speed_kmh
    assert reader[1][9] == "car"  # object_class
