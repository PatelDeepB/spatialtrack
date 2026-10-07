"""Telemetry data export module writing per-frame spatial records to CSV format."""

import csv
import io
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from spatialtrack.core.exceptions import ExportError
from spatialtrack.core.types import SpatialRecord


class TelemetryExporter:
    """Exports per-frame spatial telemetry records to structured CSV files."""

    CSV_HEADERS: tuple[str, ...] = (
        "frame_index",
        "track_id",
        "pixel_x",
        "pixel_y",
        "world_x",
        "world_y",
        "speed_ms",
        "speed_kmh",
        "is_speed_violation",
        "object_class",
    )

    def __init__(self, file_path: str | Path) -> None:
        """Initialize exporter with target CSV destination path.

        Args:
            file_path: Output CSV file destination.
        """
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._file: io.TextIOWrapper | None = None
        self._writer: Any | None = None

    def open(self) -> Self:
        """Open CSV file and write header row if newly created."""
        try:
            file_is_new = not self.file_path.is_file() or self.file_path.stat().st_size == 0
            self._file = open(self.file_path, "a", newline="", encoding="utf-8")
            self._writer = csv.writer(self._file)
            if file_is_new:
                self._writer.writerow(self.CSV_HEADERS)
                self._file.flush()
            return self
        except OSError as err:
            raise ExportError(f"Failed to open telemetry CSV at {self.file_path}: {err}") from err

    def export_record(self, record: SpatialRecord) -> None:
        """Write a single SpatialRecord as a CSV row."""
        if self._file is None or self._writer is None:
            self.open()

        row = [
            int(record.frame_index),
            int(record.track_id),
            round(float(record.pixel_position[0]), 1),
            round(float(record.pixel_position[1]), 1),
            round(float(record.world_position[0]), 2),
            round(float(record.world_position[1]), 2),
            round(float(record.speed_ms), 2),
            round(float(record.speed_kmh), 2),
            int(record.is_speed_violation),
            record.object_class.name.lower(),
        ]
        assert self._writer is not None
        self._writer.writerow(row)

    def export_records(self, records: Sequence[SpatialRecord]) -> None:
        """Write a sequence of SpatialRecord objects and flush."""
        for record in records:
            self.export_record(record)
        if self._file is not None:
            self._file.flush()

    def close(self) -> None:
        """Close CSV file stream."""
        if self._file is not None and not self._file.closed:
            self._file.close()
            self._file = None
            self._writer = None

    def __enter__(self) -> Self:
        """Context manager entry point."""
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Context manager exit point releasing file resources."""
        self.close()
