"""Structured JSON Lines logger for spatial and security events."""

import io
import json
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Self

from spatialtrack.core.exceptions import ExportError
from spatialtrack.core.types import SpatialEvent


class EventLogger:
    """Thread-safe file appender logging structured spatial events in JSON Lines format."""

    def __init__(self, file_path: str | Path) -> None:
        """Initialize logger destination file.

        Args:
            file_path: Destination path for the .jsonl event log file.
        """
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._file: io.TextIOWrapper | None = None

    def open(self) -> Self:
        """Open the target event log file for appending."""
        try:
            self._file = open(self.file_path, "a", encoding="utf-8")
            return self
        except OSError as err:
            raise ExportError(f"Failed to open event log at {self.file_path}: {err}") from err

    def log_event(self, event: SpatialEvent) -> None:
        """Write a single SpatialEvent as a JSON line."""
        if self._file is None:
            self.open()

        data = {
            "event_type": event.event_type.value,
            "track_id": int(event.track_id),
            "frame_index": int(event.frame_index),
            "timestamp_sec": round(event.timestamp_sec, 3),
            "world_position": [round(coord, 2) for coord in event.world_position],
            "metadata": event.metadata,
        }
        assert self._file is not None
        self._file.write(json.dumps(data) + "\n")
        self._file.flush()

    def log_events(self, events: Sequence[SpatialEvent]) -> None:
        """Write multiple SpatialEvent objects."""
        for event in events:
            self.log_event(event)

    def close(self) -> None:
        """Close log file stream."""
        if self._file is not None and not self._file.closed:
            self._file.close()
            self._file = None

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
