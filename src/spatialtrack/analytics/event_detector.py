"""Debounced speed violation detection for temporal telemetry events."""

from collections.abc import Sequence
from dataclasses import dataclass

from spatialtrack.core.config import AnalyticsConfig
from spatialtrack.core.types import EventType, SpatialEvent, SpatialRecord, TrackId


@dataclass(slots=True)
class _TrackViolationState:
    """Internal state for debouncing speed violations across consecutive frames."""

    consecutive_frames: int = 0
    has_emitted_violation: bool = False


class SpeedViolationDetector:
    """Detects and debounces speed violations, emitting discrete SpatialEvent instances."""

    def __init__(self, config: AnalyticsConfig, fps: float) -> None:
        """Initialize detector with configuration and video frame rate.

        Args:
            config: AnalyticsConfig specifying debounce frames and limits.
            fps: Video frames per second for precise timestamping.
        """
        self.config = config
        self._fps = max(1.0, float(fps))
        self._track_states: dict[TrackId, _TrackViolationState] = {}

    def update(self, records: Sequence[SpatialRecord]) -> list[SpatialEvent]:
        """Evaluate spatial records and emit debounced speed violation events.

        Args:
            records: Current frame spatial records.

        Returns:
            List of emitted SpatialEvent objects for the current frame.
        """
        events: list[SpatialEvent] = []

        for record in records:
            event = self._evaluate_single_record(record)
            if event is not None:
                events.append(event)

        return events

    def _evaluate_single_record(self, record: SpatialRecord) -> SpatialEvent | None:
        """Evaluate single record and return event if debounce threshold met."""
        state = self._track_states.setdefault(record.track_id, _TrackViolationState())

        if not record.is_speed_violation:
            state.consecutive_frames = 0
            state.has_emitted_violation = False
            return None

        state.consecutive_frames += 1

        if (
            state.consecutive_frames >= self.config.speed_violation_debounce_frames
            and not state.has_emitted_violation
        ):
            state.has_emitted_violation = True
            timestamp_sec = float(record.frame_index) / self._fps
            excess_speed = float(record.speed_kmh) - self.config.speed_limit_kmh

            return SpatialEvent(
                event_type=EventType.SPEED_VIOLATION,
                track_id=record.track_id,
                frame_index=record.frame_index,
                timestamp_sec=timestamp_sec,
                world_position=record.world_position,
                metadata={
                    "speed_kmh": round(float(record.speed_kmh), 2),
                    "speed_limit_kmh": round(self.config.speed_limit_kmh, 2),
                    "excess_kmh": round(excess_speed, 2),
                    "object_class": record.object_class.name.lower(),
                },
            )

        return None

    def remove_track(self, track_id: TrackId) -> None:
        """Purge state for a retired track ID."""
        self._track_states.pop(track_id, None)
