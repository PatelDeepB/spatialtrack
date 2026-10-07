"""Polygonal zone boundary monitoring and dwell-time detection."""

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from spatialtrack.core.types import EventType, SpatialEvent, SpatialRecord, TrackId


@dataclass(frozen=True, slots=True)
class MonitoredZone:
    """Geofenced polygonal zone defined in metric world coordinates."""

    name: str
    polygon_m: list[tuple[float, float]]
    zone_type: str = "restricted"
    max_dwell_sec: float = 30.0


@dataclass(slots=True)
class _TrackZonePresence:
    """Tracks presence of an object inside a specific zone."""

    entry_time_sec: float = 0.0
    is_inside: bool = False
    dwell_event_emitted: bool = False


class ZoneMonitor:
    """Monitors vehicles relative to polygon zones, emitting entry, exit, and dwell events."""

    def __init__(self, zones: Sequence[MonitoredZone], fps: float) -> None:
        """Initialize monitor with target zones and video frame rate.

        Args:
            zones: Sequence of MonitoredZone polygon boundaries.
            fps: Video frames per second.
        """
        self.zones = list(zones)
        self._fps = max(1.0, float(fps))
        self._contours: dict[str, NDArray[np.float32]] = {
            zone.name: np.array(zone.polygon_m, dtype=np.float32) for zone in self.zones
        }
        # Mapping: (track_id, zone_name) -> _TrackZonePresence
        self._presence: dict[tuple[TrackId, str], _TrackZonePresence] = {}

    def update(self, records: Sequence[SpatialRecord]) -> list[SpatialEvent]:
        """Check all spatial records against zones and return emitted events.

        Args:
            records: Current frame spatial records.

        Returns:
            List of emitted zone-related SpatialEvents.
        """
        events: list[SpatialEvent] = []

        for record in records:
            for zone in self.zones:
                zone_events = self._evaluate_track_zone(record, zone)
                events.extend(zone_events)

        return events

    def _evaluate_track_zone(
        self,
        record: SpatialRecord,
        zone: MonitoredZone,
    ) -> list[SpatialEvent]:
        """Evaluate single track position relative to a zone boundary."""
        contour = self._contours[zone.name]
        pos_x, pos_y = record.world_position
        is_inside_now = cv2.pointPolygonTest(contour, (pos_x, pos_y), False) >= 0

        key = (record.track_id, zone.name)
        presence = self._presence.setdefault(key, _TrackZonePresence())
        current_time_sec = float(record.frame_index) / self._fps
        emitted: list[SpatialEvent] = []

        # 1. Entry event
        if is_inside_now and not presence.is_inside:
            presence.is_inside = True
            presence.entry_time_sec = current_time_sec
            presence.dwell_event_emitted = False
            emitted.append(self._create_event(EventType.ZONE_ENTRY, record, zone, current_time_sec))

        # 2. Dwell event
        elif is_inside_now and presence.is_inside and not presence.dwell_event_emitted:
            dwell_duration = current_time_sec - presence.entry_time_sec
            if dwell_duration >= zone.max_dwell_sec:
                presence.dwell_event_emitted = True
                emitted.append(
                    self._create_event(
                        EventType.ZONE_DWELL,
                        record,
                        zone,
                        current_time_sec,
                        dwell_duration_sec=round(dwell_duration, 2),
                    )
                )

        # 3. Exit event
        elif not is_inside_now and presence.is_inside:
            presence.is_inside = False
            dwell_duration = current_time_sec - presence.entry_time_sec
            emitted.append(
                self._create_event(
                    EventType.ZONE_EXIT,
                    record,
                    zone,
                    current_time_sec,
                    dwell_duration_sec=round(dwell_duration, 2),
                )
            )

        return emitted

    def _create_event(
        self,
        event_type: EventType,
        record: SpatialRecord,
        zone: MonitoredZone,
        timestamp_sec: float,
        dwell_duration_sec: float | None = None,
    ) -> SpatialEvent:
        """Helper to instantiate typed SpatialEvent for zone triggers."""
        metadata: dict[str, str | int | float | bool] = {
            "zone_name": zone.name,
            "zone_type": zone.zone_type,
            "object_class": record.object_class.name.lower(),
        }
        if dwell_duration_sec is not None:
            metadata["dwell_duration_sec"] = dwell_duration_sec

        return SpatialEvent(
            event_type=event_type,
            track_id=record.track_id,
            frame_index=record.frame_index,
            timestamp_sec=timestamp_sec,
            world_position=record.world_position,
            metadata=metadata,
        )

    def remove_track(self, track_id: TrackId) -> None:
        """Purge presence states for a track."""
        for zone in self.zones:
            self._presence.pop((track_id, zone.name), None)
