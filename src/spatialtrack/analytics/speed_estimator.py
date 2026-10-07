"""Metric speed estimation with exponential moving average smoothing and jitter suppression."""

import math
from dataclasses import dataclass

from spatialtrack.core.constants import MS_TO_KMH_MULTIPLIER
from spatialtrack.core.types import (
    FrameIndex,
    KilometersPerHour,
    MetersPerSecond,
    SpatialRecord,
    Track,
    TrackId,
    WorldCoord,
)


@dataclass(slots=True)
class _TrackSpeedState:
    """Internal historical tracking state for a single object speed estimate."""

    previous_position: WorldCoord
    previous_frame_index: FrameIndex
    smoothed_speed_kmh: float


class SpeedEstimator:
    """Calculates smoothed real-world speed (m/s and km/h) from metric coordinate displacements."""

    def __init__(
        self,
        fps: float,
        smoothing_alpha: float = 0.3,
        min_displacement_m: float = 0.05,
        speed_limit_kmh: float = 60.0,
    ) -> None:
        """Initialize speed estimator parameters.

        Args:
            fps: Video frames per second.
            smoothing_alpha: Exponential smoothing weight [0.0, 1.0].
            min_displacement_m: Minimum displacement in meters to filter stationary noise.
            speed_limit_kmh: Speed violation threshold in km/h.
        """
        self._fps = max(1.0, float(fps))
        self._alpha = smoothing_alpha
        self._min_displacement_m = min_displacement_m
        self._speed_limit_kmh = speed_limit_kmh
        self._states: dict[TrackId, _TrackSpeedState] = {}

    def estimate_speed(
        self,
        track: Track,
        world_position: WorldCoord,
        frame_index: FrameIndex,
    ) -> SpatialRecord:
        """Compute metric speed and produce SpatialRecord for a tracked object.

        Args:
            track: Current Track object.
            world_position: Projected metric coordinate (X, Y) in meters.
            frame_index: Current sequential frame index.

        Returns:
            SpatialRecord populated with smoothed speeds and violation flags.
        """
        state = self._states.get(track.track_id)
        if state is None:
            self._states[track.track_id] = _TrackSpeedState(
                previous_position=world_position,
                previous_frame_index=frame_index,
                smoothed_speed_kmh=0.0,
            )
            return self._build_record(track, world_position, frame_index, 0.0, 0.0)

        delta_frames = max(1, int(frame_index) - int(state.previous_frame_index))
        delta_time_sec = delta_frames / self._fps
        dx = world_position[0] - state.previous_position[0]
        dy = world_position[1] - state.previous_position[1]
        displacement_m = math.hypot(dx, dy)

        if displacement_m < self._min_displacement_m:
            instant_speed_ms = 0.0
            instant_speed_kmh = 0.0
        else:
            instant_speed_ms = displacement_m / delta_time_sec
            instant_speed_kmh = instant_speed_ms * MS_TO_KMH_MULTIPLIER

        # Apply exponential moving average (EMA)
        smoothed_kmh = (
            self._alpha * instant_speed_kmh + (1.0 - self._alpha) * state.smoothed_speed_kmh
        )
        smoothed_ms = smoothed_kmh / MS_TO_KMH_MULTIPLIER

        state.previous_position = world_position
        state.previous_frame_index = frame_index
        state.smoothed_speed_kmh = smoothed_kmh

        return self._build_record(track, world_position, frame_index, smoothed_ms, smoothed_kmh)

    def _build_record(
        self,
        track: Track,
        world_position: WorldCoord,
        frame_index: FrameIndex,
        speed_ms: float,
        speed_kmh: float,
    ) -> SpatialRecord:
        """Construct immutable SpatialRecord."""
        is_violation = speed_kmh > self._speed_limit_kmh
        return SpatialRecord(
            track_id=track.track_id,
            frame_index=frame_index,
            pixel_position=track.bbox.bottom_center,
            world_position=world_position,
            speed_ms=MetersPerSecond(speed_ms),
            speed_kmh=KilometersPerHour(speed_kmh),
            object_class=track.object_class,
            is_speed_violation=is_violation,
        )

    def remove_track(self, track_id: TrackId) -> None:
        """Remove cached speed state for a deleted track."""
        self._states.pop(track_id, None)
