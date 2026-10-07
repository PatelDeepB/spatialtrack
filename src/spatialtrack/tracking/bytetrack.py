"""ByteTrack multi-object tracking implementation with two-stage association."""

from collections.abc import Sequence
from enum import Enum, auto

from spatialtrack.core.config import TrackingConfig
from spatialtrack.core.types import (
    Detection,
    FrameIndex,
    PixelCoord,
    Track,
    TrackId,
)
from spatialtrack.tracking.association import compute_iou_matrix, linear_assignment
from spatialtrack.tracking.base_tracker import BaseTracker
from spatialtrack.tracking.kalman_filter import KalmanFilter


class TrackState(Enum):
    """Lifecycle states of a tracked visual object."""

    TENTATIVE = auto()
    CONFIRMED = auto()
    LOST = auto()
    DELETED = auto()


class STrack:
    """Internal single-object tracking representation managing state and Kalman distribution."""

    def __init__(
        self,
        detection: Detection,
        track_id: TrackId,
        kalman_filter: KalmanFilter,
    ) -> None:
        """Initialize a new track from a single detection."""
        self.track_id = track_id
        self._kalman_filter = kalman_filter
        self.object_class = detection.object_class
        self.confidence = detection.confidence

        measurement = kalman_filter.bbox_to_measurement(detection.bbox)
        self.mean, self.covariance = kalman_filter.initiate(measurement)
        self.bbox = detection.bbox

        self.age = 1
        self.hits = 1
        self.time_since_update = 0
        self.state = TrackState.TENTATIVE
        self.pixel_trail: list[PixelCoord] = [detection.bbox.bottom_center]
        self.velocity_px: tuple[float, float] = (0.0, 0.0)

    def predict(self) -> None:
        """Advance track state forward using Kalman filter."""
        self.mean, self.covariance = self._kalman_filter.predict(self.mean, self.covariance)
        self.bbox = self._kalman_filter.state_to_bbox(self.mean)
        self.age += 1
        self.time_since_update += 1

    def update(self, detection: Detection, max_trail_length: int) -> None:
        """Update track state with a matched detection."""
        measurement = self._kalman_filter.bbox_to_measurement(detection.bbox)
        self.mean, self.covariance = self._kalman_filter.update(
            self.mean, self.covariance, measurement
        )
        self.bbox = self._kalman_filter.state_to_bbox(self.mean)
        self.confidence = detection.confidence
        self.object_class = detection.object_class
        self.hits += 1
        self.time_since_update = 0

        # Update trajectory trail and compute instantaneous pixel velocity
        current_pos = self.bbox.bottom_center
        if self.pixel_trail:
            prev_pos = self.pixel_trail[-1]
            self.velocity_px = (current_pos[0] - prev_pos[0], current_pos[1] - prev_pos[1])
        self.pixel_trail.append(current_pos)
        if len(self.pixel_trail) > max_trail_length:
            self.pixel_trail.pop(0)

    def to_public_track(self) -> Track:
        """Convert internal state representation to public immutable Track dataclass."""
        return Track(
            track_id=self.track_id,
            bbox=self.bbox,
            object_class=self.object_class,
            confidence=self.confidence,
            age=self.age,
            time_since_update=self.time_since_update,
            pixel_trail=list(self.pixel_trail),
            velocity_px=self.velocity_px,
            is_confirmed=(self.state == TrackState.CONFIRMED),
        )


class ByteTracker(BaseTracker):
    """Production-grade ByteTrack implementation with two-stage bipartite matching."""

    def __init__(self, config: TrackingConfig) -> None:
        """Initialize tracker with hyperparameters and state pools."""
        self.config = config
        self._kalman_filter = KalmanFilter()
        self._next_id: int = 1
        self._tracked_tracks: list[STrack] = []
        self._lost_tracks: list[STrack] = []

    def reset(self) -> None:
        """Clear all active tracks and reset ID counter."""
        self._next_id = 1
        self._tracked_tracks.clear()
        self._lost_tracks.clear()

    def update(
        self,
        detections: Sequence[Detection],
        frame_index: FrameIndex,
    ) -> list[Track]:
        """Perform two-stage tracking association on incoming detections."""
        high_dets, low_dets = self._split_detections(detections)

        # Predict existing tracks forward
        all_active = self._tracked_tracks + self._lost_tracks
        for track in all_active:
            track.predict()

        # Stage 1: Associate active tracks with high-confidence detections
        rem_tracks, rem_high_dets = self._associate_stage_one(all_active, high_dets)

        # Stage 2: Associate remaining tracks with low-confidence detections
        unmatched_tracks = self._associate_stage_two(rem_tracks, low_dets)

        # Manage lifecycle: init new tracks, update confirmed / lost states
        self._manage_track_lifecycle(rem_high_dets, unmatched_tracks)

        # Return only confirmed active tracks
        return [
            track.to_public_track()
            for track in self._tracked_tracks
            if track.state == TrackState.CONFIRMED
        ]

    def _split_detections(
        self,
        detections: Sequence[Detection],
    ) -> tuple[list[Detection], list[Detection]]:
        """Partition detections into high and low confidence subsets."""
        high: list[Detection] = []
        low: list[Detection] = []
        for det in detections:
            if det.confidence >= self.config.high_threshold:
                high.append(det)
            elif det.confidence >= self.config.low_threshold:
                low.append(det)
        return high, low

    def _associate_stage_one(
        self,
        tracks: list[STrack],
        high_dets: list[Detection],
    ) -> tuple[list[STrack], list[Detection]]:
        """Associate tracks with high-confidence detections via IoU matching."""
        if not tracks or not high_dets:
            return tracks, high_dets

        track_boxes = [t.bbox for t in tracks]
        det_boxes = [d.bbox for d in high_dets]
        cost_matrix = 1.0 - compute_iou_matrix(track_boxes, det_boxes)

        matches, unmatched_t, unmatched_d = linear_assignment(
            cost_matrix, threshold=1.0 - self.config.iou_threshold
        )

        for t_idx, d_idx in matches:
            tracks[t_idx].update(high_dets[d_idx], self.config.max_trail_length)
            if (
                tracks[t_idx].state == TrackState.TENTATIVE
                and tracks[t_idx].hits >= self.config.min_hits
            ):
                tracks[t_idx].state = TrackState.CONFIRMED
            elif tracks[t_idx].state == TrackState.LOST:
                tracks[t_idx].state = TrackState.CONFIRMED

        rem_tracks = [tracks[i] for i in unmatched_t]
        rem_dets = [high_dets[j] for j in unmatched_d]
        return rem_tracks, rem_dets

    def _associate_stage_two(
        self,
        rem_tracks: list[STrack],
        low_dets: list[Detection],
    ) -> list[STrack]:
        """Associate remaining tracks with low-confidence detections."""
        if rem_tracks and low_dets:
            track_boxes = [t.bbox for t in rem_tracks]
            det_boxes = [d.bbox for d in low_dets]
            cost_matrix = 1.0 - compute_iou_matrix(track_boxes, det_boxes)

            matches, unmatched_t, _ = linear_assignment(
                cost_matrix, threshold=1.0 - self.config.iou_threshold
            )

            for t_idx, d_idx in matches:
                rem_tracks[t_idx].update(low_dets[d_idx], self.config.max_trail_length)
                if (
                    rem_tracks[t_idx].state == TrackState.TENTATIVE
                    and rem_tracks[t_idx].hits >= self.config.min_hits
                ):
                    rem_tracks[t_idx].state = TrackState.CONFIRMED
                elif rem_tracks[t_idx].state == TrackState.LOST:
                    rem_tracks[t_idx].state = TrackState.CONFIRMED

            return [rem_tracks[i] for i in unmatched_t]

        return rem_tracks

    def _manage_track_lifecycle(
        self,
        unmatched_high_dets: list[Detection],
        unmatched_tracks: list[STrack],
    ) -> None:
        """Spawn new tentative tracks and retire expired tracks."""
        # Initialize new tracks for unmatched high-confidence detections
        for det in unmatched_high_dets:
            new_track = STrack(det, TrackId(self._next_id), self._kalman_filter)
            self._next_id += 1
            if self.config.min_hits <= 1:
                new_track.state = TrackState.CONFIRMED
            self._tracked_tracks.append(new_track)

        # Handle unmatched tracks (mark lost or remove)
        for track in unmatched_tracks:
            if track.state == TrackState.CONFIRMED:
                track.state = TrackState.LOST

        # Prune tracks that exceeded max_age or tentative unconfirmed tracks
        self._prune_expired_tracks()

    def _prune_expired_tracks(self) -> None:
        """Filter tracks into tracked and lost pools, removing dead tracks."""
        new_tracked: list[STrack] = []
        new_lost: list[STrack] = []

        all_tracks = self._tracked_tracks + self._lost_tracks
        for track in all_tracks:
            if track.time_since_update > self.config.max_age:
                track.state = TrackState.DELETED
                continue
            if track.state == TrackState.TENTATIVE and track.time_since_update > 1:
                track.state = TrackState.DELETED
                continue

            if track.state in (TrackState.CONFIRMED, TrackState.TENTATIVE):
                if track.time_since_update == 0:
                    new_tracked.append(track)
                else:
                    new_lost.append(track)
            elif track.state == TrackState.LOST:
                new_lost.append(track)

        self._tracked_tracks = new_tracked
        self._lost_tracks = new_lost
