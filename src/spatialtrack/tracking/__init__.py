"""Tracking package providing Kalman state estimation, association, and ByteTrack."""

from spatialtrack.tracking.association import compute_iou_matrix, linear_assignment
from spatialtrack.tracking.base_tracker import BaseTracker
from spatialtrack.tracking.bytetrack import ByteTracker, STrack, TrackState
from spatialtrack.tracking.kalman_filter import KalmanFilter

__all__ = [
    "BaseTracker",
    "ByteTracker",
    "KalmanFilter",
    "STrack",
    "TrackState",
    "compute_iou_matrix",
    "linear_assignment",
]
