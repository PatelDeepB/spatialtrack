"""Projection package handling camera calibration, homography transforms, and coordinate mapping."""

from spatialtrack.projection.calibration import (
    load_calibration,
    save_calibration,
    validate_calibration_points,
)
from spatialtrack.projection.coordinate_mapper import CoordinateMapper
from spatialtrack.projection.homography import (
    compute_homography_matrices,
    project_points,
)

__all__ = [
    "CoordinateMapper",
    "compute_homography_matrices",
    "load_calibration",
    "project_points",
    "save_calibration",
    "validate_calibration_points",
]
