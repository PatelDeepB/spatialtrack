"""Unit tests for the homography computation and projection module."""

import numpy as np
import pytest

from spatialtrack.core.exceptions import CalibrationError
from spatialtrack.core.types import PixelCoord, WorldCoord
from spatialtrack.projection.homography import compute_homography_matrices, project_points


def test_homography_square_to_metric_roundtrip() -> None:
    """Verify forward and inverse perspective projection on known rectangle correspondences."""
    # Arrange: 100x100 pixel square mapped to 10m x 10m area
    src_pts = [
        PixelCoord((0.0, 0.0)),
        PixelCoord((100.0, 0.0)),
        PixelCoord((100.0, 100.0)),
        PixelCoord((0.0, 100.0)),
    ]
    dst_pts = [
        WorldCoord((0.0, 0.0)),
        WorldCoord((10.0, 0.0)),
        WorldCoord((10.0, 10.0)),
        WorldCoord((0.0, 10.0)),
    ]

    # Act
    forward_h, inverse_h = compute_homography_matrices(src_pts, dst_pts)

    # Test center point (50, 50) -> should be (5.0, 5.0)
    query_px = np.array([[50.0, 50.0]], dtype=np.float32)
    world_projected = project_points(query_px, forward_h)
    pixel_recovered = project_points(world_projected, inverse_h)

    # Assert
    assert abs(world_projected[0, 0] - 5.0) < 1e-3
    assert abs(world_projected[0, 1] - 5.0) < 1e-3
    assert abs(pixel_recovered[0, 0] - 50.0) < 1e-3
    assert abs(pixel_recovered[0, 1] - 50.0) < 1e-3


def test_homography_invalid_points_raises_error() -> None:
    """Verify that fewer than 4 points raises CalibrationError."""
    # Arrange
    src_pts = [PixelCoord((0.0, 0.0)), PixelCoord((10.0, 10.0))]
    dst_pts = [WorldCoord((0.0, 0.0)), WorldCoord((10.0, 10.0))]

    # Act & Assert
    with pytest.raises(CalibrationError):
        compute_homography_matrices(src_pts, dst_pts)
