"""Homography computation and perspective coordinate transformation."""

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from spatialtrack.core.exceptions import CalibrationError
from spatialtrack.core.types import PixelCoord, WorldCoord


def compute_homography_matrices(
    source_points: Sequence[PixelCoord],
    target_points: Sequence[WorldCoord],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Compute forward and inverse 3x3 perspective homography transformation matrices.

    Args:
        source_points: Exactly 4 points in pixel coordinates (u, v).
        target_points: Exactly 4 corresponding points in metric world coordinates (X, Y).

    Returns:
        Tuple of (forward_matrix, inverse_matrix) as 3x3 float32 arrays.

    Raises:
        CalibrationError: If points are degenerate or matrix cannot be inverted.
    """
    if len(source_points) != 4 or len(target_points) != 4:
        raise CalibrationError("Perspective transform requires exactly 4 point correspondences.")

    src_arr = np.array(source_points, dtype=np.float32)
    dst_arr = np.array(target_points, dtype=np.float32)

    raw_h = cv2.getPerspectiveTransform(src_arr, dst_arr)
    if raw_h is None or not np.all(np.isfinite(raw_h)):
        raise CalibrationError("Failed to solve perspective transformation matrix.")

    forward_h: NDArray[np.float64] = np.asarray(raw_h, dtype=np.float64)
    determinant = float(np.linalg.det(forward_h))
    if abs(determinant) < 1e-7:
        raise CalibrationError("Degenerate homography matrix with near-zero determinant.")

    inverse_h = np.linalg.inv(forward_h)
    return forward_h.astype(np.float32), inverse_h.astype(np.float32)


def project_points(
    points: NDArray[np.float32],
    transform_matrix: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Apply 3x3 homography matrix to an array of 2D coordinates.

    Args:
        points: (N, 2) array of coordinates.
        transform_matrix: 3x3 perspective transformation matrix.

    Returns:
        (N, 2) array of transformed coordinates.
    """
    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    # Convert to homogeneous coordinates (N, 3)
    num_pts = points.shape[0]
    ones = np.ones((num_pts, 1), dtype=np.float32)
    homo_coords = np.hstack([points, ones])

    # Matrix multiplication: (3, 3) dot (3, N) -> (3, N)
    transformed = np.dot(transform_matrix, homo_coords.T).T

    # Perspective division by w
    w = transformed[:, 2:3]
    w_safe = np.where(np.abs(w) < 1e-6, 1e-6, w)

    result_xy: NDArray[np.float32] = (transformed[:, :2] / w_safe).astype(np.float32)
    return result_xy
