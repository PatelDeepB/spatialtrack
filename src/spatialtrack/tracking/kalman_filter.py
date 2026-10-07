"""2D Kalman Filter for bounding box state tracking with constant velocity model."""

import numpy as np
from numpy.typing import NDArray

from spatialtrack.core.types import BoundingBox


class KalmanFilter:
    """Kalman filter for tracking bounding boxes in image pixel space.

    8-dimensional state space:
        [cx, cy, a, h, v_cx, v_cy, v_a, v_h]
    Where:
        (cx, cy): Centroid coordinates of the bounding box.
        a: Aspect ratio (width / height).
        h: Height of the bounding box.
        v_*: Respective velocities in pixels per frame.

    4-dimensional measurement space:
        [cx, cy, a, h]
    """

    def __init__(self) -> None:
        """Initialize constant state transition and measurement matrices."""
        # 8x8 Constant velocity transition matrix
        self._motion_mat = np.eye(8, 8, dtype=np.float32)
        for i in range(4):
            self._motion_mat[i, i + 4] = 1.0

        # 4x8 Measurement projection matrix
        self._update_mat = np.eye(4, 8, dtype=np.float32)

        # Noise scaling parameters based on object height
        self._std_weight_position = 1.0 / 20.0
        self._std_weight_velocity = 1.0 / 160.0

    def initiate(
        self,
        measurement: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Create new track state from initial bounding box measurement.

        Args:
            measurement: 4D bounding box measurement [cx, cy, a, h].

        Returns:
            Tuple of (mean, covariance) for the initial state.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel].astype(np.float32)

        height = float(measurement[3])
        std = [
            2.0 * self._std_weight_position * height,
            2.0 * self._std_weight_position * height,
            1e-2,
            2.0 * self._std_weight_position * height,
            10.0 * self._std_weight_velocity * height,
            10.0 * self._std_weight_velocity * height,
            1e-5,
            10.0 * self._std_weight_velocity * height,
        ]
        covariance = np.diag(np.square(std)).astype(np.float32)
        return mean, covariance

    def predict(
        self,
        mean: NDArray[np.float32],
        covariance: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Advance state distribution one time step forward according to motion model.

        Args:
            mean: Current 8D state mean vector.
            covariance: Current 8x8 state covariance matrix.

        Returns:
            Tuple of (predicted_mean, predicted_covariance).
        """
        height = float(mean[3])
        std_pos = [
            self._std_weight_position * height,
            self._std_weight_position * height,
            1e-2,
            self._std_weight_position * height,
        ]
        std_vel = [
            self._std_weight_velocity * height,
            self._std_weight_velocity * height,
            1e-5,
            self._std_weight_velocity * height,
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel])).astype(np.float32)

        mean_pred = np.dot(self._motion_mat, mean)
        cov_pred = (
            np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        )
        return mean_pred.astype(np.float32), cov_pred.astype(np.float32)

    def update(
        self,
        mean: NDArray[np.float32],
        covariance: NDArray[np.float32],
        measurement: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Compute corrected state distribution using incoming bounding box observation.

        Args:
            mean: Predicted 8D state mean vector.
            covariance: Predicted 8x8 state covariance matrix.
            measurement: Incoming 4D bounding box measurement [cx, cy, a, h].

        Returns:
            Tuple of (updated_mean, updated_covariance).
        """
        height = float(mean[3])
        std = [
            self._std_weight_position * height,
            self._std_weight_position * height,
            1e-1,
            self._std_weight_position * height,
        ]
        meas_cov = np.diag(np.square(std)).astype(np.float32)

        projected_mean = np.dot(self._update_mat, mean)
        projected_cov = (
            np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T)) + meas_cov
        )

        kalman_gain = np.linalg.solve(
            projected_cov,
            np.dot(self._update_mat, covariance.T),
        ).T

        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_cov = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean.astype(np.float32), new_cov.astype(np.float32)

    @staticmethod
    def bbox_to_measurement(bbox: BoundingBox) -> NDArray[np.float32]:
        """Convert BoundingBox into 4D Kalman measurement [cx, cy, a, h]."""
        width = bbox.width
        height = max(1.0, bbox.height)
        center_x = bbox.x1 + width / 2.0
        center_y = bbox.y1 + height / 2.0
        aspect_ratio = width / height
        return np.array([center_x, center_y, aspect_ratio, height], dtype=np.float32)

    @staticmethod
    def state_to_bbox(state_mean: NDArray[np.float32]) -> BoundingBox:
        """Convert 8D Kalman state mean into BoundingBox."""
        center_x = float(state_mean[0])
        center_y = float(state_mean[1])
        aspect_ratio = max(1e-4, float(state_mean[2]))
        height = max(1.0, float(state_mean[3]))
        width = aspect_ratio * height

        x1 = center_x - width / 2.0
        y1 = center_y - height / 2.0
        x2 = center_x + width / 2.0
        y2 = center_y + height / 2.0
        return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
