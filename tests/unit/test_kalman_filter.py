"""Unit tests for the 2D Kalman Filter."""

import numpy as np

from spatialtrack.core.types import BoundingBox
from spatialtrack.tracking.kalman_filter import KalmanFilter


def test_kalman_filter_initiate() -> None:
    """Verify that initiate returns valid 8D state vector and 8x8 positive diagonal covariance."""
    # Arrange
    kf = KalmanFilter()
    measurement = np.array([100.0, 150.0, 1.5, 80.0], dtype=np.float32)

    # Act
    mean, cov = kf.initiate(measurement)

    # Assert
    assert mean.shape == (8,)
    assert cov.shape == (8, 8)
    assert np.allclose(mean[:4], measurement)
    assert np.all(mean[4:] == 0.0)  # Initial velocity is zero
    assert np.all(np.diag(cov) > 0.0)  # Covariance must be positive


def test_kalman_filter_predict_and_update_cycle() -> None:
    """Verify that predict advances state and update corrects measurement."""
    # Arrange
    kf = KalmanFilter()
    measurement1 = np.array([100.0, 100.0, 1.0, 50.0], dtype=np.float32)
    measurement2 = np.array([105.0, 105.0, 1.0, 50.0], dtype=np.float32)

    # Act
    mean, cov = kf.initiate(measurement1)
    mean_pred, cov_pred = kf.predict(mean, cov)
    mean_upd, cov_upd = kf.update(mean_pred, cov_pred, measurement2)

    # Assert
    assert mean_pred.shape == (8,)
    # After update with positive displacement, velocity should reflect positive change
    assert mean_upd[4] > 0.0  # vx > 0
    assert mean_upd[5] > 0.0  # vy > 0
    # Uncertainty in position should be reduced after observation
    assert cov_upd[0, 0] < cov_pred[0, 0]


def test_kalman_bbox_conversion_roundtrip() -> None:
    """Verify bounding box to measurement and state to bounding box conversion accuracy."""
    # Arrange
    kf = KalmanFilter()
    original_bbox = BoundingBox(x1=20.0, y1=30.0, x2=80.0, y2=110.0)

    # Act
    measurement = kf.bbox_to_measurement(original_bbox)
    state_mean = np.r_[measurement, np.zeros(4, dtype=np.float32)]
    recovered_bbox = kf.state_to_bbox(state_mean)

    # Assert
    assert abs(recovered_bbox.x1 - original_bbox.x1) < 1e-4
    assert abs(recovered_bbox.y1 - original_bbox.y1) < 1e-4
    assert abs(recovered_bbox.x2 - original_bbox.x2) < 1e-4
    assert abs(recovered_bbox.y2 - original_bbox.y2) < 1e-4
