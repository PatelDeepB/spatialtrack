"""Unit tests for the association and matching module."""

import numpy as np

from spatialtrack.core.types import BoundingBox
from spatialtrack.tracking.association import compute_iou_matrix, linear_assignment


def test_compute_iou_matrix_pairwise() -> None:
    """Verify pairwise IoU matrix computation between two box sets."""
    # Arrange
    boxes_a = [
        BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0),
        BoundingBox(x1=50.0, y1=50.0, x2=60.0, y2=60.0),
    ]
    boxes_b = [
        BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0),  # Exact match with a[0]
        BoundingBox(x1=200.0, y1=200.0, x2=210.0, y2=210.0),  # Disjoint
    ]

    # Act
    iou_mat = compute_iou_matrix(boxes_a, boxes_b)

    # Assert
    assert iou_mat.shape == (2, 2)
    assert abs(iou_mat[0, 0] - 1.0) < 1e-4  # Match
    assert abs(iou_mat[0, 1] - 0.0) < 1e-4  # Disjoint
    assert abs(iou_mat[1, 0] - 0.0) < 1e-4  # Disjoint


def test_linear_assignment_optimal_matching() -> None:
    """Verify Hungarian solver pairs lowest cost pairs and rejects above threshold."""
    # Arrange
    cost_matrix = np.array(
        [
            [0.1, 0.9],
            [0.8, 0.2],
        ],
        dtype=np.float32,
    )

    # Act
    matches, unmatched_r, unmatched_c = linear_assignment(cost_matrix, threshold=0.5)

    # Assert
    assert (0, 0) in matches
    assert (1, 1) in matches
    assert len(unmatched_r) == 0
    assert len(unmatched_c) == 0


def test_linear_assignment_gating_threshold() -> None:
    """Verify that matches exceeding cost threshold are flagged as unmatched."""
    # Arrange
    cost_matrix = np.array([[0.8]], dtype=np.float32)

    # Act
    matches, unmatched_r, unmatched_c = linear_assignment(cost_matrix, threshold=0.5)

    # Assert
    assert len(matches) == 0
    assert 0 in unmatched_r
    assert 0 in unmatched_c
