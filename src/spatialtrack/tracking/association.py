"""Bipartite matching and cost association for multi-object tracking."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from spatialtrack.core.types import BoundingBox


def compute_iou_matrix(
    boxes_a: Sequence[BoundingBox],
    boxes_b: Sequence[BoundingBox],
) -> NDArray[np.float32]:
    """Compute pairwise Intersection over Union (IoU) matrix between two sets of boxes.

    Args:
        boxes_a: Sequence of N bounding boxes.
        boxes_b: Sequence of M bounding boxes.

    Returns:
        (N, M) float32 matrix containing IoU overlap scores in range [0.0, 1.0].
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    arr_a = np.array([[b.x1, b.y1, b.x2, b.y2] for b in boxes_a], dtype=np.float32)
    arr_b = np.array([[b.x1, b.y1, b.x2, b.y2] for b in boxes_b], dtype=np.float32)

    # Broadcast to (N, M, 2)
    top_left = np.maximum(arr_a[:, None, :2], arr_b[None, :, :2])
    bottom_right = np.minimum(arr_a[:, None, 2:], arr_b[None, :, 2:])

    intersection_wh = np.maximum(0.0, bottom_right - top_left)
    intersection = intersection_wh[:, :, 0] * intersection_wh[:, :, 1]

    area_a = (arr_a[:, 2] - arr_a[:, 0]) * (arr_a[:, 3] - arr_a[:, 1])
    area_b = (arr_b[:, 2] - arr_b[:, 0]) * (arr_b[:, 3] - arr_b[:, 1])

    union = area_a[:, None] + area_b[None, :] - intersection
    union = np.maximum(union, 1e-6)

    iou_matrix: NDArray[np.float32] = (intersection / union).astype(np.float32)
    return iou_matrix


def linear_assignment(
    cost_matrix: NDArray[np.float32],
    threshold: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Solve the linear sum assignment problem with threshold gating.

    Args:
        cost_matrix: Cost matrix of shape (N, M) where lower values represent better matches.
        threshold: Maximum allowed cost. Matches with cost > threshold are rejected.

    Returns:
        Tuple containing:
            - matches: List of (row_idx, col_idx) pairs.
            - unmatched_rows: List of unmatched row indices.
            - unmatched_cols: List of unmatched column indices.
    """
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matches: list[tuple[int, int]] = []
    unmatched_rows = list(range(cost_matrix.shape[0]))
    unmatched_cols = list(range(cost_matrix.shape[1]))

    for r, c in zip(row_indices, col_indices, strict=False):
        row_idx, col_idx = int(r), int(c)
        if cost_matrix[row_idx, col_idx] <= threshold:
            matches.append((row_idx, col_idx))
            if row_idx in unmatched_rows:
                unmatched_rows.remove(row_idx)
            if col_idx in unmatched_cols:
                unmatched_cols.remove(col_idx)

    return matches, unmatched_rows, unmatched_cols
