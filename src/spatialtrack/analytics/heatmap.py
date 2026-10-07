"""Spatial density heatmap accumulator using Gaussian kernel estimation."""

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from spatialtrack.core.types import WorldCoord


class SpatialHeatmap:
    """Accumulates vehicle traffic density in metric world coordinates as a 2D Gaussian heatmap."""

    def __init__(
        self,
        world_bounds: tuple[float, float, float, float] = (0.0, 15.0, 0.0, 40.0),
        grid_resolution: tuple[int, int] = (200, 200),
        sigma_cells: float = 4.0,
    ) -> None:
        """Initialize heatmap accumulation grid and metric boundaries.

        Args:
            world_bounds: Tuple of (min_x, max_x, min_y, max_y) in meters.
            grid_resolution: Tuple of (grid_height, grid_width) cells.
            sigma_cells: Gaussian kernel standard deviation in grid cells.
        """
        self.min_x, self.max_x, self.min_y, self.max_y = world_bounds
        self.grid_h, self.grid_w = grid_resolution
        self.sigma = sigma_cells
        self._grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)

    def accumulate(self, world_positions: Sequence[WorldCoord]) -> None:
        """Add Gaussian splats centered at current world positions.

        Args:
            world_positions: Sequence of metric world positions to accumulate.
        """
        span_x = max(1.0, self.max_x - self.min_x)
        span_y = max(1.0, self.max_y - self.min_y)

        for pos in world_positions:
            norm_x = (pos[0] - self.min_x) / span_x
            norm_y = (pos[1] - self.min_y) / span_y

            if 0.0 <= norm_x <= 1.0 and 0.0 <= norm_y <= 1.0:
                gx = int(round(norm_x * (self.grid_w - 1)))
                gy = int(round((1.0 - norm_y) * (self.grid_h - 1)))
                self._add_gaussian(gx, gy)

    def _add_gaussian(self, center_x: int, center_y: int) -> None:
        """Add a localized Gaussian distribution centered at (center_x, center_y)."""
        radius = int(math_ceil(self.sigma * 3.0))
        y_min = max(0, center_y - radius)
        y_max = min(self.grid_h, center_y + radius + 1)
        x_min = max(0, center_x - radius)
        x_max = min(self.grid_w, center_x + radius + 1)

        y_coords, x_coords = np.ogrid[y_min:y_max, x_min:x_max]
        dist_sq = (x_coords - center_x) ** 2 + (y_coords - center_y) ** 2
        kernel = np.exp(-dist_sq / (2.0 * self.sigma**2))
        self._grid[y_min:y_max, x_min:x_max] += kernel

    def render_overlay(
        self,
        base_canvas: NDArray[np.uint8],
        alpha: float = 0.5,
    ) -> NDArray[np.uint8]:
        """Apply color-mapped heatmap overlay onto a target image canvas.

        Args:
            base_canvas: Image canvas (H, W, 3) to overlay onto.
            alpha: Blend weight of the colored heatmap.

        Returns:
            Blended BGR image canvas.
        """
        max_val = float(np.max(self._grid))
        if max_val <= 0.0:
            return base_canvas.copy()

        # Normalize to [0, 255]
        normalized = (self._grid / max_val * 255.0).astype(np.uint8)
        resized = cv2.resize(normalized, (base_canvas.shape[1], base_canvas.shape[0]))
        colored_map = cv2.applyColorMap(resized, cv2.COLORMAP_JET)

        # Mask out cold zero regions to keep background clear
        mask = resized > 10
        output = base_canvas.copy()
        output[mask] = cv2.addWeighted(
            base_canvas[mask],
            1.0 - alpha,
            colored_map[mask],
            alpha,
            0,
        )
        return output

    def reset(self) -> None:
        """Clear density accumulation grid."""
        self._grid.fill(0.0)


def math_ceil(val: float) -> int:
    """Helper for integer ceiling."""
    return int(np.ceil(val))
