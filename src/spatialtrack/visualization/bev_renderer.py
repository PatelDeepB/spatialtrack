"""Bird's-Eye-View (BEV) orthographic metric map renderer."""

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from spatialtrack.analytics.heatmap import SpatialHeatmap
from spatialtrack.core.config import VisualizationConfig
from spatialtrack.core.types import SpatialRecord, WorldCoord
from spatialtrack.visualization.color_palette import get_color_for_id


class BevRenderer:
    """Renders top-down orthographic bird's-eye-view metric map of tracked subjects."""

    def __init__(
        self,
        config: VisualizationConfig,
        world_bounds: tuple[float, float, float, float] = (0.0, 15.0, 0.0, 40.0),
    ) -> None:
        """Initialize BEV renderer dimensions and metric coordinate bounds.

        Args:
            config: VisualizationConfig formatting parameters.
            world_bounds: Tuple of (min_x, max_x, min_y, max_y) in meters.
        """
        self.config = config
        self.width = config.bev_width
        self.height = config.bev_height
        self.min_x, self.max_x, self.min_y, self.max_y = world_bounds
        self.margin = 30  # Margin in canvas pixels

    def render(
        self,
        records: Sequence[SpatialRecord],
        trails_world: dict[int, list[WorldCoord]] | None = None,
        heatmap: SpatialHeatmap | None = None,
    ) -> NDArray[np.uint8]:
        """Render complete BEV canvas with road lanes, trails, and vehicle positions.

        Args:
            records: SpatialRecord objects for currently active tracks.
            trails_world: Optional dictionary mapping track_id to historical world coords.
            heatmap: Optional SpatialHeatmap to overlay onto metric canvas.

        Returns:
            Rendered BGR image canvas of shape (height, width, 3).
        """
        canvas: NDArray[np.uint8] = np.full(
            (self.height, self.width, 3), (35, 35, 35), dtype=np.uint8
        )
        self._draw_metric_grid(canvas)

        if heatmap is not None:
            canvas = heatmap.render_overlay(canvas, alpha=0.4)

        if trails_world:
            self._draw_all_world_trails(canvas, trails_world)

        for record in records:
            self._draw_vehicle_beacon(canvas, record)

        return canvas

    def world_to_canvas(self, world_pos: WorldCoord) -> tuple[int, int]:
        """Transform metric world coordinate (X, Y) to BEV canvas pixel (px, py)."""
        x_m, y_m = world_pos
        usable_w = self.width - 2 * self.margin
        usable_h = self.height - 2 * self.margin

        span_x = max(1.0, self.max_x - self.min_x)
        span_y = max(1.0, self.max_y - self.min_y)

        # In BEV, Y is longitudinal (pointing up), X is lateral (pointing right)
        norm_x = (x_m - self.min_x) / span_x
        norm_y = (y_m - self.min_y) / span_y

        canvas_x = int(round(self.margin + norm_x * usable_w))
        canvas_y = int(round((self.height - self.margin) - norm_y * usable_h))

        clamped_x = max(5, min(self.width - 5, canvas_x))
        clamped_y = max(5, min(self.height - 5, canvas_y))
        return clamped_x, clamped_y

    def _draw_metric_grid(self, canvas: NDArray[np.uint8]) -> None:
        """Draw subtle metric grid lines and border on canvas."""
        # Draw road boundary border
        top_left = (self.margin, self.margin)
        bottom_right = (self.width - self.margin, self.height - self.margin)
        cv2.rectangle(canvas, top_left, bottom_right, (70, 70, 70), 1)

        # Draw Title
        cv2.putText(
            canvas,
            "BIRD'S-EYE-VIEW (METRIC MAP)",
            (self.margin, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        # Draw distance intervals
        for meter_y in range(10, int(self.max_y), 10):
            _, py = self.world_to_canvas(WorldCoord((self.min_x, float(meter_y))))
            cv2.line(canvas, (self.margin, py), (self.width - self.margin, py), (55, 55, 55), 1)
            cv2.putText(
                canvas,
                f"{meter_y}m",
                (5, py + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (120, 120, 120),
                1,
                cv2.LINE_AA,
            )

    def _draw_all_world_trails(
        self,
        canvas: NDArray[np.uint8],
        trails_world: dict[int, list[WorldCoord]],
    ) -> None:
        """Draw historical trails on the BEV map."""
        for track_id, trail in trails_world.items():
            if len(trail) < 2:
                continue
            color = get_color_for_id(track_id)
            canvas_pts = [self.world_to_canvas(pt) for pt in trail]
            for i in range(1, len(canvas_pts)):
                cv2.line(canvas, canvas_pts[i - 1], canvas_pts[i], color, 2, cv2.LINE_AA)

    def _draw_vehicle_beacon(self, canvas: NDArray[np.uint8], record: SpatialRecord) -> None:
        """Render individual vehicle beacon and speed tag."""
        cx, cy = self.world_to_canvas(record.world_position)
        color = get_color_for_id(int(record.track_id))

        # Circle beacon
        cv2.circle(canvas, (cx, cy), 6, color, -1)
        cv2.circle(canvas, (cx, cy), 8, (255, 255, 255), 1)

        # Label tag
        speed_text = f"#{record.track_id}: {record.speed_kmh:.1f} km/h"
        text_color = (50, 50, 255) if record.is_speed_violation else (255, 255, 255)
        cv2.putText(
            canvas,
            speed_text,
            (cx + 10, cy + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            text_color,
            1,
            cv2.LINE_AA,
        )
