"""Dual-pane dashboard compositor joining camera feed, BEV map, and telemetry banner."""

import cv2
import numpy as np
from numpy.typing import NDArray

from spatialtrack.core.config import VisualizationConfig
from spatialtrack.core.types import LatencyBreakdown


class Dashboard:
    """Composites synchronized camera feed, BEV orthographic map, and telemetry strip."""

    def __init__(self, config: VisualizationConfig) -> None:
        """Initialize dashboard parameters."""
        self.config = config
        self._banner_height = 42

    def compose(
        self,
        camera_frame: NDArray[np.uint8],
        bev_frame: NDArray[np.uint8],
        active_tracks_count: int,
        violations_count: int,
        latency: LatencyBreakdown | None = None,
    ) -> NDArray[np.uint8]:
        """Combine camera and BEV views side-by-side with bottom telemetry banner.

        Args:
            camera_frame: Annotated camera frame (H, W_cam, 3).
            bev_frame: Rendered BEV map (H_bev, W_bev, 3).
            active_tracks_count: Current count of active tracked objects.
            violations_count: Count of current speed violations.
            latency: Optional LatencyBreakdown snapshot.

        Returns:
            Single synchronized composite image canvas.
        """
        cam_h, cam_w = camera_frame.shape[:2]

        # Resize BEV frame to match camera frame vertical resolution
        scaled_bev = cv2.resize(bev_frame, (self.config.bev_width, cam_h))

        # Horizontal stack
        side_by_side = np.hstack([camera_frame, scaled_bev])
        total_w = side_by_side.shape[1]

        # Create bottom telemetry strip
        banner = np.full((self._banner_height, total_w, 3), (25, 25, 25), dtype=np.uint8)
        self._draw_telemetry_banner(banner, total_w, active_tracks_count, violations_count, latency)

        return np.vstack([side_by_side, banner])

    def _draw_telemetry_banner(
        self,
        banner: NDArray[np.uint8],
        width: int,
        tracks_count: int,
        violations_count: int,
        latency: LatencyBreakdown | None,
    ) -> None:
        """Render informative metrics on bottom banner."""
        fps_text = f"FPS: {latency.estimated_fps:.1f}" if latency else "FPS: --"
        lat_text = f"Latency: {latency.total_ms:.1f}ms" if latency else ""
        telemetry = (
            f"SPATIALTRACK (CPU)  |  {fps_text}  |  "
            f"Tracks: {tracks_count}  |  Violations: {violations_count}  |  {lat_text}"
        )

        cv2.putText(
            banner,
            telemetry,
            (20, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
