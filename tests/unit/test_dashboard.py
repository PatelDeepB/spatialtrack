"""Unit tests for the dual-pane Dashboard compositor."""

import numpy as np

from spatialtrack.core.config import VisualizationConfig
from spatialtrack.core.types import LatencyBreakdown
from spatialtrack.visualization.dashboard import Dashboard


def test_dashboard_compositor_dimensions() -> None:
    """Verify that Dashboard horizontally stacks camera and BEV and appends telemetry banner."""
    # Arrange
    config = VisualizationConfig(bev_width=400)
    dashboard = Dashboard(config)

    camera_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    bev_frame = np.zeros((600, 400, 3), dtype=np.uint8)
    latency = LatencyBreakdown(inference_ms=100.0, tracking_ms=1.0)

    # Act
    composite = dashboard.compose(
        camera_frame=camera_frame,
        bev_frame=bev_frame,
        active_tracks_count=5,
        violations_count=1,
        latency=latency,
    )

    # Assert
    # Width = 640 + 400 = 1040
    # Height = 480 + 42 (banner) = 522
    assert composite.shape == (522, 1040, 3)
    assert composite.dtype == np.uint8
