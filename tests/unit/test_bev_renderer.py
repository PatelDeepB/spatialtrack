"""Unit tests for the BevRenderer top-down map visualization."""

import numpy as np

from spatialtrack.core.config import VisualizationConfig
from spatialtrack.core.types import (
    FrameIndex,
    KilometersPerHour,
    MetersPerSecond,
    ObjectClass,
    PixelCoord,
    SpatialRecord,
    TrackId,
    WorldCoord,
)
from spatialtrack.visualization.bev_renderer import BevRenderer


def test_bev_renderer_renders_valid_canvas() -> None:
    """Verify BEV renderer generates canvas with correct dimensions and drawn elements."""
    # Arrange
    config = VisualizationConfig(bev_width=300, bev_height=500)
    renderer = BevRenderer(config, world_bounds=(0.0, 10.0, 0.0, 30.0))

    record = SpatialRecord(
        track_id=TrackId(1),
        frame_index=FrameIndex(1),
        pixel_position=PixelCoord((100.0, 200.0)),
        world_position=WorldCoord((5.0, 15.0)),
        speed_ms=MetersPerSecond(12.0),
        speed_kmh=KilometersPerHour(43.2),
        object_class=ObjectClass.CAR,
        is_speed_violation=False,
    )

    # Act
    canvas = renderer.render([record])

    # Assert
    assert canvas.shape == (500, 300, 3)
    assert canvas.dtype == np.uint8
    # Canvas should not be empty
    assert np.any(canvas > 40)
