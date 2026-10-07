"""Unit tests for the SpatialHeatmap accumulator and overlay renderer."""

import numpy as np

from spatialtrack.analytics.heatmap import SpatialHeatmap
from spatialtrack.core.types import WorldCoord


def test_spatial_heatmap_accumulation_and_overlay() -> None:
    """Verify heatmap accumulates density and renders onto target canvas."""
    # Arrange
    heatmap = SpatialHeatmap(
        world_bounds=(0.0, 20.0, 0.0, 40.0),
        grid_resolution=(100, 100),
        sigma_cells=3.0,
    )
    base_canvas = np.zeros((300, 300, 3), dtype=np.uint8)

    # Act
    heatmap.accumulate([WorldCoord((10.0, 20.0))])  # Center coordinate
    overlay = heatmap.render_overlay(base_canvas, alpha=0.5)

    # Assert
    assert np.max(heatmap._grid) > 0.0
    assert overlay.shape == (300, 300, 3)
    # Overlay should have colorful pixels where Gaussian splat was added
    assert np.any(overlay > 0)


def test_spatial_heatmap_reset() -> None:
    """Verify that reset clears grid accumulation to all zeros."""
    # Arrange
    heatmap = SpatialHeatmap()
    heatmap.accumulate([WorldCoord((5.0, 5.0))])
    assert np.max(heatmap._grid) > 0.0

    # Act
    heatmap.reset()

    # Assert
    assert np.all(heatmap._grid == 0.0)
