"""Unit tests for SpatialTrackEngine."""

from pathlib import Path

import numpy as np
import pytest

from spatialtrack.core.config import (
    DetectionConfig,
    ProjectionConfig,
    SpatialTrackConfig,
    TrackingConfig,
    VisualizationConfig,
)
from spatialtrack.core.exceptions import VideoSourceError
from spatialtrack.core.types import FrameIndex
from spatialtrack.pipeline.engine import SpatialTrackEngine


def test_engine_init_without_calibration() -> None:
    """Verify engine initializes with mapper/bev/heatmap as None when no calibration provided."""
    # Arrange & Act
    config = SpatialTrackConfig(
        detection=DetectionConfig(model_path=Path("models/yolov10n_int8.onnx")),
        projection=ProjectionConfig(calibration_path=None),
    )
    engine = SpatialTrackEngine(config)

    # Assert
    assert engine.mapper is None
    assert engine.bev_renderer is None
    assert engine.heatmap is None


def test_engine_process_frame_uncalibrated() -> None:
    """Verify process_frame runs detection and tracking without throwing when uncalibrated."""
    # Arrange
    config = SpatialTrackConfig(
        detection=DetectionConfig(model_path=Path("models/yolov10n_int8.onnx")),
        tracking=TrackingConfig(),
        projection=ProjectionConfig(calibration_path=None),
        visualization=VisualizationConfig(),
    )
    engine = SpatialTrackEngine(config)
    synthetic_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Act
    res = engine.process_frame(synthetic_frame, FrameIndex(0))

    # Assert
    assert res.frame_index == 0
    assert res.annotated_frame.shape == (480, 640, 3)
    assert len(res.spatial_records) == 0
    assert len(res.events) == 0


def test_engine_run_empty_source_raises() -> None:
    """Verify engine.run raises VideoSourceError when source is empty."""
    # Arrange
    config = SpatialTrackConfig(video_source="")
    engine = SpatialTrackEngine(config)

    # Act & Assert
    with pytest.raises(VideoSourceError, match="No video source provided"):
        engine.run()
