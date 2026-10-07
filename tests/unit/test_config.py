"""Unit tests for configuration loading and validation."""

from pathlib import Path

import pytest

from spatialtrack.core.config import SpatialTrackConfig
from spatialtrack.core.exceptions import ConfigurationError


def test_default_config_instantiation() -> None:
    """Verify default configuration initializes with valid default parameters."""
    # Arrange & Act
    config = SpatialTrackConfig()

    # Assert
    assert config.detection.input_size == 640
    assert config.detection.confidence_threshold == 0.25
    assert config.tracking.max_age == 30
    assert config.projection.speed_smoothing_alpha == 0.3
    assert config.analytics.speed_limit_kmh == 60.0
    assert config.log_level == "INFO"


def test_yaml_config_loading(tmp_path: Path) -> None:
    """Verify configuration loads properly from a valid YAML file."""
    # Arrange
    yaml_content = """
detection:
  input_size: 512
  confidence_threshold: 0.35
analytics:
  speed_limit_kmh: 80.0
video_source: "test_highway.mp4"
"""
    yaml_file = tmp_path / "custom_config.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    # Act
    config = SpatialTrackConfig.from_yaml(yaml_file)

    # Assert
    assert config.detection.input_size == 512
    assert config.detection.confidence_threshold == 0.35
    assert config.analytics.speed_limit_kmh == 80.0
    assert config.video_source == "test_highway.mp4"


def test_yaml_missing_file_raises_error() -> None:
    """Verify that attempting to load a non-existent YAML file raises ConfigurationError."""
    # Arrange
    non_existent = Path("non_existent_path_xyz.yaml")

    # Act & Assert
    with pytest.raises(ConfigurationError):
        SpatialTrackConfig.from_yaml(non_existent)
