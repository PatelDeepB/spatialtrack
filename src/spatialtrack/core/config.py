"""Configuration models and loader for SpatialTrack using Pydantic."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from spatialtrack.core.exceptions import ConfigurationError


class DetectionConfig(BaseModel):
    """Detection model inference parameters."""

    model_path: Path = Path("models/yolov10n_int8.onnx")
    input_size: int = Field(default=640, ge=128, le=1920)
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    nms_iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    target_classes: list[str] = Field(
        default_factory=lambda: ["car", "truck", "bus", "motorcycle", "bicycle", "person"]
    )
    num_threads: int = Field(default=2, ge=1, le=32)


class TrackingConfig(BaseModel):
    """Tracking hyperparameters for state association and lifecycle."""

    high_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    low_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    max_age: int = Field(default=30, ge=1)
    min_hits: int = Field(default=3, ge=1)
    iou_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    max_trail_length: int = Field(default=50, ge=5)


class ProjectionConfig(BaseModel):
    """Perspective transformation and speed smoothing parameters."""

    calibration_path: Path | None = None
    speed_smoothing_alpha: float = Field(default=0.3, ge=0.0, le=1.0)
    min_displacement_m: float = Field(default=0.05, ge=0.0)


class AnalyticsConfig(BaseModel):
    """Event detection thresholds and spatial aggregation parameters."""

    speed_limit_kmh: float = Field(default=60.0, ge=0.0)
    speed_violation_debounce_frames: int = Field(default=5, ge=1)
    enable_heatmap: bool = True
    heatmap_resolution: tuple[int, int] = (200, 200)
    heatmap_sigma: float = Field(default=5.0, ge=0.1)
    zones_config_path: Path | None = None


class VisualizationConfig(BaseModel):
    """Display and rendering formatting parameters."""

    show_bboxes: bool = True
    show_trails: bool = True
    show_speed_labels: bool = True
    show_bev: bool = True
    trail_fade: bool = True
    bev_width: int = Field(default=400, ge=100)
    bev_height: int = Field(default=600, ge=100)
    font_scale: float = Field(default=0.6, ge=0.1, le=3.0)
    line_thickness: int = Field(default=2, ge=1, le=10)


class OutputConfig(BaseModel):
    """Export and recording destination settings."""

    output_video_path: Path | None = None
    event_log_path: Path | None = None
    telemetry_csv_path: Path | None = None
    video_codec: str = "mp4v"
    video_fps: float | None = None


class SpatialTrackConfig(BaseSettings):
    """Root configuration container supporting YAML loading and environment overrides."""

    model_config = {"env_prefix": "SPATIALTRACK_"}

    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    projection: ProjectionConfig = Field(default_factory=ProjectionConfig)
    analytics: AnalyticsConfig = Field(default_factory=AnalyticsConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    video_source: str = ""
    log_level: str = "INFO"

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "SpatialTrackConfig":
        """Load and parse configuration from a YAML file path.

        Args:
            config_path: Path to target YAML configuration file.

        Returns:
            Validated SpatialTrackConfig instance.

        Raises:
            ConfigurationError: If the file does not exist or content is invalid.
        """
        path = Path(config_path)
        if not path.is_file():
            raise ConfigurationError(f"Configuration file not found at: {path}")

        try:
            with open(path, encoding="utf-8") as stream:
                data: dict[str, Any] | None = yaml.safe_load(stream)
            if not isinstance(data, dict):
                raise ConfigurationError(f"Configuration file at {path} must contain a mapping.")
            return cls.model_validate(data)
        except Exception as err:
            if isinstance(err, ConfigurationError):
                raise
            raise ConfigurationError(f"Failed to parse configuration at {path}: {err}") from err
