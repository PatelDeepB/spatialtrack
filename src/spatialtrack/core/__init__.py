"""Core module containing domain types, configuration, exceptions, and constants."""

from spatialtrack.core.config import SpatialTrackConfig
from spatialtrack.core.constants import COCO_CLASSES, DEFAULT_INPUT_SIZE
from spatialtrack.core.exceptions import (
    CalibrationError,
    ConfigurationError,
    InferenceError,
    ModelLoadError,
    SpatialTrackError,
    VideoSourceError,
)
from spatialtrack.core.types import (
    BoundingBox,
    CalibrationData,
    Confidence,
    Detection,
    EventType,
    FrameIndex,
    KilometersPerHour,
    LatencyBreakdown,
    MetersPerSecond,
    ObjectClass,
    PixelCoord,
    SpatialEvent,
    SpatialRecord,
    Track,
    TrackId,
    WorldCoord,
)

__all__ = [
    "BoundingBox",
    "COCO_CLASSES",
    "CalibrationData",
    "CalibrationError",
    "Confidence",
    "ConfigurationError",
    "DEFAULT_INPUT_SIZE",
    "Detection",
    "EventType",
    "FrameIndex",
    "InferenceError",
    "KilometersPerHour",
    "LatencyBreakdown",
    "MetersPerSecond",
    "ModelLoadError",
    "ObjectClass",
    "PixelCoord",
    "SpatialEvent",
    "SpatialRecord",
    "SpatialTrackConfig",
    "SpatialTrackError",
    "Track",
    "TrackId",
    "VideoSourceError",
    "WorldCoord",
]
