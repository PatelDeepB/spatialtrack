"""Custom typed exception hierarchy for SpatialTrack."""


class SpatialTrackError(Exception):
    """Base exception for all SpatialTrack runtime errors."""


class ModelLoadError(SpatialTrackError):
    """Raised when the ONNX model cannot be loaded, found, or initialized."""


class VideoSourceError(SpatialTrackError):
    """Raised when a video stream, file, or webcam cannot be opened or decoded."""


class CalibrationError(SpatialTrackError):
    """Raised when camera calibration data is missing, corrupted, or geometrically invalid."""


class ConfigurationError(SpatialTrackError):
    """Raised when user configuration cannot be parsed or has invalid parameters."""


class InferenceError(SpatialTrackError):
    """Raised when the inference runtime encounters an unexpected evaluation error."""


class ExportError(SpatialTrackError):
    """Raised when output recording or telemetry export encounters an IO failure."""
