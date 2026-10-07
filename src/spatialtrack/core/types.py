"""Domain types and strict data models for SpatialTrack."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import NewType

TrackId = NewType("TrackId", int)
FrameIndex = NewType("FrameIndex", int)
MetersPerSecond = NewType("MetersPerSecond", float)
KilometersPerHour = NewType("KilometersPerHour", float)
Confidence = NewType("Confidence", float)
PixelCoord = NewType("PixelCoord", tuple[float, float])
WorldCoord = NewType("WorldCoord", tuple[float, float])


class ObjectClass(Enum):
    """Categorical classification of tracked dynamic objects."""

    CAR = auto()
    TRUCK = auto()
    BUS = auto()
    MOTORCYCLE = auto()
    BICYCLE = auto()
    PEDESTRIAN = auto()
    UNKNOWN = auto()

    @classmethod
    def from_label(cls, label: str) -> "ObjectClass":
        """Map standard class string names to ObjectClass enum."""
        mapping = {
            "car": cls.CAR,
            "truck": cls.TRUCK,
            "bus": cls.BUS,
            "motorcycle": cls.MOTORCYCLE,
            "bicycle": cls.BICYCLE,
            "person": cls.PEDESTRIAN,
            "pedestrian": cls.PEDESTRIAN,
        }
        return mapping.get(label.lower().strip(), cls.UNKNOWN)


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Axis-aligned bounding box coordinates (x1, y1, x2, y2) in pixel space."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def centroid(self) -> PixelCoord:
        """Calculate the center point of the bounding box."""
        center_x = (self.x1 + self.x2) / 2.0
        center_y = (self.y1 + self.y2) / 2.0
        return PixelCoord((center_x, center_y))

    @property
    def bottom_center(self) -> PixelCoord:
        """Calculate the bottom-center point, ideal for ground-plane contact estimation."""
        center_x = (self.x1 + self.x2) / 2.0
        return PixelCoord((center_x, self.y2))

    @property
    def area(self) -> float:
        """Calculate bounding box area in pixels squared."""
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    @property
    def width(self) -> float:
        """Calculate bounding box horizontal span."""
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        """Calculate bounding box vertical span."""
        return max(0.0, self.y2 - self.y1)

    @property
    def aspect_ratio(self) -> float:
        """Calculate width to height aspect ratio, returning zero if height is zero."""
        if self.height <= 0.0:
            return 0.0
        return self.width / self.height


@dataclass(frozen=True, slots=True)
class Detection:
    """Individual object detection emitted by a visual detector on a single frame."""

    bbox: BoundingBox
    confidence: Confidence
    object_class: ObjectClass
    frame_index: FrameIndex


@dataclass(slots=True)
class Track:
    """Temporal trajectory track maintaining persistent identity across frames."""

    track_id: TrackId
    bbox: BoundingBox
    object_class: ObjectClass
    confidence: Confidence
    age: int
    time_since_update: int
    pixel_trail: list[PixelCoord] = field(default_factory=list)
    velocity_px: tuple[float, float] = (0.0, 0.0)
    is_confirmed: bool = False


@dataclass(frozen=True, slots=True)
class SpatialRecord:
    """Ground-projected telemetry record for a tracked subject."""

    track_id: TrackId
    frame_index: FrameIndex
    pixel_position: PixelCoord
    world_position: WorldCoord
    speed_ms: MetersPerSecond
    speed_kmh: KilometersPerHour
    object_class: ObjectClass
    is_speed_violation: bool


class EventType(Enum):
    """Categorical event indicators emitted by the analytics rules engine."""

    SPEED_VIOLATION = "speed_violation"
    ZONE_ENTRY = "zone_entry"
    ZONE_EXIT = "zone_exit"
    ZONE_DWELL = "zone_dwell"
    LANE_CROSSING = "lane_crossing"


@dataclass(frozen=True, slots=True)
class SpatialEvent:
    """Discrete security or traffic event detected during spatial monitoring."""

    event_type: EventType
    track_id: TrackId
    frame_index: FrameIndex
    timestamp_sec: float
    world_position: WorldCoord
    metadata: dict[str, str | int | float | bool]


@dataclass(frozen=True, slots=True)
class CalibrationData:
    """Ground perspective calibration mapping four pixel coordinates to meters."""

    source_points_px: list[PixelCoord]
    target_points_m: list[WorldCoord]
    frame_width: int
    frame_height: int
    description: str = ""


@dataclass(frozen=True, slots=True)
class LatencyBreakdown:
    """Execution latency breakdown per pipeline stage in milliseconds."""

    decode_ms: float = 0.0
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0
    tracking_ms: float = 0.0
    projection_ms: float = 0.0
    analytics_ms: float = 0.0
    visualization_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        """Calculate aggregate pipeline duration."""
        return (
            self.decode_ms
            + self.preprocess_ms
            + self.inference_ms
            + self.postprocess_ms
            + self.tracking_ms
            + self.projection_ms
            + self.analytics_ms
            + self.visualization_ms
        )

    @property
    def estimated_fps(self) -> float:
        """Calculate throughput in frames per second based on aggregate latency."""
        if self.total_ms <= 0.0:
            return 0.0
        return 1000.0 / self.total_ms
