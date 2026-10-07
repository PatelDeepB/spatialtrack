# SpatialTrack: CPU-Optimized Spatial Telemetry & Speed Estimation Engine

## Complete Implementation Plan

---

## 1. Project Vision & Identity

**Project Name:** `SpatialTrack`
**Tagline:** *Real-time spatial telemetry, speed estimation, and trajectory analytics on CPU.*

**What SpatialTrack Does:**
SpatialTrack is a production-grade computer vision engine that ingests traffic or surveillance video, detects and tracks vehicles/pedestrians at 30+ FPS on a standard CPU, projects their pixel trajectories onto a calibrated metric bird's-eye-view map, and computes real-world speed, lane violations, and spatial heatmaps in real time.

**What Makes It Exceptional:**
- Zero GPU dependency: INT8-quantized ONNX inference at 30+ FPS on a dual-core CPU
- True projective geometry: Homography-based pixel-to-meter coordinate transformation
- Temporal state estimation: Kalman-filtered tracking with Hungarian algorithm association
- Production architecture: Clean separation of detection, tracking, projection, and analytics layers
- Interactive dual-view demo: Raw annotated feed + synchronized bird's-eye-view map

---

## 2. Technical Architecture

```
                        SpatialTrack Architecture
 ============================================================================

  +---------------------+
  |   Video Source       |  (File, RTSP URL, or Webcam)
  +----------+----------+
             |
             v
  +----------+----------+
  |   Frame Decoder      |  OpenCV VideoCapture + frame queue
  |   (Async Producer)   |  Decoupled from processing via threading
  +----------+----------+
             |
             v
  +----------+----------+
  |   Detector Module    |  YOLOv10n / NanoDet-Plus (INT8 ONNX)
  |   (ONNX Runtime CPU) |  Pre-process -> Infer -> Post-process (NMS)
  +----------+----------+
             |
             | List[Detection(bbox, class_id, confidence)]
             v
  +----------+----------+
  |   Tracker Module     |  ByteTrack / Norfair
  |   (Kalman + Hungarian)|  Associates detections across frames
  +----------+----------+  Maintains track_id, velocity, age
             |
             | List[Track(track_id, bbox, velocity, trail)]
             v
  +----------+----------+
  |   Projection Module  |  Homography Matrix (cv2.getPerspectiveTransform)
  |   (Pixel -> Metric)  |  Projects bbox centroids to BEV coordinates
  +----------+----------+  Converts px/frame displacement to m/s -> km/h
             |
             | List[SpatialRecord(track_id, world_xy, speed_kmh, trail_world)]
             v
  +----------+----------+
  |   Analytics Engine   |  Speed violation detection
  |   (Event Rules)      |  Lane boundary crossing detection
  +----------+----------+  Zone dwell-time aggregation
             |                Heatmap accumulation
             v
  +----------+----------+
  |   Visualizer         |  Dual-pane renderer:
  |   (OpenCV + Overlay) |    Left:  Annotated camera feed
  +----------+----------+    Right: Bird's-eye-view metric map
             |
             v
  +----------+----------+
  |   Output Sink        |  Display window / Video file / WebSocket stream
  +---------------------+  JSON event log / CSV telemetry export
```

### Data Flow Summary

```
Frame (HxWx3 uint8)
  -> Letterbox + Normalize (1x3x640x640 float32)
    -> ONNX Inference (raw tensor output)
      -> NMS + Filter (List[Detection])
        -> ByteTrack Association (List[Track])
          -> Homography Project (List[SpatialRecord])
            -> Analytics Rules (List[SpatialEvent])
              -> Visualize + Export
```

---

## 3. Repository Structure

```
spatialtrack/
|
|-- .github/
|   |-- workflows/
|   |   |-- ci.yml                    # Lint + Test + Type-check on every PR
|   |   |-- release.yml               # Auto-publish to PyPI on tag push
|   |-- ISSUE_TEMPLATE/
|   |   |-- bug_report.md
|   |   |-- feature_request.md
|   |-- PULL_REQUEST_TEMPLATE.md
|
|-- docs/
|   |-- getting_started.md            # Quick-start guide with screenshots
|   |-- calibration_guide.md          # Step-by-step homography calibration
|   |-- architecture.md               # System design deep-dive
|   |-- api_reference.md              # Public API documentation
|   |-- deployment.md                 # Hugging Face Spaces / Docker deploy guide
|   |-- benchmarks.md                 # CPU latency benchmarks across hardware
|   |-- assets/
|       |-- architecture_diagram.png
|       |-- demo_screenshot.png
|       |-- calibration_example.png
|
|-- src/
|   |-- spatialtrack/
|   |   |-- __init__.py               # Package version + public API exports
|   |   |-- __main__.py               # CLI entry point (python -m spatialtrack)
|   |   |
|   |   |-- core/
|   |   |   |-- __init__.py
|   |   |   |-- types.py              # Domain types: Detection, Track, SpatialRecord
|   |   |   |-- config.py             # Pydantic settings: thresholds, paths, model config
|   |   |   |-- constants.py          # UPPER_SNAKE_CASE constants
|   |   |   |-- exceptions.py         # Custom typed exceptions
|   |   |
|   |   |-- detection/
|   |   |   |-- __init__.py
|   |   |   |-- base_detector.py      # Abstract base class (Protocol)
|   |   |   |-- onnx_detector.py      # ONNX Runtime INT8 YOLOv10n detector
|   |   |   |-- preprocessing.py      # Letterbox resize, normalization, CHW transpose
|   |   |   |-- postprocessing.py     # NMS, confidence filtering, bbox scaling
|   |   |
|   |   |-- tracking/
|   |   |   |-- __init__.py
|   |   |   |-- base_tracker.py       # Abstract base class (Protocol)
|   |   |   |-- bytetrack.py          # ByteTrack implementation (Kalman + Hungarian)
|   |   |   |-- kalman_filter.py      # Custom 2D Kalman filter for bbox state
|   |   |   |-- association.py        # IoU + cost matrix + Hungarian solver
|   |   |
|   |   |-- projection/
|   |   |   |-- __init__.py
|   |   |   |-- homography.py         # Compute and apply perspective transform
|   |   |   |-- calibration.py        # Interactive calibration tool (pick 4 points)
|   |   |   |-- coordinate_mapper.py  # Pixel-to-world and world-to-pixel transforms
|   |   |
|   |   |-- analytics/
|   |   |   |-- __init__.py
|   |   |   |-- speed_estimator.py    # Displacement/time -> m/s -> km/h
|   |   |   |-- zone_monitor.py       # Polygon zone entry/exit/dwell detection
|   |   |   |-- event_detector.py     # Speed violations, lane crossings
|   |   |   |-- heatmap.py            # Gaussian-weighted spatial heatmap accumulator
|   |   |
|   |   |-- visualization/
|   |   |   |-- __init__.py
|   |   |   |-- annotator.py          # Draw bboxes, trails, speed labels on camera feed
|   |   |   |-- bev_renderer.py       # Render bird's-eye-view map with projected tracks
|   |   |   |-- dashboard.py          # Compose dual-pane view + telemetry panel
|   |   |   |-- color_palette.py      # Deterministic color assignment per track_id
|   |   |
|   |   |-- io/
|   |   |   |-- __init__.py
|   |   |   |-- video_reader.py       # Async frame producer (threaded VideoCapture)
|   |   |   |-- video_writer.py       # Output video sink with codec selection
|   |   |   |-- event_logger.py       # Structured JSON event log writer
|   |   |   |-- telemetry_exporter.py # CSV/JSON telemetry export
|   |   |
|   |   |-- pipeline/
|   |   |   |-- __init__.py
|   |   |   |-- engine.py             # Main processing loop: orchestrates all modules
|   |   |   |-- profiler.py           # Per-stage latency measurement and reporting
|   |   |
|   |   |-- cli/
|   |       |-- __init__.py
|   |       |-- main.py               # Typer CLI: run, calibrate, benchmark, export
|   |       |-- validators.py         # CLI argument validation helpers
|   |
|-- app/
|   |-- streamlit_app.py              # Streamlit demo UI for Hugging Face Spaces
|   |-- components/
|   |   |-- video_uploader.py         # File upload + sample video selector
|   |   |-- config_sidebar.py         # Interactive threshold sliders
|   |   |-- results_display.py        # Dual-pane video + metrics display
|   |-- assets/
|       |-- sample_traffic.mp4        # Short demo clip (< 25 MB)
|       |-- sample_calibration.json   # Pre-configured calibration for demo clip
|
|-- tests/
|   |-- __init__.py
|   |-- conftest.py                   # Shared fixtures: sample frames, mock detections
|   |-- unit/
|   |   |-- test_preprocessing.py
|   |   |-- test_postprocessing.py
|   |   |-- test_kalman_filter.py
|   |   |-- test_association.py
|   |   |-- test_homography.py
|   |   |-- test_speed_estimator.py
|   |   |-- test_zone_monitor.py
|   |   |-- test_heatmap.py
|   |   |-- test_config.py
|   |-- integration/
|   |   |-- test_detection_pipeline.py
|   |   |-- test_tracking_pipeline.py
|   |   |-- test_full_pipeline.py
|   |-- fixtures/
|       |-- sample_frame_720p.jpg     # Test image fixture
|       |-- sample_detections.json    # Serialized detection fixtures
|       |-- sample_calibration.json   # Test calibration data
|
|-- scripts/
|   |-- download_model.py             # Download + quantize ONNX model
|   |-- benchmark_cpu.py              # Run standardized CPU latency benchmark
|   |-- export_quantized_model.py     # PyTorch -> ONNX -> INT8 quantization script
|
|-- configs/
|   |-- default.yaml                  # Default configuration (thresholds, model path, etc.)
|   |-- traffic_highway.yaml          # Preset for highway traffic monitoring
|   |-- traffic_intersection.yaml     # Preset for intersection monitoring
|   |-- pedestrian_indoor.yaml        # Preset for indoor pedestrian tracking
|
|-- models/                           # .gitignore'd; models downloaded at runtime
|   |-- .gitkeep
|
|-- .gitignore
|-- .pre-commit-config.yaml           # Ruff + MyPy + Pytest pre-commit hooks
|-- pyproject.toml                    # Modern Python packaging (PEP 621)
|-- Dockerfile                        # Multi-stage build for deployment
|-- Makefile                          # Common dev commands: lint, test, run, bench
|-- LICENSE                           # MIT License
|-- README.md                         # Comprehensive project README
|-- CONTRIBUTING.md                   # Contribution guidelines
|-- CHANGELOG.md                      # Keep-a-changelog format
```

---

## 4. Domain Types (The Foundation)

> **IMPORTANT:** Defining strict domain types before writing any logic is what separates
> production code from script-level prototypes. Every module communicates through these
> typed contracts.

```python
# src/spatialtrack/core/types.py

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import NewType

import numpy as np
from numpy.typing import NDArray


# -- Domain Primitives (not raw floats/ints) --
TrackId = NewType("TrackId", int)
FrameIndex = NewType("FrameIndex", int)
MetersPerSecond = NewType("MetersPerSecond", float)
KilometersPerHour = NewType("KilometersPerHour", float)
Confidence = NewType("Confidence", float)
PixelCoord = NewType("PixelCoord", tuple[float, float])      # (u, v) in pixels
WorldCoord = NewType("WorldCoord", tuple[float, float])       # (X, Y) in meters


class ObjectClass(Enum):
    """Detected object categories relevant to traffic telemetry."""
    CAR = auto()
    TRUCK = auto()
    BUS = auto()
    MOTORCYCLE = auto()
    BICYCLE = auto()
    PEDESTRIAN = auto()
    UNKNOWN = auto()


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Axis-aligned bounding box in pixel coordinates (x1, y1, x2, y2)."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def centroid(self) -> PixelCoord:
        """Calculate the center point of the bounding box."""
        return PixelCoord(((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2))

    @property
    def area(self) -> float:
        """Calculate the area of the bounding box in square pixels."""
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    @property
    def width(self) -> float:
        """Calculate the width of the bounding box in pixels."""
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        """Calculate the height of the bounding box in pixels."""
        return max(0.0, self.y2 - self.y1)

    @property
    def aspect_ratio(self) -> float:
        """Calculate width/height aspect ratio. Returns 0.0 if height is zero."""
        if self.height == 0.0:
            return 0.0
        return self.width / self.height


@dataclass(frozen=True, slots=True)
class Detection:
    """A single object detection result from one frame."""
    bbox: BoundingBox
    confidence: Confidence
    object_class: ObjectClass
    frame_index: FrameIndex


@dataclass(slots=True)
class Track:
    """A tracked object with persistent identity across frames."""
    track_id: TrackId
    bbox: BoundingBox
    object_class: ObjectClass
    confidence: Confidence
    age: int                                        # frames since first detection
    time_since_update: int                          # frames since last matched detection
    pixel_trail: list[PixelCoord] = field(default_factory=list)
    velocity_px: tuple[float, float] = (0.0, 0.0)  # pixels/frame
    is_confirmed: bool = False                      # True after N consecutive matches


@dataclass(frozen=True, slots=True)
class SpatialRecord:
    """A track projected into calibrated world coordinates with speed."""
    track_id: TrackId
    frame_index: FrameIndex
    pixel_position: PixelCoord
    world_position: WorldCoord
    speed_ms: MetersPerSecond
    speed_kmh: KilometersPerHour
    object_class: ObjectClass
    is_speed_violation: bool


class EventType(Enum):
    """Categories of spatial events detected by the analytics engine."""
    SPEED_VIOLATION = "speed_violation"
    ZONE_ENTRY = "zone_entry"
    ZONE_EXIT = "zone_exit"
    ZONE_DWELL = "zone_dwell"
    LANE_CROSSING = "lane_crossing"


@dataclass(frozen=True, slots=True)
class SpatialEvent:
    """A discrete event detected by the analytics engine."""
    event_type: EventType
    track_id: TrackId
    frame_index: FrameIndex
    timestamp_sec: float
    world_position: WorldCoord
    metadata: dict[str, str | int | float | bool]


@dataclass(frozen=True, slots=True)
class CalibrationData:
    """Stores the 4-point correspondence for homography calibration."""
    source_points_px: list[PixelCoord]    # 4 points in pixel space
    target_points_m: list[WorldCoord]     # 4 corresponding points in meters
    frame_width: int
    frame_height: int
    description: str = ""                 # e.g., "Highway cam A, facing north"


@dataclass(frozen=True, slots=True)
class LatencyBreakdown:
    """Per-frame timing for each processing stage in milliseconds."""
    decode_ms: float
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    tracking_ms: float
    projection_ms: float
    analytics_ms: float
    visualization_ms: float

    @property
    def total_ms(self) -> float:
        """Calculate total processing time across all stages."""
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
        """Calculate estimated FPS from total latency."""
        if self.total_ms <= 0:
            return 0.0
        return 1000.0 / self.total_ms
```

---

## 5. Configuration System

```python
# src/spatialtrack/core/config.py

from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class DetectionConfig(BaseModel):
    """Detection model and inference settings."""
    model_path: Path = Path("models/yolov10n_int8.onnx")
    input_size: int = 640
    confidence_threshold: float = Field(0.25, ge=0.0, le=1.0)
    nms_iou_threshold: float = Field(0.45, ge=0.0, le=1.0)
    target_classes: list[str] = ["car", "truck", "bus", "motorcycle", "bicycle", "person"]
    num_threads: int = Field(2, ge=1, le=16)


class TrackingConfig(BaseModel):
    """ByteTrack tracker settings."""
    high_threshold: float = Field(0.6, ge=0.0, le=1.0)
    low_threshold: float = Field(0.1, ge=0.0, le=1.0)
    max_age: int = Field(30, ge=1)           # frames before track deletion
    min_hits: int = Field(3, ge=1)           # consecutive detections to confirm
    iou_threshold: float = Field(0.3, ge=0.0, le=1.0)
    max_trail_length: int = Field(50, ge=10)


class ProjectionConfig(BaseModel):
    """Homography projection settings."""
    calibration_path: Path | None = None
    speed_smoothing_alpha: float = Field(0.3, ge=0.0, le=1.0)
    min_displacement_m: float = Field(0.05, ge=0.0)  # ignore micro-jitter


class AnalyticsConfig(BaseModel):
    """Analytics and event detection settings."""
    speed_limit_kmh: float = Field(60.0, ge=0.0)
    speed_violation_debounce_frames: int = Field(5, ge=1)
    enable_heatmap: bool = True
    heatmap_resolution: tuple[int, int] = (200, 200)
    heatmap_sigma: float = Field(5.0, ge=0.1)
    zones_config_path: Path | None = None


class VisualizationConfig(BaseModel):
    """Rendering and display settings."""
    show_bboxes: bool = True
    show_trails: bool = True
    show_speed_labels: bool = True
    show_bev: bool = True
    trail_fade: bool = True
    bev_width: int = Field(400, ge=100)
    bev_height: int = Field(600, ge=100)
    font_scale: float = Field(0.6, ge=0.1, le=3.0)
    line_thickness: int = Field(2, ge=1, le=5)


class OutputConfig(BaseModel):
    """Output file and export settings."""
    output_video_path: Path | None = None
    event_log_path: Path | None = None
    telemetry_csv_path: Path | None = None
    video_codec: str = "mp4v"
    video_fps: float | None = None  # None = match source FPS


class SpatialTrackConfig(BaseSettings):
    """
    Root configuration for SpatialTrack.

    Loaded from (in priority order):
    1. Environment variables (prefixed SPATIALTRACK_)
    2. YAML config file
    3. Default values
    """
    model_config = {"env_prefix": "SPATIALTRACK_"}

    detection: DetectionConfig = DetectionConfig()
    tracking: TrackingConfig = TrackingConfig()
    projection: ProjectionConfig = ProjectionConfig()
    analytics: AnalyticsConfig = AnalyticsConfig()
    visualization: VisualizationConfig = VisualizationConfig()
    output: OutputConfig = OutputConfig()

    video_source: str = ""  # File path, RTSP URL, or camera index
    log_level: str = "INFO"
```

---

## 6. Custom Exception Hierarchy

```python
# src/spatialtrack/core/exceptions.py


class SpatialTrackError(Exception):
    """Base exception for all SpatialTrack errors."""


class ModelLoadError(SpatialTrackError):
    """Raised when the ONNX model cannot be loaded or is invalid."""


class VideoSourceError(SpatialTrackError):
    """Raised when the video source cannot be opened or is unreachable."""


class CalibrationError(SpatialTrackError):
    """Raised when calibration data is missing, invalid, or geometrically degenerate."""


class ConfigurationError(SpatialTrackError):
    """Raised when the configuration file is invalid or missing required fields."""


class InferenceError(SpatialTrackError):
    """Raised when ONNX Runtime inference fails unexpectedly."""


class ExportError(SpatialTrackError):
    """Raised when writing output files (video, CSV, JSON) fails."""
```

---

## 7. Implementation Phases

### Phase 1: Foundation and Detection Pipeline
**Goal:** Build the core infrastructure and get INT8 ONNX detection running at target FPS on CPU.

| Task | Details | Key Files |
|:-----|:--------|:----------|
| 1.1 Project scaffold | `pyproject.toml`, directory structure, `.gitignore`, `Makefile`, pre-commit config | Root config files |
| 1.2 Domain types | All dataclasses, enums, and NewType definitions from Section 4 | `core/types.py` |
| 1.3 Configuration system | Pydantic `BaseSettings` with YAML file loading and env var overrides | `core/config.py` |
| 1.4 Custom exceptions | Typed error hierarchy: `CalibrationError`, `ModelLoadError`, `VideoSourceError` | `core/exceptions.py` |
| 1.5 Model download script | Download YOLOv10n ONNX from official repo, verify SHA256 hash | `scripts/download_model.py` |
| 1.6 INT8 quantization script | ONNX Runtime static quantization with calibration dataset | `scripts/export_quantized_model.py` |
| 1.7 Preprocessing module | Letterbox resize preserving aspect ratio, normalize to [0,1], HWC->CHW transpose | `detection/preprocessing.py` |
| 1.8 ONNX detector | Load INT8 model, run inference, return raw outputs | `detection/onnx_detector.py` |
| 1.9 Postprocessing module | Confidence filtering, class filtering, NMS, rescale bboxes to original frame | `detection/postprocessing.py` |
| 1.10 Async video reader | Threaded frame producer with configurable queue depth and FPS throttling | `io/video_reader.py` |
| 1.11 Unit tests | Test preprocessing output shapes, NMS correctness, bbox scaling math | `tests/unit/test_preprocessing.py`, `test_postprocessing.py` |

#### 1.7 Preprocessing: Detailed Algorithm

```
Input:  BGR frame (H, W, 3) uint8
Step 1: Calculate letterbox dimensions
        - ratio = min(target_size/H, target_size/W)
        - new_h, new_w = int(H * ratio), int(W * ratio)
Step 2: Resize to (new_h, new_w) using cv2.INTER_LINEAR
Step 3: Create canvas (target_size, target_size, 3) filled with (114, 114, 114)
Step 4: Paste resized image centered on canvas
Step 5: BGR -> RGB color conversion
Step 6: Normalize pixel values: float32 / 255.0
Step 7: Transpose HWC -> CHW: (3, target_size, target_size)
Step 8: Add batch dimension: (1, 3, target_size, target_size)
Output: float32 tensor (1, 3, 640, 640), padding offsets (dx, dy), scale ratio
```

#### 1.9 Postprocessing: Detailed Algorithm

```
Input:  Raw model output tensor, padding offsets, scale ratio
Step 1: Parse output tensor into (N, 6) array: [x1, y1, x2, y2, conf, class_id]
Step 2: Filter by confidence_threshold
Step 3: Filter by target_classes (keep only vehicles/pedestrians)
Step 4: Apply Non-Maximum Suppression (NMS):
        - Sort by confidence descending
        - For each box, suppress all lower-conf boxes with IoU > nms_threshold
Step 5: Rescale coordinates:
        - Remove letterbox padding offset (subtract dx, dy)
        - Divide by scale ratio to get original frame coordinates
Step 6: Clamp coordinates to frame boundaries
Output: List[Detection] with original-frame-space bounding boxes
```

**Phase 1 Milestone:** Run `python -m spatialtrack detect --source sample.mp4` and see bounding boxes drawn on each frame at 30+ FPS with per-frame latency printed to console.

---

### Phase 2: Multi-Object Tracking
**Goal:** Implement ByteTrack with Kalman filtering and Hungarian association to maintain persistent track IDs across frames.

| Task | Details | Key Files |
|:-----|:--------|:----------|
| 2.1 Kalman filter | 2D constant-velocity Kalman filter for bbox state `[cx, cy, ar, h, vcx, vcy, var, vh]` | `tracking/kalman_filter.py` |
| 2.2 IoU association | Vectorized IoU computation between N predicted and M detected bboxes | `tracking/association.py` |
| 2.3 Hungarian solver | `scipy.optimize.linear_sum_assignment` wrapper with cost matrix construction | `tracking/association.py` |
| 2.4 ByteTrack implementation | Two-stage association (high-conf first, then low-conf), track lifecycle management | `tracking/bytetrack.py` |
| 2.5 Trail accumulation | Append centroid to per-track `pixel_trail` with configurable max trail length | `tracking/bytetrack.py` |
| 2.6 Track annotator | Draw colored bounding boxes, track ID labels, and fading trajectory trails | `visualization/annotator.py` |
| 2.7 Color palette | Deterministic hash-based color assignment so each `track_id` gets a stable color | `visualization/color_palette.py` |
| 2.8 Unit tests | Test Kalman predict/update cycle, IoU edge cases, Hungarian on known cost matrices | `tests/unit/test_kalman_filter.py`, `test_association.py` |
| 2.9 Integration test | Feed a sequence of synthetic detections through ByteTrack and verify track continuity | `tests/integration/test_tracking_pipeline.py` |

#### 2.1 Kalman Filter: State Model Detail

```
State Vector (8-dimensional):
  x = [cx, cy, a, h, v_cx, v_cy, v_a, v_h]^T

Where:
  cx, cy  = bounding box centroid (pixels)
  a       = aspect ratio (width / height)
  h       = bounding box height (pixels)
  v_*     = velocity of each respective variable (pixels/frame)

State Transition Matrix F (constant velocity model):
  F = | 1 0 0 0 1 0 0 0 |
      | 0 1 0 0 0 1 0 0 |
      | 0 0 1 0 0 0 1 0 |
      | 0 0 0 1 0 0 0 1 |
      | 0 0 0 0 1 0 0 0 |
      | 0 0 0 0 0 1 0 0 |
      | 0 0 0 0 0 0 1 0 |
      | 0 0 0 0 0 0 0 1 |

Measurement Vector (4-dimensional):
  z = [cx, cy, a, h]^T

Measurement Matrix H:
  H = | 1 0 0 0 0 0 0 0 |
      | 0 1 0 0 0 0 0 0 |
      | 0 0 1 0 0 0 0 0 |
      | 0 0 0 1 0 0 0 0 |
```

#### 2.4 ByteTrack: Two-Stage Association Algorithm

```
Input:  List[Detection] for current frame, existing tracks from previous frame

Step 1: Split detections by confidence
        - high_dets = detections with confidence >= high_threshold (0.6)
        - low_dets  = detections with confidence >= low_threshold (0.1) AND < high_threshold

Step 2: Predict all existing tracks forward using Kalman Filter
        - predicted_bboxes = [track.kalman.predict() for track in active_tracks]

Step 3: FIRST ASSOCIATION (high-confidence detections)
        - Compute IoU cost matrix between predicted_bboxes and high_dets
        - Run Hungarian algorithm on cost matrix
        - Matched pairs: update track with detection (Kalman update)
        - Unmatched tracks: carry forward to second stage
        - Unmatched detections: carry forward to second stage

Step 4: SECOND ASSOCIATION (low-confidence detections)
        - Compute IoU cost matrix between unmatched_tracks and low_dets
        - Run Hungarian algorithm
        - Matched pairs: update track with low-conf detection
        - Unmatched tracks: increment time_since_update

Step 5: Track Lifecycle Management
        - New tracks: Create tentative track for each unmatched high-conf detection
        - Confirm: Track becomes confirmed after min_hits consecutive matches
        - Delete: Remove tracks where time_since_update > max_age

Output: List[Track] with updated positions, trails, and lifecycle states
```

**Phase 2 Milestone:** Run `python -m spatialtrack track --source sample.mp4` and see persistent colored trails following each vehicle across the entire clip.

---

### Phase 3: Homography Projection and Speed Estimation
**Goal:** Transform pixel trajectories into calibrated world coordinates and compute metric speed.

| Task | Details | Key Files |
|:-----|:--------|:----------|
| 3.1 Interactive calibration tool | OpenCV window where user clicks 4 source points on the camera frame, then inputs their real-world metric positions. Saves calibration as JSON. | `projection/calibration.py` |
| 3.2 Homography computation | `cv2.getPerspectiveTransform(src_pts, dst_pts)` to compute 3x3 matrix `H` | `projection/homography.py` |
| 3.3 Coordinate mapper | Apply `H` to project `PixelCoord -> WorldCoord` and inverse `H` for reverse mapping | `projection/coordinate_mapper.py` |
| 3.4 Speed estimator | Given two consecutive `WorldCoord` positions and known FPS: `distance / time_delta` -> `m/s` -> `km/h`, with EMA smoothing | `analytics/speed_estimator.py` |
| 3.5 Bird's-eye-view renderer | Render a clean top-down map with projected dots, trails, and speed labels per track | `visualization/bev_renderer.py` |
| 3.6 Dual-pane dashboard | Side-by-side compositor: `[Camera Feed | BEV Map]` with telemetry strip | `visualization/dashboard.py` |
| 3.7 Unit tests | Test homography with known correspondences, test speed calculation with synthetic data | `tests/unit/test_homography.py`, `test_speed_estimator.py` |

#### 3.1 Calibration: Step-by-Step User Workflow

```
1. Run: python -m spatialtrack calibrate --source sample.mp4
2. First frame displays in OpenCV window
3. User clicks 4 points on the road surface (e.g., lane markings, curb corners)
4. For each point, user enters real-world coordinates in meters via terminal prompt:
   "Point 1 clicked at pixel (523, 412). Enter world X (meters): 0.0"
   "Enter world Y (meters): 0.0"
5. After 4 points, the tool:
   a. Computes the homography matrix H
   b. Renders a preview showing the warped BEV perspective
   c. Draws a grid overlay on the original frame to visually verify projection quality
   d. Saves calibration data to JSON: { source_points, target_points, frame_size }
6. Output: configs/my_calibration.json
```

#### 3.4 Speed Estimation: Algorithm Detail

```
For each track at frame t:

Step 1: Project current pixel centroid to world coordinates
        world_t = H * pixel_centroid_t  (homography transform)

Step 2: Retrieve previous world position
        world_prev = track.world_trail[-1]

Step 3: Calculate Euclidean displacement in meters
        displacement_m = sqrt((world_t.x - world_prev.x)^2 + (world_t.y - world_prev.y)^2)

Step 4: Skip if displacement below jitter threshold (< 0.05m)
        This prevents stationary objects from showing noisy speeds.

Step 5: Calculate instantaneous speed
        time_delta_sec = 1.0 / source_fps
        speed_ms = displacement_m / time_delta_sec
        speed_kmh = speed_ms * 3.6

Step 6: Apply Exponential Moving Average (EMA) smoothing
        smoothed_speed = alpha * speed_kmh + (1 - alpha) * previous_smoothed_speed
        alpha = 0.3 (configurable)

Output: SpatialRecord with smoothed speed_kmh
```

**Phase 3 Milestone:** Run `python -m spatialtrack run --source sample.mp4 --calibration calibration.json` and see the full dual-pane view with real-time km/h speed labels.

---

### Phase 4: Analytics Engine and Event Detection
**Goal:** Build the intelligence layer that transforms raw spatial data into actionable events.

| Task | Details | Key Files |
|:-----|:--------|:----------|
| 4.1 Speed violation detector | Configurable speed threshold per zone; emits `SpatialEvent` when a track exceeds limit for N consecutive frames (debounce) | `analytics/event_detector.py` |
| 4.2 Zone monitor | Define polygonal zones (entry, restricted, parking); detect zone entry/exit/dwell using `cv2.pointPolygonTest` on world coordinates | `analytics/zone_monitor.py` |
| 4.3 Spatial heatmap | Gaussian kernel accumulator on a 2D grid aligned to BEV coordinates; normalized and rendered as a color-mapped overlay | `analytics/heatmap.py` |
| 4.4 Event logger | Structured JSON Lines event log with timestamp, event_type, track_id, position, metadata | `io/event_logger.py` |
| 4.5 Telemetry exporter | CSV export of per-frame telemetry: frame_index, track_id, pixel_x, pixel_y, world_x, world_y, speed_kmh | `io/telemetry_exporter.py` |
| 4.6 Unit tests | Test zone containment with known polygons, test heatmap accumulation, test event debouncing | `tests/unit/test_zone_monitor.py`, `test_heatmap.py` |

#### 4.1 Speed Violation Detection: Debounce Logic

```
For each track:
  Maintain a counter: consecutive_violation_frames = 0

  Each frame:
    if track.speed_kmh > speed_limit_kmh:
      consecutive_violation_frames += 1
    else:
      consecutive_violation_frames = 0

    if consecutive_violation_frames >= debounce_threshold (e.g., 5 frames):
      Emit SpatialEvent(
        event_type="speed_violation",
        track_id=track.track_id,
        metadata={
          "speed_kmh": track.speed_kmh,
          "speed_limit_kmh": speed_limit_kmh,
          "excess_kmh": track.speed_kmh - speed_limit_kmh
        }
      )
      Reset counter to avoid duplicate events for same violation
```

#### 4.2 Zone Monitor: Zone Configuration Format

```yaml
# configs/zones.yaml
zones:
  - name: "restricted_area"
    type: "restricted"
    polygon_world_m:     # Polygon vertices in world coordinates (meters)
      - [0.0, 0.0]
      - [10.0, 0.0]
      - [10.0, 5.0]
      - [0.0, 5.0]
    max_dwell_sec: 30    # Alert if object dwells longer than this

  - name: "entry_gate"
    type: "entry"
    polygon_world_m:
      - [12.0, 0.0]
      - [15.0, 0.0]
      - [15.0, 3.0]
      - [12.0, 3.0]
```

#### 4.3 Heatmap: Gaussian Accumulation

```
Maintain a 2D grid: heatmap_grid[H][W] = float, initialized to zeros
Grid dimensions map to world coordinate bounds.

For each track position (world_x, world_y) per frame:
  1. Map world_x, world_y to grid cell (gx, gy)
  2. Add a 2D Gaussian kernel centered at (gx, gy):
     heatmap_grid[y][x] += exp(-((x-gx)^2 + (y-gy)^2) / (2 * sigma^2))

For visualization:
  1. Normalize: heatmap_grid / max(heatmap_grid)
  2. Apply colormap: cv2.applyColorMap(normalized * 255, cv2.COLORMAP_JET)
  3. Alpha-blend with BEV map: output = alpha * heatmap_colored + (1-alpha) * bev_map
```

**Phase 4 Milestone:** Run the full pipeline and get a JSON event log of speed violations, a CSV telemetry dump, and a rendered heatmap overlay on the BEV map.

---

### Phase 5: Pipeline Orchestration and Profiling
**Goal:** Wire all modules into a clean, profiled processing loop with CLI interface.

| Task | Details | Key Files |
|:-----|:--------|:----------|
| 5.1 Pipeline engine | Main loop orchestrating all modules with profiler context managers | `pipeline/engine.py` |
| 5.2 Performance profiler | Per-stage wall-clock measurement with rolling averages | `pipeline/profiler.py` |
| 5.3 CLI with Typer | Commands: `run`, `detect`, `track`, `calibrate`, `benchmark`, `export` | `cli/main.py` |
| 5.4 Video writer | Output annotated video to file with configurable codec and resolution | `io/video_writer.py` |
| 5.5 Integration test | End-to-end test with synthetic video | `tests/integration/test_full_pipeline.py` |

#### 5.1 Pipeline Engine: Main Loop Pseudocode

```python
def run(config: SpatialTrackConfig) -> None:
    """Main processing loop orchestrating all pipeline stages."""
    # Initialize modules
    reader = VideoReader(config.video_source)
    detector = OnnxDetector(config.detection)
    tracker = ByteTracker(config.tracking)
    mapper = CoordinateMapper(config.projection)
    speed_estimator = SpeedEstimator(config.projection, reader.fps)
    event_detector = EventDetector(config.analytics)
    zone_monitor = ZoneMonitor(config.analytics)
    heatmap = HeatmapAccumulator(config.analytics)
    annotator = FrameAnnotator(config.visualization)
    bev_renderer = BevRenderer(config.visualization)
    dashboard = Dashboard(config.visualization)
    profiler = StageProfiler()
    writer = VideoWriter(config.output) if config.output.output_video_path else None
    event_logger = EventLogger(config.output) if config.output.event_log_path else None

    frame_index = 0
    for frame in reader:
        with profiler.measure("preprocess"):
            tensor, meta = preprocess(frame, config.detection.input_size)

        with profiler.measure("inference"):
            raw_output = detector.infer(tensor)

        with profiler.measure("postprocess"):
            detections = postprocess(raw_output, meta, config.detection)

        with profiler.measure("tracking"):
            tracks = tracker.update(detections, frame_index)

        with profiler.measure("projection"):
            spatial_records = mapper.project_tracks(tracks, frame_index)
            speed_estimator.update(spatial_records)

        with profiler.measure("analytics"):
            events = event_detector.check(spatial_records)
            events += zone_monitor.check(spatial_records)
            heatmap.accumulate(spatial_records)

        with profiler.measure("visualization"):
            annotated = annotator.draw(frame, tracks, spatial_records)
            bev_frame = bev_renderer.render(spatial_records, heatmap)
            output_frame = dashboard.compose(annotated, bev_frame, profiler.summary())

        # Output
        if writer:
            writer.write(output_frame)
        if event_logger and events:
            event_logger.log_batch(events)

        frame_index += 1

    # Cleanup and final exports
    profiler.print_summary()
```

#### 5.3 CLI Commands

```
Usage: python -m spatialtrack [COMMAND] [OPTIONS]

Commands:
  run         Run the full pipeline (detect + track + project + analyze + visualize)
  detect      Run detection only (no tracking)
  track       Run detection + tracking (no projection)
  calibrate   Launch interactive calibration tool
  benchmark   Run standardized CPU latency benchmark (100 frames)
  export      Re-export telemetry from a previous run's event log

Common Options:
  --source PATH        Video file, RTSP URL, or camera index (required)
  --config PATH        YAML configuration file (default: configs/default.yaml)
  --calibration PATH   Calibration JSON file (required for run/export)
  --output PATH        Output video file path
  --export-csv PATH    Export per-frame telemetry to CSV
  --export-events PATH Export events to JSON Lines file
  --show               Display live output window (default: False)
  --verbose            Enable debug logging
```

**Phase 5 Milestone:** Full CLI working end-to-end with profiled output.

---

### Phase 6: Interactive Demo and Free Deployment
**Goal:** Build a polished Streamlit web interface and deploy to Hugging Face Spaces at $0 cost.

| Task | Details | Key Files |
|:-----|:--------|:----------|
| 6.1 Streamlit app | File upload + sample video selector, config sidebar, dual-pane display, downloads | `app/streamlit_app.py` |
| 6.2 Pre-configured demo | Bundle short traffic clip with pre-computed calibration | `app/assets/` |
| 6.3 Dockerfile | Multi-stage build targeting HF Spaces free CPU tier | `Dockerfile` |
| 6.4 Hugging Face Space config | YAML frontmatter with `sdk: docker`, `app_port: 8501` | HF Space README |
| 6.5 Deploy and verify | Push to HF, verify within free tier limits (2 vCPUs, 16 GB RAM) | Manual verification |

#### 6.1 Streamlit App: UI Layout

```
+-----------------------------------------------------------------------+
|  SpatialTrack                                        [GitHub] [Docs]  |
+-------------------+---------------------------------------------------+
| SIDEBAR           |  MAIN AREA                                        |
|                   |                                                   |
| [Upload Video]    |  +-------------------------+-------------------+  |
| [Use Sample]      |  |                         |                   |  |
|                   |  |   Camera Feed            |   Bird's-Eye     |  |
| -- Detection --   |  |   (Annotated)            |   View Map       |  |
| Confidence: 0.25  |  |                         |                   |  |
| NMS IoU:    0.45  |  +-------------------------+-------------------+  |
|                   |                                                   |
| -- Tracking --    |  +---------------------------------------------------+
| Max Age:    30    |  | Telemetry Panel                               |  |
| Min Hits:   3     |  | FPS: 32.1 | Active Tracks: 7 | Events: 3     |  |
|                   |  | Latency: Detect 12ms | Track 3ms | Total 28ms |  |
| -- Analytics --   |  +---------------------------------------------------+
| Speed Limit: 60   |                                                   |
|                   |  +---------------------------------------------------+
| [Run Pipeline]    |  | Event Log (Last 10)                           |  |
|                   |  | [speed_violation] Track #14: 72.3 km/h        |  |
+-------------------+  | [zone_entry] Track #7: entered "restricted"   |  |
                        +---------------------------------------------------+
                        |                                                   |
                        | [Download Annotated Video] [Download CSV]         |
                        +---------------------------------------------------+
```

#### 6.3 Dockerfile

```dockerfile
# Multi-stage build for Hugging Face Spaces (free CPU tier)

# Stage 1: Builder
FROM python:3.11-slim AS builder
WORKDIR /build
COPY pyproject.toml .
COPY src/ src/
COPY app/ app/
RUN pip install --no-cache-dir --prefix=/install ".[app]"

# Stage 2: Runner
FROM python:3.11-slim AS runner
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ src/
COPY app/ app/
COPY configs/ configs/
COPY models/ models/

# Download model at build time
RUN python scripts/download_model.py

# Hugging Face Spaces expects port 7860, but Streamlit uses 8501
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true"]
```

**Phase 6 Milestone:** Live public URL where anyone can upload a traffic video and see real-time spatial tracking and speed estimation.

---

### Phase 7: Documentation, CI/CD, and Open-Source Polish
**Goal:** Make the repository production-quality for public consumption.

| Task | Details | Key Files |
|:-----|:--------|:----------|
| 7.1 README.md | Banner, features with GIFs, quick start, architecture diagram, benchmarks, badges | `README.md` |
| 7.2 Getting started guide | Step-by-step: install, download model, calibrate, run | `docs/getting_started.md` |
| 7.3 Calibration guide | Illustrated: how to pick points, measure distances, verify | `docs/calibration_guide.md` |
| 7.4 Architecture deep-dive | Module-by-module explanation with diagrams | `docs/architecture.md` |
| 7.5 Benchmark report | Latency table across CPU tiers with reproduction instructions | `docs/benchmarks.md` |
| 7.6 GitHub Actions CI | Ruff lint, MyPy strict, Pytest + coverage on every PR | `.github/workflows/ci.yml` |
| 7.7 CONTRIBUTING.md | Dev setup, branch naming, commit conventions, PR checklist | `CONTRIBUTING.md` |
| 7.8 CHANGELOG.md | Initial v0.1.0 entry in keep-a-changelog format | `CHANGELOG.md` |

#### 7.1 README Structure

```markdown
# SpatialTrack

<!-- Badges: CI status, coverage, PyPI version, license, Python version -->

> Real-time spatial telemetry, speed estimation, and trajectory analytics on CPU.

[Live Demo] | [Documentation] | [Getting Started]

## Demo GIF (dual-pane: camera feed + BEV map)

## Features
- 30+ FPS INT8 ONNX detection on CPU
- ByteTrack multi-object tracking with Kalman filtering
- Homography-based metric projection (pixels -> meters)
- Real-time speed estimation with EMA smoothing
- Speed violation and zone monitoring events
- Spatial density heatmaps
- Interactive calibration tool
- Per-stage latency profiler

## Quick Start (3 commands)
  pip install spatialtrack
  spatialtrack download-model
  spatialtrack run --source video.mp4 --calibration cal.json --show

## Architecture Diagram

## CPU Benchmark Table

## Configuration

## Citation

## License (MIT)
```

---

## 8. Technology Stack

| Layer | Technology | Why This Choice |
|:------|:-----------|:----------------|
| **Language** | Python 3.11+ | Industry standard for CV, rich ecosystem |
| **Detection Model** | YOLOv10n (nano) | Best accuracy/speed ratio for CPU |
| **Inference Runtime** | ONNX Runtime (CPU EP) | Cross-platform, INT8 quantization support |
| **Quantization** | ONNX Runtime Static INT8 | 2-3x speedup over FP32 on CPU |
| **Tracking** | ByteTrack (custom impl) | Lightweight, no Re-ID network needed |
| **State Estimation** | Kalman Filter (NumPy) | Classical, mathematically grounded |
| **Association** | Hungarian Algorithm (SciPy) | Optimal assignment, well-tested |
| **Projection** | OpenCV Homography | Direct `getPerspectiveTransform` |
| **Configuration** | Pydantic v2 + YAML | Type-safe config with validation |
| **CLI** | Typer | Modern, auto-generated help |
| **Web Demo** | Streamlit | Fast to build, free hosting |
| **Testing** | Pytest + Coverage | Industry standard |
| **Linting** | Ruff | Replaces flake8 + isort + black |
| **Type Checking** | MyPy (strict mode) | Catch type bugs before runtime |
| **CI/CD** | GitHub Actions | Free for public repos |
| **Packaging** | pyproject.toml (PEP 621) | Modern standard |
| **Containerization** | Docker (multi-stage) | Reproducible deployment |
| **Hosting** | Hugging Face Spaces (Free) | Free CPU Docker hosting |

---

## 9. Key Dependencies

```toml
# pyproject.toml [project.dependencies]
dependencies = [
    "numpy>=1.24,<2.0",
    "opencv-python-headless>=4.8",
    "onnxruntime>=1.16",
    "scipy>=1.11",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "pyyaml>=6.0",
    "typer>=0.9",
    "rich>=13.0",
    "structlog>=23.0",
]

# [project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "ruff>=0.1",
    "mypy>=1.5",
    "pre-commit>=3.4",
]
app = [
    "streamlit>=1.28",
    "streamlit-webrtc>=0.47",
]
```

---

## 10. Mathematical Foundations

### 10.1 Homography Projection

Given 4 pixel-space points (u_i, v_i) and their corresponding world-space points (X_i, Y_i), we compute the 3x3 homography matrix H such that:

```
| X' |       | u |
| Y' | = H * | v |
| w  |       | 1 |

World coordinates: X = X'/w,  Y = Y'/w
```

This transforms any pixel coordinate on the road plane into real-world metric coordinates.

### 10.2 Kalman Filter State Model

State vector for each tracked object (8-dimensional):

```
x = [cx, cy, a, h, v_cx, v_cy, v_a, v_h]^T
```

Where cx, cy is the bbox centroid, a is the aspect ratio, h is the height, and v_* are velocities per frame.

Constant velocity prediction:

```
x(k|k-1) = F * x(k-1|k-1)
P(k|k-1) = F * P(k-1|k-1) * F^T + Q
```

Kalman update when a detection is matched:

```
K = P(k|k-1) * H^T * (H * P(k|k-1) * H^T + R)^(-1)
x(k|k) = x(k|k-1) + K * (z - H * x(k|k-1))
P(k|k) = (I - K * H) * P(k|k-1)
```

### 10.3 Speed Estimation

Given two consecutive world positions P_t = (X_t, Y_t) and P_(t-1) = (X_(t-1), Y_(t-1)) at known FPS:

```
displacement = sqrt((X_t - X_(t-1))^2 + (Y_t - Y_(t-1))^2)   [meters]
speed_ms     = displacement / (1 / FPS)                         [m/s]
speed_kmh    = speed_ms * 3.6                                   [km/h]
```

Smoothed with exponential moving average:

```
smoothed_t = alpha * speed_kmh_t + (1 - alpha) * smoothed_(t-1)
alpha = 0.3 (configurable)
```

### 10.4 Hungarian Algorithm (Optimal Assignment)

Constructs cost matrix C_ij where:

```
C_ij = 1 - IoU(predicted_i, detected_j)
```

Solves the linear assignment problem to minimize total cost, producing optimal track-to-detection pairs.

### 10.5 IoU (Intersection over Union)

```
intersection = max(0, min(x2_a, x2_b) - max(x1_a, x1_b)) *
               max(0, min(y2_a, y2_b) - max(y1_a, y1_b))
union = area_a + area_b - intersection
IoU = intersection / union  (0.0 to 1.0)
```

---

## 11. Testing Strategy

### Unit Tests (Fast, Isolated, < 50ms each)

| Module | Test Cases |
|:-------|:-----------|
| **Preprocessing** | Output tensor shape `(1, 3, 640, 640)`, value range `[0, 1]`, letterbox padding correctness, aspect ratio preservation |
| **NMS** | Overlapping bboxes suppressed correctly, non-overlapping preserved, empty input returns empty |
| **Kalman Filter** | Predict/update cycle with known inputs converges, state vector dimensions correct, covariance stays positive-definite |
| **IoU** | No overlap returns 0.0, perfect overlap returns 1.0, partial overlap matches expected, zero-area bbox handled |
| **Hungarian** | Known 3x3 cost matrix returns optimal assignment, rectangular matrices handled, inf costs handled |
| **Homography** | 4 known point pairs with forward projection accurate within 0.01m, inverse projection round-trips correctly |
| **Speed** | Known displacement (10m) + known FPS (30) returns expected km/h, jitter threshold filters micro-displacement |
| **Zone Monitor** | Point inside polygon returns True, outside returns False, on edge returns True |
| **Heatmap** | Single point accumulation at center produces expected Gaussian shape, reset clears grid |
| **Config** | Valid YAML loads correctly, missing required field raises ConfigurationError, env var override works |

### Integration Tests (Module Chains, < 2s each)

| Pipeline | Description |
|:---------|:------------|
| **Detection** | Sample frame -> preprocessing -> mock ONNX inference -> postprocessing -> verify Detection objects have valid bboxes |
| **Tracking** | Sequence of 10 synthetic detection frames -> ByteTrack -> verify track IDs persist across all frames |
| **Full Pipeline** | 30-frame synthetic video -> engine -> verify SpatialRecords output matches expected, events log contains expected violations |

### Performance Tests (Non-blocking, Informational)

```
scripts/benchmark_cpu.py:
  - Load model once
  - Run 100-frame inference loop (warm up first 10 frames, measure last 90)
  - Report: mean, median, p95, p99 latency per stage
  - Report: total pipeline FPS
  - Report: memory usage (RSS)
```

---

## 12. CI/CD Pipeline

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install ruff
      - run: ruff check src/ tests/
      - run: ruff format --check src/ tests/

  typecheck:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: mypy src/spatialtrack --strict

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -v --cov=spatialtrack --cov-report=xml
      - uses: codecov/codecov-action@v3
        with:
          file: coverage.xml
```

---

## 13. Deployment Targets (All Free)

| Platform | Method | Cost | Limits |
|:---------|:-------|:-----|:-------|
| **Hugging Face Spaces** | Docker SDK | Free | 2 vCPUs, 16 GB RAM, public URL |
| **Streamlit Cloud** | Direct GitHub deploy | Free | Public repos, 1 GB RAM |
| **GitHub Pages** | Static docs site | Free | Documentation hosting |
| **PyPI** | `python -m build` + `twine upload` | Free | Installable via `pip install spatialtrack` |

---

## 14. Execution Timeline

| Week | Phase | Deliverable |
|:-----|:------|:------------|
| **Week 1** | Phase 1: Foundation + Detection | INT8 ONNX detector running at 30+ FPS on CPU |
| **Week 2** | Phase 2: Tracking | ByteTrack with persistent colored trails |
| **Week 3** | Phase 3: Projection + Speed | Homography calibration + dual-pane BEV view with km/h labels |
| **Week 4** | Phase 4: Analytics | Speed violations, zone monitoring, heatmaps, event logs |
| **Week 5** | Phase 5: Pipeline + CLI | Complete CLI with profiler, video export, telemetry CSV |
| **Week 6** | Phase 6 + 7: Demo + Polish | Live Hugging Face demo, full documentation, CI/CD, public GitHub release |

---

## 15. Resume Impact

When listing this project on your resume, frame it as:

*"Designed and open-sourced SpatialTrack, a CPU-optimized spatial telemetry engine
achieving 30+ FPS INT8 inference with homography-based metric projection,
Kalman-filtered multi-object tracking, and real-time speed estimation.
Deployed as a zero-cost interactive demo on Hugging Face Spaces."*

---

## 16. Approval Checklist

Before proceeding to implementation, verify:

- [ ] Project name "SpatialTrack" is acceptable
- [ ] Repository structure covers all needed modules
- [ ] Domain types are comprehensive and correctly typed
- [ ] Configuration system covers all tunable parameters
- [ ] 7-phase timeline is realistic for your schedule
- [ ] Technology stack choices are approved
- [ ] Free deployment targets are sufficient
- [ ] Testing strategy is thorough enough
- [ ] CI/CD pipeline covers lint, type-check, and tests
- [ ] Documentation plan meets open-source standards
