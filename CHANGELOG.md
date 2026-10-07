# Changelog

All notable changes to this project are documented in this file.
This changelog helps track every modification, decision, and milestone.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - Phase 6: Interactive Demo and Free Deployment (Completed)

### Added
- **Streamlit Web Application (`app/streamlit_app.py`):**
  - Interactive web application running full spatial tracking pipeline on CPU.
  - Dual-pane live display showing annotated camera stream and Bird's-Eye-View metric canvas.
  - Real-time KPI metric cards: pipeline FPS, active tracks, speed violations, and stage latency.
  - In-browser artifact downloads: processed MP4 video, telemetry CSV, and event JSONL files.
- **Modular Web UI Components (`app/components/`):**
  - `config_sidebar.py`: controls for video source selection (demo clip or custom upload), detection confidence, CPU threads, speed limit thresholds, and spatial heatmaps.
  - `results_display.py`: KPI dashboard layout, pandas event log table, and export buttons.
- **Multi-Stage Dockerfile (`Dockerfile`):**
  - Production container build targeting Hugging Face Spaces free CPU tier (2 vCPUs, 16 GB RAM).
  - Pre-downloads INT8 ONNX model at build time, configures non-root user permissions, and exposes port 7860 with health checks.
- **Automated Unit Tests (`tests/unit/test_app_components.py`):**
  - Tests covering sidebar configuration extraction, metric cards rendering, and layout headers.
  - Total passing tests increased to 68 with 89% codebase coverage.

## [0.5.0] - Phase 5: Pipeline Orchestration and Engine (Completed)

### Added
- **Unified Pipeline Engine (`src/spatialtrack/pipeline/engine.py`):**
  - Implemented `SpatialTrackEngine` coordinating all pipeline stages: detection, ByteTrack tracking, homography projection, metric velocity estimation, zone monitoring, heatmaps, and dashboard compositing.
  - Generator API `process_stream()` yielding typed `FrameResult` objects per frame.
  - Runner method `run()` executing end-to-end processing with automated sink routing (video container, CSV, JSONL).
  - Achieved 99% test coverage on engine orchestration.
- **Context-Managed Video Writer (`src/spatialtrack/io/video_writer.py`):**
  - Implemented `VideoWriter` wrapping OpenCV VideoWriter with directory creation, codec management, and dimension normalization.
- **Interactive Calibration Tool (`src/spatialtrack/cli/calibration_cli.py`):**
  - 4-point ground homography calibration wizard supporting interactive mouse picking and programmatic point strings.
  - Matrix non-singularity and condition number checks.
- **Standardized CPU Benchmark (`src/spatialtrack/cli/benchmark_cli.py`):**
  - Latency and throughput benchmarking measuring per-stage percentiles across CPU thread counts.
- **CLI Architecture Refactoring (`src/spatialtrack/cli/`):**
  - Modularized into `main.py`, `runners.py`, `calibration_cli.py`, and `benchmark_cli.py`.
  - Added `spatialtrack run`, `spatialtrack calibrate`, and `spatialtrack benchmark` commands.
  - Every CLI file is kept under 240 lines with all functions under 35 lines.
- **Automated Tests (`tests/`):**
  - Added `test_video_writer.py`, `test_engine.py`, `test_calibration_cli.py`, `test_benchmark_cli.py`, and `test_full_pipeline.py`.
  - Total passing tests increased to 64 with 89% code coverage.

## [0.4.0] - Phase 4: Analytics Engine and Event Detection (Completed)

### Added
- **Speed Violation Detector (`src/spatialtrack/analytics/event_detector.py`):**
  - Temporal debouncing state machine requiring consecutive violation frames before emitting discrete `SpatialEvent`.
  - Calculates exact excess speed and event timestamp.
- **Polygonal Zone Monitor (`src/spatialtrack/analytics/zone_monitor.py`):**
  - Geofenced polygonal zone monitoring in real-world metric coordinates.
  - Generates `ZONE_ENTRY`, `ZONE_EXIT`, and `ZONE_DWELL` events based on configurable dwell time thresholds.
- **Spatial Heatmap Accumulator (`src/spatialtrack/analytics/heatmap.py`):**
  - 2D Gaussian density kernel accumulation over orthographic world coordinate grid.
  - JET colormap rendering with alpha blending and zero-density background masking.
- **Structured Event Logger (`src/spatialtrack/io/event_logger.py`):**
  - Thread-safe JSON Lines (`.jsonl`) writer recording structured spatial events.
- **Telemetry Exporter (`src/spatialtrack/io/telemetry_exporter.py`):**
  - High-performance CSV telemetry exporter writing frame indices, track IDs, pixel and world coordinates, and velocity.
- **CLI Telemetry Integration (`src/spatialtrack/cli/main.py`):**
  - Added `--export-csv`, `--export-events`, and `--heatmap` CLI options to `spatialtrack telemetry`.
  - Refactored CLI commands and frame loops into modular helper functions under 40 lines each.
- **Automated Unit Tests (`tests/unit/`):**
  - Added comprehensive test suites: `test_event_detector.py`, `test_zone_monitor.py`, `test_heatmap.py`, `test_event_logger.py`, `test_telemetry_exporter.py`.
  - Achieved 45 passing tests with 83% test coverage and 100% strict MyPy and Ruff compliance.

## [0.3.0] - Phase 3: Homography Projection and Speed Estimation (Completed)

### Added
- **Calibration Storage and Validation (`src/spatialtrack/projection/calibration.py`):**
  - Validation ensuring exactly 4 distinct, non-collinear correspondence points.
  - JSON serialization and deserialization utilities for camera calibration files.
  - Bundled verified sample calibration for demo traffic video (`app/assets/sample_calibration.json`).
- **Homography Perspective Transform (`src/spatialtrack/projection/homography.py`):**
  - Solves 3x3 forward and inverse homography matrices using OpenCV perspective transform.
  - Validates matrix condition number and non-singularity.
  - Efficient perspective point projection using homogeneous coordinates and perspective division.
- **Coordinate Mapper (`src/spatialtrack/projection/coordinate_mapper.py`):**
  - Bidirectional coordinate mapping between 2D camera pixel coordinates and ground metric coordinates (meters).
  - Trajectory trail projection mapping pixel trails into world coordinate space.
- **Metric Speed Estimator (`src/spatialtrack/analytics/speed_estimator.py`):**
  - Calculates instantaneous Euclidean displacement and velocity in meters/second and km/h.
  - Exponential Moving Average (EMA) smoothing suppressing sensor frame noise.
  - Micro-jitter suppression filtering out stationary displacement noise below threshold.
  - Speed limit threshold violation flagging.
- **Bird's-Eye-View (BEV) Map Renderer (`src/spatialtrack/visualization/bev_renderer.py`):**
  - Top-down orthographic road canvas with asphalt background, metric interval grid lines, and distance indicators.
  - Renders dynamic vehicle beacons, projected world trajectory trails, and speed tags (highlighted in red for speed violations).
- **Dual-Pane Dashboard Compositor (`src/spatialtrack/visualization/dashboard.py`):**
  - Composites synchronized side-by-side view: annotated camera feed on left, metric BEV map on right.
  - Bottom telemetry banner displaying real-time FPS, track count, active violations, and pipeline latency breakdown.
- **CLI Telemetry Command (`src/spatialtrack/cli/main.py`):**
  - Added `spatialtrack telemetry` command executing full pipeline: detection, tracking, homography projection, speed estimation, and dual-pane visualization.
- **Automated Tests for Phase 3 (`tests/`):**
  - `tests/unit/test_homography.py`: Forward and inverse projective geometry roundtrip verification.
  - `tests/unit/test_coordinate_mapper.py`: Bidirectional coordinate mapping and JSON serialization tests.
  - `tests/unit/test_speed_estimator.py`: Velocity calculation, violation flagging, and jitter suppression tests.
  - `tests/unit/test_bev_renderer.py`: Orthographic map rendering tests.
  - `tests/unit/test_dashboard.py`: Side-by-side compositor dimension and banner tests.

## [0.2.0] - Phase 2: Multi-Object Tracking (Completed)

### Added
- **Base Tracker Protocol (`src/spatialtrack/tracking/base_tracker.py`):** Defined `BaseTracker` protocol with `update` and `reset` lifecycle methods.
- **2D Kalman Filter (`src/spatialtrack/tracking/kalman_filter.py`):**
  - Implemented 8-dimensional state vector `[cx, cy, a, h, v_cx, v_cy, v_a, v_h]` using constant-velocity transition matrix.
  - Formulated measurement updates from 4-dimensional bounding box observations `[cx, cy, a, h]`.
  - Added object-height proportional process and measurement noise covariance scaling.
  - Implemented bidirectional conversions between `BoundingBox` and Kalman distribution space with 100% test coverage.
- **Bipartite Matching and Cost Association (`src/spatialtrack/tracking/association.py`):**
  - Vectorized pairwise IoU overlap matrix computation between track predictions and new detections.
  - Hungarian algorithm solver via `scipy.optimize.linear_sum_assignment` with maximum cost threshold gating.
- **ByteTrack Tracker Engine (`src/spatialtrack/tracking/bytetrack.py`):**
  - Implemented two-stage bipartite association: Stage 1 matches high-confidence detections; Stage 2 matches low-confidence detections to recover occluded objects.
  - Full track lifecycle state machine: `TENTATIVE` -> `CONFIRMED` -> `LOST` -> `DELETED`.
  - Monotonic track ID assignment, ground contact point tracking, instantaneous pixel velocity estimation, and bounded trajectory trail accumulation.
- **Visual Trajectory Annotation (`src/spatialtrack/visualization/annotator.py`):**
  - Enhanced `FrameAnnotator` with `draw_tracks` method.
  - Renders progressive fading trajectory trails connecting historical positions.
  - Renders ground contact points and persistent identity badges (`#ID ClassName`).
- **CLI Tracking Command (`src/spatialtrack/cli/main.py`):**
  - Added `spatialtrack track` command orchestrating video decoding, INT8 detection, and ByteTrack tracking.
  - Live console telemetry displaying detections, active tracks, and sub-millisecond tracking latency.
- **Automated Tests for Phase 2 (`tests/`):**
  - `tests/unit/test_kalman_filter.py`: Unit tests for initiate, predict, update, and coordinate roundtrips (100% coverage).
  - `tests/unit/test_association.py`: Unit tests for IoU matrix calculation and Hungarian matching.
  - `tests/unit/test_bytetrack.py`: Unit tests for track persistence across frames and expiration after max age.
  - `tests/unit/test_annotator.py`: Unit test verifying track rendering and fading trail visualization.
  - `tests/integration/test_tracking_pipeline.py`: Multi-frame integration test verifying trajectory continuity and zero identity switching.

## [0.1.0] - Phase 1: Foundation and Detection Pipeline (Completed)

### Added
- **Project Blueprint:** Created `IMPLEMENTATION_PLAN.md` with full 7-phase system architecture, domain types, mathematical foundations, testing matrix, and deployment guide.
- **Repository Scaffolding:**
  - `pyproject.toml`: Modern PEP 621 packaging with strict dependencies, build system, and tool configurations (Ruff, MyPy, Pytest).
  - `.gitignore`: Configured for Python virtual environments, compiled objects, large model binaries, and test artifacts.
  - `configs/default.yaml`: Default system configuration covering detection, tracking, projection, analytics, visualization, and output parameters.
- **Core Domain Layer (`src/spatialtrack/core/`):**
  - `types.py`: Strict domain typing with primitives (`TrackId`, `FrameIndex`, `MetersPerSecond`, `KilometersPerHour`, `Confidence`, `PixelCoord`, `WorldCoord`), `ObjectClass` enum with string mapping, `BoundingBox` geometry dataclass with centroid and ground bottom-center calculations, `Detection`, `Track`, `SpatialRecord`, `SpatialEvent`, `CalibrationData`, and `LatencyBreakdown`.
  - `config.py`: Type-safe configuration models via Pydantic v2 `BaseSettings` with YAML deserialization and environment variable overrides.
  - `constants.py`: System constants (COCO classes, default input dimensions, multipliers).
  - `exceptions.py`: Custom typed exception hierarchy (`SpatialTrackError`, `ModelLoadError`, `VideoSourceError`, `CalibrationError`, `ConfigurationError`, `InferenceError`, `ExportError`).
- **Detection Module (`src/spatialtrack/detection/`):**
  - `base_detector.py`: Abstract Protocol defining uniform detector contract.
  - `preprocessing.py`: Aspect-ratio-preserving letterbox resizing with padding, BGR to RGB conversion, float32 normalization to [0.0, 1.0], and NCHW tensor transformation.
  - `postprocessing.py`: Vectorized IoU calculation, Non-Maximum Suppression (NMS), coordinate rescaling back to unpadded source resolution, and multi-format proposal decoding (YOLOv10 and standard YOLO formats).
  - `onnx_detector.py`: Production-grade ONNX Runtime CPU detector with intra/inter-thread tuning and error handling.
- **IO Module (`src/spatialtrack/io/`):**
  - `video_reader.py`: Threaded asynchronous video frame reader with bounded queue, decoupling video decoding from downstream inference for smooth real-time streaming.
- **Visualization Module (`src/spatialtrack/visualization/`):**
  - `color_palette.py`: Deterministic BGR color palette generator for consistent tracking identifiers.
  - `annotator.py`: Frame annotator rendering bounding boxes, badge labels, and confidence tags without mutating original image arrays.
- **Pipeline and Profiling Layer (`src/spatialtrack/pipeline/`):**
  - `profiler.py`: Per-stage latency profiler context manager computing rolling percentiles (Mean, P95, Min, Max) and estimated FPS throughput.
- **CLI and Entrypoints (`src/spatialtrack/cli/`):**
  - `main.py`: Typer CLI providing `detect` command with live frame-by-frame latency reporting and summary table.
  - `__main__.py`: Direct module execution entrypoint (`python -m spatialtrack`).
- **Scripts and Assets:**
  - `scripts/download_model.py`: Automated downloader for official YOLOv10n weights with dynamic INT8 quantization.
  - `scripts/export_quantized_model.py`: CLI tool for INT8 model quantization.
  - `scripts/benchmark_cpu.py`: Standardized benchmark comparing FP32 vs INT8 models across thread configurations.
  - `app/assets/sample_traffic.mp4`: 720p 150-frame traffic video clip for end-to-end testing and demo execution.
- **Automated Testing Suite (`tests/`):**
  - `tests/unit/test_config.py`: Verifies Pydantic defaults and YAML parsing.
  - `tests/unit/test_preprocessing.py`: Verifies letterbox geometry, aspect ratio preservation, and tensor formatting.
  - `tests/unit/test_postprocessing.py`: Verifies vectorized IoU, NMS filtering, and box rescaling.
  - `tests/unit/test_video_reader.py`: Verifies threaded video decoding and error boundaries.
  - `tests/unit/test_color_palette.py`: Verifies deterministic color generation.
  - `tests/unit/test_annotator.py`: Verifies image immutability and drawing.
  - `tests/unit/test_profiler.py`: Verifies latency measurement and summary breakdown.
  - `tests/unit/test_cli.py`: Verifies CLI version and parameter validation.
  - `tests/integration/test_detection_pipeline.py`: Verifies ONNX Runtime INT8 model loading and inference on synthetic frames.
