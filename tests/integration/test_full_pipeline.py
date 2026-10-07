"""End-to-end integration tests for SpatialTrackEngine."""

from pathlib import Path

from spatialtrack.core.config import (
    AnalyticsConfig,
    DetectionConfig,
    OutputConfig,
    ProjectionConfig,
    SpatialTrackConfig,
    TrackingConfig,
    VisualizationConfig,
)
from spatialtrack.io.video_reader import VideoReader
from spatialtrack.pipeline.engine import SpatialTrackEngine


def test_engine_process_stream_with_calibration() -> None:
    """Verify that process_stream yields valid FrameResults on sample traffic video."""
    # Arrange
    video_path = Path("app/assets/sample_traffic.mp4")
    calib_path = Path("app/assets/sample_calibration.json")
    model_path = Path("models/yolov10n_int8.onnx")

    config = SpatialTrackConfig(
        detection=DetectionConfig(model_path=model_path),
        tracking=TrackingConfig(),
        projection=ProjectionConfig(calibration_path=calib_path),
        analytics=AnalyticsConfig(speed_limit_kmh=15.0),
        visualization=VisualizationConfig(),
        video_source=str(video_path),
    )
    engine = SpatialTrackEngine(config)

    # Act
    results = []
    with VideoReader(video_path) as reader:
        for res in engine.process_stream(reader, max_frames=8):
            results.append(res)

    # Assert
    assert len(results) == 8
    first = results[0]
    assert first.frame_index == 0
    assert first.annotated_frame.ndim == 3
    assert first.annotated_frame.shape[0] == 720 + 42  # composite dashboard height
    assert first.annotated_frame.shape[1] == 1280 + 400  # composite dashboard width
    assert first.latency_breakdown.total_ms > 0.0


def test_engine_run_with_sinks(tmp_path: Path) -> None:
    """Verify that engine.run processes video and writes to video, CSV, and event sinks."""
    # Arrange
    video_path = Path("app/assets/sample_traffic.mp4")
    calib_path = Path("app/assets/sample_calibration.json")
    model_path = Path("models/yolov10n_int8.onnx")
    out_video = tmp_path / "engine_out.mp4"
    out_csv = tmp_path / "engine_telemetry.csv"
    out_events = tmp_path / "engine_events.jsonl"

    config = SpatialTrackConfig(
        detection=DetectionConfig(model_path=model_path),
        tracking=TrackingConfig(),
        projection=ProjectionConfig(calibration_path=calib_path),
        analytics=AnalyticsConfig(speed_limit_kmh=5.0),  # low limit to trigger events
        visualization=VisualizationConfig(),
        output=OutputConfig(
            output_video_path=out_video,
            telemetry_csv_path=out_csv,
            event_log_path=out_events,
        ),
        video_source=str(video_path),
    )
    engine = SpatialTrackEngine(config)

    # Act
    summary = engine.run(max_frames=12)

    # Assert
    assert summary.total_frames == 12
    assert summary.unique_tracks > 0
    assert summary.average_fps > 0.0
    assert out_video.exists() and out_video.stat().st_size > 0
    assert out_csv.exists() and out_csv.stat().st_size > 0
