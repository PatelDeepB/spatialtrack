"""Main CLI application using Typer and Rich."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import typer
from numpy.typing import NDArray
from rich.console import Console

from spatialtrack.analytics.event_detector import SpeedViolationDetector
from spatialtrack.analytics.heatmap import SpatialHeatmap
from spatialtrack.analytics.speed_estimator import SpeedEstimator
from spatialtrack.core.config import (
    AnalyticsConfig,
    DetectionConfig,
    TrackingConfig,
    VisualizationConfig,
)
from spatialtrack.core.types import FrameIndex, SpatialRecord, WorldCoord
from spatialtrack.detection.onnx_detector import OnnxDetector
from spatialtrack.io.event_logger import EventLogger
from spatialtrack.io.telemetry_exporter import TelemetryExporter
from spatialtrack.io.video_reader import VideoReader
from spatialtrack.pipeline.profiler import StageProfiler
from spatialtrack.projection.calibration import load_calibration
from spatialtrack.projection.coordinate_mapper import CoordinateMapper
from spatialtrack.tracking.bytetrack import ByteTracker
from spatialtrack.visualization.annotator import FrameAnnotator
from spatialtrack.visualization.bev_renderer import BevRenderer
from spatialtrack.visualization.dashboard import Dashboard

app = typer.Typer(
    name="spatialtrack",
    help="CPU-optimized spatial telemetry and speed estimation engine.",
    add_completion=False,
)
console = Console()


@dataclass
class _TelemetryContext:
    """Internal container aggregating telemetry pipeline components."""

    detector: OnnxDetector
    tracker: ByteTracker
    mapper: CoordinateMapper
    speed_estimator: SpeedEstimator
    bev_renderer: BevRenderer
    dashboard: Dashboard
    profiler: StageProfiler
    annotator: FrameAnnotator
    violation_detector: SpeedViolationDetector
    heatmap: SpatialHeatmap | None
    csv_exporter: TelemetryExporter | None
    event_logger: EventLogger | None


@app.command()
def detect(
    source: str = typer.Option(..., "--source", "-s", help="Path to video file or webcam index."),
    model_path: Path = typer.Option(
        Path("models/yolov10n_int8.onnx"),
        "--model",
        "-m",
        help="Path to ONNX model weights.",
    ),
    conf: float = typer.Option(0.25, "--conf", "-c", help="Confidence detection threshold."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Optional output video path."),
    max_frames: int | None = typer.Option(None, "--max-frames", help="Maximum frames to process."),
    show: bool = typer.Option(False, "--show", help="Display visual playback window."),
) -> None:
    """Run CPU-optimized object detection on video stream and report frame latency."""
    console.print("[bold cyan]SpatialTrack[/bold cyan] - Starting detection pipeline on CPU")
    detector = OnnxDetector(DetectionConfig(model_path=model_path, confidence_threshold=conf))
    annotator = FrameAnnotator(VisualizationConfig())
    profiler = StageProfiler()
    writer: cv2.VideoWriter | None = None

    try:
        with VideoReader(source) as reader:
            writer = _init_writer(output, reader.fps, (reader.width, reader.height))
            frames = _run_detect_loop(
                reader, detector, annotator, profiler, writer, max_frames, show
            )
    finally:
        _cleanup_io(writer, show)

    console.print(f"\n[bold green]Finished processing {frames} frames.[/bold green]")
    profiler.print_summary_table(console)


@app.command()
def track(
    source: str = typer.Option(..., "--source", "-s", help="Path to video file or webcam index."),
    model_path: Path = typer.Option(
        Path("models/yolov10n_int8.onnx"),
        "--model",
        "-m",
        help="Path to ONNX model weights.",
    ),
    conf: float = typer.Option(0.25, "--conf", "-c", help="Confidence detection threshold."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Optional output video path."),
    max_frames: int | None = typer.Option(None, "--max-frames", help="Maximum frames to process."),
    show: bool = typer.Option(False, "--show", help="Display visual playback window."),
) -> None:
    """Run CPU-optimized multi-object tracking (ByteTrack) with persistent trails."""
    console.print("[bold cyan]SpatialTrack[/bold cyan] - Starting ByteTrack pipeline on CPU")
    detector = OnnxDetector(DetectionConfig(model_path=model_path, confidence_threshold=conf))
    tracker = ByteTracker(TrackingConfig())
    annotator = FrameAnnotator(VisualizationConfig(show_trails=True))
    profiler = StageProfiler()
    writer: cv2.VideoWriter | None = None

    try:
        with VideoReader(source) as reader:
            writer = _init_writer(output, reader.fps, (reader.width, reader.height))
            frames = _run_track_loop(
                reader, detector, tracker, annotator, profiler, writer, max_frames, show
            )
    finally:
        _cleanup_io(writer, show)

    console.print(f"\n[bold green]Finished tracking {frames} frames.[/bold green]")
    profiler.print_summary_table(console)


@app.command()
def telemetry(
    source: str = typer.Option(..., "--source", "-s", help="Path to video file or webcam index."),
    calibration: Path = typer.Option(
        Path("app/assets/sample_calibration.json"),
        "--calibration",
        "-c",
        help="Path to calibration JSON file.",
    ),
    model_path: Path = typer.Option(
        Path("models/yolov10n_int8.onnx"),
        "--model",
        "-m",
        help="Path to ONNX model weights.",
    ),
    output: Path | None = typer.Option(None, "--output", "-o", help="Optional output video path."),
    max_frames: int | None = typer.Option(None, "--max-frames", help="Maximum frames to process."),
    speed_limit: float = typer.Option(60.0, "--speed-limit", help="Speed limit in km/h."),
    export_csv: Path | None = typer.Option(
        None, "--export-csv", help="Export per-frame CSV telemetry."
    ),
    export_events: Path | None = typer.Option(
        None, "--export-events", help="Export JSONL event logs."
    ),
    heatmap: bool = typer.Option(False, "--heatmap", help="Overlay 2D density heatmap on BEV."),
    show: bool = typer.Option(False, "--show", help="Display visual playback window."),
) -> None:
    """Run full spatial telemetry pipeline on CPU with synchronized dual-pane BEV view."""
    console.print("[bold cyan]SpatialTrack[/bold cyan] - Starting Spatial Telemetry & Speed Engine")
    mapper = CoordinateMapper(load_calibration(calibration))
    writer: cv2.VideoWriter | None = None
    csv_exp = TelemetryExporter(export_csv) if export_csv else None
    evt_log = EventLogger(export_events) if export_events else None

    try:
        with VideoReader(source) as reader:
            ctx = _build_telemetry_ctx(
                mapper, reader.fps, speed_limit, model_path, heatmap, csv_exp, evt_log
            )
            out_dim = (reader.width + 400, reader.height + 42)
            writer = _init_writer(output, reader.fps, out_dim)
            frame_count = _run_telemetry_loop(reader, ctx, writer, max_frames, show)
    finally:
        _cleanup_telemetry(writer, show, csv_exp, evt_log)

    console.print(f"\n[bold green]Finished telemetry on {frame_count} frames.[/bold green]")
    ctx.profiler.print_summary_table(console)


@app.command()
def version() -> None:
    """Print package version."""
    from spatialtrack import __version__

    console.print(f"SpatialTrack version: [bold cyan]{__version__}[/bold cyan]")


def _run_detect_loop(
    reader: VideoReader,
    detector: OnnxDetector,
    annotator: FrameAnnotator,
    profiler: StageProfiler,
    writer: cv2.VideoWriter | None,
    max_frames: int | None,
    show: bool,
) -> int:
    """Iterate over frames running detection and visualization."""
    count = 0
    for frame in reader:
        if max_frames is not None and count >= max_frames:
            break
        with profiler.measure("inference"):
            detections = detector.detect(frame, FrameIndex(count))
        with profiler.measure("visualization"):
            annotated = annotator.draw_detections(frame, detections)
        if writer is not None:
            writer.write(annotated)
        if show and _show_frame("SpatialTrack - Detection", annotated):
            break
        count += 1
    return count


def _run_track_loop(
    reader: VideoReader,
    detector: OnnxDetector,
    tracker: ByteTracker,
    annotator: FrameAnnotator,
    profiler: StageProfiler,
    writer: cv2.VideoWriter | None,
    max_frames: int | None,
    show: bool,
) -> int:
    """Iterate over frames running ByteTrack tracking and visualization."""
    count = 0
    for frame in reader:
        if max_frames is not None and count >= max_frames:
            break
        with profiler.measure("inference"):
            detections = detector.detect(frame, FrameIndex(count))
        with profiler.measure("tracking"):
            tracks = tracker.update(detections, FrameIndex(count))
        with profiler.measure("visualization"):
            annotated = annotator.draw_tracks(frame, tracks)
        if writer is not None:
            writer.write(annotated)
        if show and _show_frame("SpatialTrack - Tracking", annotated):
            break
        count += 1
    return count


def _process_telemetry_frame(
    ctx: _TelemetryContext,
    frame: NDArray[np.uint8],
    frame_idx: int,
) -> NDArray[np.uint8]:
    """Process single video frame through detection, tracking, projection, and telemetry."""
    with ctx.profiler.measure("inference"):
        detections = ctx.detector.detect(frame, FrameIndex(frame_idx))
    with ctx.profiler.measure("tracking"):
        tracks = ctx.tracker.update(detections, FrameIndex(frame_idx))

    records: list[SpatialRecord] = []
    world_trails: dict[int, list[WorldCoord]] = {}
    with ctx.profiler.measure("projection_speed"):
        for track_obj in tracks:
            world_pos, w_trail = ctx.mapper.project_track(track_obj)
            record = ctx.speed_estimator.estimate_speed(track_obj, world_pos, FrameIndex(frame_idx))
            records.append(record)
            world_trails[int(track_obj.track_id)] = w_trail

    if ctx.heatmap is not None:
        ctx.heatmap.accumulate([r.world_position for r in records])
    if ctx.csv_exporter is not None:
        ctx.csv_exporter.export_records(records)
    if ctx.event_logger is not None:
        events = ctx.violation_detector.update(records)
        ctx.event_logger.log_events(events)

    with ctx.profiler.measure("visualization"):
        annotated_cam = ctx.annotator.draw_tracks(frame, tracks)
        bev_map = ctx.bev_renderer.render(records, world_trails, heatmap=ctx.heatmap)
        violations = sum(1 for r in records if r.is_speed_violation)
        return ctx.dashboard.compose(
            annotated_cam,
            bev_map,
            len(tracks),
            violations,
            ctx.profiler.get_current_breakdown(),
        )


def _run_telemetry_loop(
    reader: VideoReader,
    ctx: _TelemetryContext,
    writer: cv2.VideoWriter | None,
    max_frames: int | None,
    show: bool,
) -> int:
    """Run frame processing loop for telemetry and display/write output."""
    count = 0
    for frame in reader:
        if max_frames is not None and count >= max_frames:
            break
        composite = _process_telemetry_frame(ctx, frame, count)
        if writer is not None:
            writer.write(composite)
        if show and _show_frame("SpatialTrack - Telemetry & BEV", composite):
            break
        count += 1
        if count % 30 == 0:
            bd = ctx.profiler.get_current_breakdown()
            console.print(
                f"Frame {count:04d} | Infer: {bd.inference_ms:.1f}ms | "
                f"Track: {bd.tracking_ms:.2f}ms | Proj: {bd.projection_ms:.2f}ms"
            )
    return count


def _build_telemetry_ctx(
    mapper: CoordinateMapper,
    fps: float,
    speed_limit: float,
    model_path: Path,
    enable_heatmap: bool,
    csv_exp: TelemetryExporter | None,
    evt_log: EventLogger | None,
) -> _TelemetryContext:
    """Construct and configure all telemetry pipeline components."""
    world_bounds = (mapper.min_world_x, mapper.max_world_x, mapper.min_world_y, mapper.max_world_y)
    return _TelemetryContext(
        detector=OnnxDetector(DetectionConfig(model_path=model_path)),
        tracker=ByteTracker(TrackingConfig()),
        mapper=mapper,
        speed_estimator=SpeedEstimator(fps=fps, speed_limit_kmh=speed_limit),
        bev_renderer=BevRenderer(VisualizationConfig(), world_bounds=world_bounds),
        dashboard=Dashboard(VisualizationConfig()),
        profiler=StageProfiler(),
        annotator=FrameAnnotator(VisualizationConfig(show_trails=True)),
        violation_detector=SpeedViolationDetector(
            AnalyticsConfig(speed_limit_kmh=speed_limit), fps=fps
        ),
        heatmap=SpatialHeatmap(world_bounds=world_bounds) if enable_heatmap else None,
        csv_exporter=csv_exp,
        event_logger=evt_log,
    )


def _cleanup_telemetry(
    writer: cv2.VideoWriter | None,
    show: bool,
    csv_exp: TelemetryExporter | None,
    evt_log: EventLogger | None,
) -> None:
    """Clean up open writers, windows, and exporter file handles."""
    _cleanup_io(writer, show)
    if csv_exp is not None:
        csv_exp.close()
    if evt_log is not None:
        evt_log.close()


def _init_writer(
    output_path: Path | None,
    fps: float,
    dimensions: tuple[int, int],
) -> cv2.VideoWriter | None:
    """Helper to initialize VideoWriter if output path specified."""
    if output_path is None:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    return cv2.VideoWriter(str(output_path), fourcc, fps, dimensions)


def _show_frame(window_name: str, frame: NDArray[np.uint8]) -> bool:
    """Helper to display frame and check for quit key 'q'."""
    cv2.imshow(window_name, frame)
    return bool((cv2.waitKey(1) & 0xFF) == ord("q"))


def _cleanup_io(writer: cv2.VideoWriter | None, show: bool) -> None:
    """Helper to release writer and OpenCV display windows."""
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app()
