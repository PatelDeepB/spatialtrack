"""Main CLI application using Typer and Rich."""

from pathlib import Path

import typer
from rich.console import Console

from spatialtrack.cli.benchmark_cli import run_benchmark_cli
from spatialtrack.cli.calibration_cli import run_calibration_wizard
from spatialtrack.cli.runners import (
    execute_engine_with_display,
    run_detect_stream,
    run_track_stream,
)
from spatialtrack.core.config import (
    AnalyticsConfig,
    DetectionConfig,
    OutputConfig,
    ProjectionConfig,
    SpatialTrackConfig,
)
from spatialtrack.pipeline.engine import SpatialTrackEngine

app = typer.Typer(
    name="spatialtrack",
    help="CPU-optimized spatial telemetry and speed estimation engine.",
    add_completion=False,
)
console = Console()


@app.command()
def run(
    source: str = typer.Option(..., "--source", "-s", help="Path to video file or webcam index."),
    config: Path | None = typer.Option(
        None, "--config", "-c", help="Path to YAML configuration file."
    ),
    model_path: Path = typer.Option(Path("models/yolov10n_int8.onnx"), "--model", "-m"),
    calibration: Path | None = typer.Option(
        None, "--calibration", help="Path to calibration JSON."
    ),
    output: Path | None = typer.Option(None, "--output", "-o", help="Optional output video path."),
    export_csv: Path | None = typer.Option(
        None, "--export-csv", help="Export per-frame CSV telemetry."
    ),
    export_events: Path | None = typer.Option(
        None, "--export-events", help="Export JSONL event logs."
    ),
    speed_limit: float = typer.Option(60.0, "--speed-limit", help="Speed limit in km/h."),
    heatmap: bool = typer.Option(False, "--heatmap", help="Enable 2D density heatmap."),
    max_frames: int | None = typer.Option(None, "--max-frames", help="Maximum frames to process."),
    show: bool = typer.Option(False, "--show", help="Display visual playback window."),
) -> None:
    """Run full spatial tracking engine with configured outputs and profiling."""
    console.print("[bold cyan]SpatialTrack[/bold cyan] - Executing Spatial Engine")
    cfg = (
        SpatialTrackConfig.from_yaml(config)
        if config
        else SpatialTrackConfig(
            detection=DetectionConfig(model_path=model_path),
            projection=ProjectionConfig(calibration_path=calibration),
            analytics=AnalyticsConfig(speed_limit_kmh=speed_limit, enable_heatmap=heatmap),
            output=OutputConfig(
                output_video_path=output,
                telemetry_csv_path=export_csv,
                event_log_path=export_events,
            ),
            video_source=source,
        )
    )
    engine = SpatialTrackEngine(cfg)
    summary = execute_engine_with_display(engine, source, max_frames, show, "SpatialTrack Engine")
    console.print(
        f"\n[bold green]Finished {summary.total_frames} frames "
        f"({summary.average_fps:.1f} FPS)[/bold green]"
    )
    engine.profiler.print_summary_table(console)


@app.command()
def detect(
    source: str = typer.Option(..., "--source", "-s", help="Path to video file or webcam index."),
    model_path: Path = typer.Option(Path("models/yolov10n_int8.onnx"), "--model", "-m"),
    conf: float = typer.Option(0.25, "--conf", "-c", help="Confidence threshold."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Optional output video path."),
    max_frames: int | None = typer.Option(None, "--max-frames", help="Maximum frames to process."),
    show: bool = typer.Option(False, "--show", help="Display visual playback window."),
) -> None:
    """Run CPU-optimized object detection on video stream."""
    console.print("[bold cyan]SpatialTrack[/bold cyan] - Starting detection pipeline on CPU")
    run_detect_stream(source, model_path, conf, output, max_frames, show)


@app.command()
def track(
    source: str = typer.Option(..., "--source", "-s", help="Path to video file or webcam index."),
    model_path: Path = typer.Option(Path("models/yolov10n_int8.onnx"), "--model", "-m"),
    conf: float = typer.Option(0.25, "--conf", "-c", help="Confidence threshold."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Optional output video path."),
    max_frames: int | None = typer.Option(None, "--max-frames", help="Maximum frames to process."),
    show: bool = typer.Option(False, "--show", help="Display visual playback window."),
) -> None:
    """Run CPU-optimized multi-object tracking (ByteTrack) with persistent trails."""
    console.print("[bold cyan]SpatialTrack[/bold cyan] - Starting ByteTrack pipeline on CPU")
    run_track_stream(source, model_path, conf, output, max_frames, show)


@app.command()
def telemetry(
    source: str = typer.Option(..., "--source", "-s", help="Path to video file or webcam index."),
    calibration: Path = typer.Option(
        Path("app/assets/sample_calibration.json"), "--calibration", "-c"
    ),
    model_path: Path = typer.Option(Path("models/yolov10n_int8.onnx"), "--model", "-m"),
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
    cfg = SpatialTrackConfig(
        detection=DetectionConfig(model_path=model_path),
        projection=ProjectionConfig(calibration_path=calibration),
        analytics=AnalyticsConfig(speed_limit_kmh=speed_limit, enable_heatmap=heatmap),
        output=OutputConfig(
            output_video_path=output,
            telemetry_csv_path=export_csv,
            event_log_path=export_events,
        ),
        video_source=source,
    )
    engine = SpatialTrackEngine(cfg)
    summary = execute_engine_with_display(
        engine, source, max_frames, show, "SpatialTrack Telemetry"
    )
    console.print(
        f"\n[bold green]Finished telemetry on {summary.total_frames} frames.[/bold green]"
    )
    engine.profiler.print_summary_table(console)


@app.command()
def calibrate(
    source: str = typer.Option(..., "--source", "-s", help="Path to video or calibration image."),
    output: Path = typer.Option(Path("configs/my_calibration.json"), "--output", "-o"),
    src_points: str | None = typer.Option(
        None, "--src-points", help="Semi-colon separated pixel points."
    ),
    dst_points: str | None = typer.Option(
        None, "--dst-points", help="Semi-colon separated world points (meters)."
    ),
    description: str = typer.Option("Standard 4-point ground calibration", "--description"),
) -> None:
    """Interactive camera ground calibration wizard and homography solver."""
    run_calibration_wizard(source, output, src_points, dst_points, description)


@app.command()
def benchmark(
    source: str = typer.Option(..., "--source", "-s", help="Path to video stream source."),
    model_path: Path = typer.Option(Path("models/yolov10n_int8.onnx"), "--model", "-m"),
    calibration: Path | None = typer.Option(None, "--calibration", "-c"),
    threads: int = typer.Option(2, "--threads", "-t", help="CPU intra-op execution threads."),
    frames: int = typer.Option(30, "--frames", "-n", help="Measured evaluation iterations."),
) -> None:
    """Run CPU performance benchmark measuring stage-by-stage latencies."""
    run_benchmark_cli(source, model_path, calibration, threads, benchmark_frames=frames)


@app.command()
def version() -> None:
    """Print package version."""
    from spatialtrack import __version__

    console.print(f"SpatialTrack version: [bold cyan]{__version__}[/bold cyan]")


if __name__ == "__main__":
    app()
