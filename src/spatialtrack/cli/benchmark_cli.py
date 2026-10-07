"""CLI benchmark module evaluating pipeline latency and throughput on CPU."""

from pathlib import Path

from rich.console import Console
from rich.table import Table

from spatialtrack.core.config import (
    AnalyticsConfig,
    DetectionConfig,
    ProjectionConfig,
    SpatialTrackConfig,
    TrackingConfig,
    VisualizationConfig,
)
from spatialtrack.pipeline.engine import SpatialTrackEngine

console = Console()


def run_benchmark_cli(
    source: str,
    model_path: Path,
    calibration_path: Path | None = None,
    num_threads: int = 2,
    warmup_frames: int = 5,
    benchmark_frames: int = 30,
) -> None:
    """Run standardized CPU pipeline benchmark and display latency breakdown table.

    Args:
        source: Path to video stream source.
        model_path: Path to ONNX model weights.
        calibration_path: Optional path to ground homography calibration file.
        num_threads: Number of CPU intra-op threads.
        warmup_frames: Initial unmeasured iterations.
        benchmark_frames: Measured iterations.
    """
    console.print("[bold cyan]SpatialTrack[/bold cyan] - Running CPU Performance Benchmark")
    console.print(
        f"Threads: {num_threads} | Model: {model_path} | Target Frames: {benchmark_frames}"
    )

    config = SpatialTrackConfig(
        detection=DetectionConfig(model_path=model_path, num_threads=num_threads),
        tracking=TrackingConfig(),
        projection=ProjectionConfig(calibration_path=calibration_path),
        analytics=AnalyticsConfig(),
        visualization=VisualizationConfig(),
        video_source=source,
    )
    engine = SpatialTrackEngine(config)

    # Warmup
    engine.run(source=source, max_frames=warmup_frames)
    engine.profiler.reset()

    # Benchmark
    summary = engine.run(source=source, max_frames=benchmark_frames)
    _print_benchmark_results(engine, summary, num_threads)


def _print_benchmark_results(
    engine: SpatialTrackEngine,
    summary: object,
    num_threads: int,
) -> None:
    """Print formatted Rich latency and throughput table."""
    engine.profiler.print_summary_table(console)

    table = Table(title="Throughput & Efficiency Summary", show_header=True)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    fps = getattr(summary, "average_fps", 0.0)
    lat = getattr(summary, "mean_latency_ms", 0.0)
    frames = getattr(summary, "total_frames", 0)

    table.add_row("CPU Intra-Op Threads", str(num_threads))
    table.add_row("Frames Evaluated", str(frames))
    table.add_row("Pipeline Mean Latency", f"{lat:.2f} ms")
    table.add_row("Estimated Pipeline Throughput", f"{fps:.2f} FPS")

    console.print(table)
