"""Unit tests for benchmark CLI module."""

from pathlib import Path

from spatialtrack.cli.benchmark_cli import run_benchmark_cli


def test_run_benchmark_cli_executes_successfully() -> None:
    """Verify benchmark runs without error and prints results."""
    # Arrange
    video_path = Path("app/assets/sample_traffic.mp4")
    model_path = Path("models/yolov10n_int8.onnx")

    # Act & Assert (should complete cleanly without raising)
    run_benchmark_cli(
        source=str(video_path),
        model_path=model_path,
        num_threads=2,
        warmup_frames=2,
        benchmark_frames=5,
    )
