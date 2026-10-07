"""Standardized CPU inference benchmark comparing FP32 vs INT8 models
across thread configurations.
"""

import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from rich.console import Console
from rich.table import Table

console = Console()


def run_benchmark(
    model_path: Path,
    num_threads: int,
    warmup_runs: int = 10,
    benchmark_runs: int = 50,
) -> tuple[float, float, float, float]:
    """Execute benchmark runs and compute latency percentiles in milliseconds.

    Args:
        model_path: Path to ONNX model file.
        num_threads: Number of CPU intra-op execution threads.
        warmup_runs: Number of unmeasured warmup iterations.
        benchmark_runs: Number of timed evaluation iterations.

    Returns:
        Tuple of (mean_ms, p95_ms, min_ms, throughput_fps).
    """
    options = ort.SessionOptions()
    options.intra_op_num_threads = num_threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)

    # Warmup
    for _ in range(warmup_runs):
        session.run(None, {input_name: dummy_input})

    # Timed runs
    durations_ms: list[float] = []
    for _ in range(benchmark_runs):
        start = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        durations_ms.append((time.perf_counter() - start) * 1000.0)

    mean_ms = float(np.mean(durations_ms))
    p95_ms = float(np.percentile(durations_ms, 95))
    min_ms = float(np.min(durations_ms))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    return mean_ms, p95_ms, min_ms, fps


def main() -> None:
    """Run comprehensive CPU benchmark suite across available models and thread counts."""
    models = [
        ("FP32", Path("models/yolov10n.onnx")),
        ("INT8 Quantized", Path("models/yolov10n_int8.onnx")),
    ]
    thread_counts = [2, 4, 8]

    table = Table(title="SpatialTrack CPU Inference Benchmark", title_style="bold green")
    table.add_column("Model Precision", style="white")
    table.add_column("Threads", style="cyan", justify="right")
    table.add_column("Mean Latency (ms)", style="green", justify="right")
    table.add_column("P95 Latency (ms)", style="yellow", justify="right")
    table.add_column("Min Latency (ms)", style="dim", justify="right")
    table.add_column("Throughput (FPS)", style="bold magenta", justify="right")

    console.print("[cyan]Running CPU benchmarks...[/cyan]")
    for label, path in models:
        if not path.is_file():
            console.print(f"[yellow]Skipping {label}: model file not found at {path}[/yellow]")
            continue

        for threads in thread_counts:
            mean_ms, p95_ms, min_ms, fps = run_benchmark(path, threads)
            table.add_row(
                label,
                str(threads),
                f"{mean_ms:.2f}",
                f"{p95_ms:.2f}",
                f"{min_ms:.2f}",
                f"{fps:.1f}",
            )

    console.print(table)


if __name__ == "__main__":
    main()
