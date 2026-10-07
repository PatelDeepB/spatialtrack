"""Stage latency profiler for real-time CPU performance monitoring."""

import time
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager

import numpy as np
from rich.console import Console
from rich.table import Table

from spatialtrack.core.types import LatencyBreakdown


class StageProfiler:
    """Measures and records per-stage execution latency across pipeline cycles."""

    def __init__(self, window_size: int = 100) -> None:
        """Initialize profiler with rolling history window size.

        Args:
            window_size: Number of most recent measurements kept per stage.
        """
        self._window_size = window_size
        self._measurements: dict[str, list[float]] = defaultdict(list)
        self._current_frame_latencies: dict[str, float] = {}

    @contextmanager
    def measure(self, stage_name: str) -> Generator[None, None, None]:
        """Measure execution time of a code block in milliseconds.

        Args:
            stage_name: Unique identifier for the pipeline stage.
        """
        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            self._current_frame_latencies[stage_name] = duration_ms
            history = self._measurements[stage_name]
            history.append(duration_ms)
            if len(history) > self._window_size:
                history.pop(0)

    def get_current_breakdown(self) -> LatencyBreakdown:
        """Construct LatencyBreakdown snapshot for the latest measured frame."""
        lat = self._current_frame_latencies
        return LatencyBreakdown(
            decode_ms=lat.get("decode", 0.0),
            preprocess_ms=lat.get("preprocess", 0.0),
            inference_ms=lat.get("inference", 0.0),
            postprocess_ms=lat.get("postprocess", 0.0),
            tracking_ms=lat.get("tracking", 0.0),
            projection_ms=lat.get("projection", 0.0),
            analytics_ms=lat.get("analytics", 0.0),
            visualization_ms=lat.get("visualization", 0.0),
        )

    def print_summary_table(self, console: Console | None = None) -> None:
        """Print formatted statistical breakdown table to rich console."""
        target_console = console or Console()
        table = Table(title="Pipeline Stage Latency Profiler (CPU)", title_style="bold cyan")
        table.add_column("Stage", style="white", justify="left")
        table.add_column("Mean (ms)", style="green", justify="right")
        table.add_column("P95 (ms)", style="yellow", justify="right")
        table.add_column("Min (ms)", style="dim", justify="right")
        table.add_column("Max (ms)", style="red", justify="right")

        total_mean_ms = 0.0
        for stage, durations in sorted(self._measurements.items()):
            if not durations:
                continue
            mean_val = float(np.mean(durations))
            p95_val = float(np.percentile(durations, 95))
            min_val = float(np.min(durations))
            max_val = float(np.max(durations))
            total_mean_ms += mean_val

            table.add_row(
                stage,
                f"{mean_val:.2f}",
                f"{p95_val:.2f}",
                f"{min_val:.2f}",
                f"{max_val:.2f}",
            )

        target_console.print(table)
        estimated_fps = 1000.0 / total_mean_ms if total_mean_ms > 0 else 0.0
        target_console.print(
            f"[bold]Total Mean Latency:[/bold] {total_mean_ms:.2f} ms | "
            f"[bold green]Throughput:[/bold green] {estimated_fps:.1f} FPS"
        )
