"""Unit tests for the stage profiler."""

import time

from spatialtrack.pipeline.profiler import StageProfiler


def test_profiler_measures_stage_latency() -> None:
    """Verify that profiler accurately measures block duration and updates breakdown."""
    # Arrange
    profiler = StageProfiler(window_size=10)

    # Act
    with profiler.measure("inference"):
        time.sleep(0.01)  # Sleep ~10ms

    breakdown = profiler.get_current_breakdown()

    # Assert
    assert breakdown.inference_ms >= 8.0  # At least ~10ms with margin
    assert breakdown.total_ms == breakdown.inference_ms
    assert breakdown.estimated_fps > 0.0
