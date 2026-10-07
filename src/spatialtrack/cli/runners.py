"""Execution runners and display loop helpers for CLI commands."""

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from rich.console import Console

from spatialtrack.core.config import DetectionConfig, TrackingConfig, VisualizationConfig
from spatialtrack.core.types import FrameIndex
from spatialtrack.detection.onnx_detector import OnnxDetector
from spatialtrack.io.video_reader import VideoReader
from spatialtrack.io.video_writer import VideoWriter
from spatialtrack.pipeline.engine import FrameResult, PipelineSummary, SpatialTrackEngine
from spatialtrack.pipeline.profiler import StageProfiler
from spatialtrack.tracking.bytetrack import ByteTracker
from spatialtrack.visualization.annotator import FrameAnnotator

console = Console()


def execute_engine_with_display(
    engine: SpatialTrackEngine,
    source: str,
    max_frames: int | None,
    show: bool,
    window_title: str,
) -> PipelineSummary:
    """Orchestrate engine execution with optional OpenCV GUI display window."""

    def on_frame(res: FrameResult) -> None:
        if show and show_frame(window_title, res.annotated_frame):
            raise KeyboardInterrupt("Interrupted by user")

    callback: Callable[[FrameResult], None] | None = on_frame if show else None
    try:
        return engine.run(source=source, max_frames=max_frames, on_frame=callback)
    finally:
        if show:
            cv2.destroyAllWindows()


def run_detect_stream(
    source: str,
    model_path: Path,
    conf: float,
    output: Path | None,
    max_frames: int | None,
    show: bool,
) -> None:
    """Run detection loop over video source and report latency metrics."""
    detector = OnnxDetector(DetectionConfig(model_path=model_path, confidence_threshold=conf))
    annotator = FrameAnnotator(VisualizationConfig())
    profiler = StageProfiler()

    with VideoReader(source) as reader:
        writer = (
            VideoWriter(output, reader.fps, (reader.width, reader.height)).open()
            if output
            else None
        )
        try:
            frames = _loop_detect(reader, detector, annotator, profiler, writer, max_frames, show)
        finally:
            cleanup_io(writer, show)

    console.print(f"\n[bold green]Finished processing {frames} frames.[/bold green]")
    profiler.print_summary_table(console)


def run_track_stream(
    source: str,
    model_path: Path,
    conf: float,
    output: Path | None,
    max_frames: int | None,
    show: bool,
) -> None:
    """Run multi-object tracking loop over video stream."""
    detector = OnnxDetector(DetectionConfig(model_path=model_path, confidence_threshold=conf))
    tracker = ByteTracker(TrackingConfig())
    annotator = FrameAnnotator(VisualizationConfig(show_trails=True))
    profiler = StageProfiler()

    with VideoReader(source) as reader:
        writer = (
            VideoWriter(output, reader.fps, (reader.width, reader.height)).open()
            if output
            else None
        )
        try:
            frames = _loop_track(
                reader, detector, tracker, annotator, profiler, writer, max_frames, show
            )
        finally:
            cleanup_io(writer, show)

    console.print(f"\n[bold green]Finished tracking {frames} frames.[/bold green]")
    profiler.print_summary_table(console)


def _loop_detect(
    reader: VideoReader,
    detector: OnnxDetector,
    annotator: FrameAnnotator,
    profiler: StageProfiler,
    writer: VideoWriter | None,
    max_frames: int | None,
    show: bool,
) -> int:
    """Execute per-frame detection iterations."""
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
        if show and show_frame("SpatialTrack - Detection", annotated):
            break
        count += 1
    return count


def _loop_track(
    reader: VideoReader,
    detector: OnnxDetector,
    tracker: ByteTracker,
    annotator: FrameAnnotator,
    profiler: StageProfiler,
    writer: VideoWriter | None,
    max_frames: int | None,
    show: bool,
) -> int:
    """Execute per-frame tracking iterations."""
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
        if show and show_frame("SpatialTrack - Tracking", annotated):
            break
        count += 1
    return count


def show_frame(window_name: str, frame: NDArray[np.uint8]) -> bool:
    """Display frame in OpenCV window and check for quit key 'q'."""
    cv2.imshow(window_name, frame)
    return bool((cv2.waitKey(1) & 0xFF) == ord("q"))


def cleanup_io(writer: VideoWriter | None, show: bool) -> None:
    """Release writer handle and close OpenCV display windows."""
    if writer is not None:
        writer.close()
    if show:
        cv2.destroyAllWindows()
