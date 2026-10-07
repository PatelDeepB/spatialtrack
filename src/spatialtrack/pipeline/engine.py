"""Unified pipeline orchestration engine executing end-to-end telemetry workflows."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from spatialtrack.analytics.event_detector import SpeedViolationDetector
from spatialtrack.analytics.heatmap import SpatialHeatmap
from spatialtrack.analytics.speed_estimator import SpeedEstimator
from spatialtrack.core.config import SpatialTrackConfig
from spatialtrack.core.exceptions import VideoSourceError
from spatialtrack.core.types import (
    Detection,
    FrameIndex,
    LatencyBreakdown,
    SpatialEvent,
    SpatialRecord,
    Track,
    TrackId,
    WorldCoord,
)
from spatialtrack.detection.onnx_detector import OnnxDetector
from spatialtrack.io.event_logger import EventLogger
from spatialtrack.io.telemetry_exporter import TelemetryExporter
from spatialtrack.io.video_reader import VideoReader
from spatialtrack.io.video_writer import VideoWriter
from spatialtrack.pipeline.profiler import StageProfiler
from spatialtrack.projection.calibration import load_calibration
from spatialtrack.projection.coordinate_mapper import CoordinateMapper
from spatialtrack.tracking.bytetrack import ByteTracker
from spatialtrack.visualization.annotator import FrameAnnotator
from spatialtrack.visualization.bev_renderer import BevRenderer
from spatialtrack.visualization.dashboard import Dashboard


@dataclass(slots=True)
class FrameResult:
    """Per-frame execution artifacts produced by SpatialTrackEngine."""

    frame_index: FrameIndex
    annotated_frame: NDArray[np.uint8]
    detections: list[Detection]
    tracks: list[Track]
    spatial_records: list[SpatialRecord]
    events: list[SpatialEvent]
    latency_breakdown: LatencyBreakdown


@dataclass(slots=True)
class PipelineSummary:
    """Execution summary statistics produced upon completion of video processing."""

    total_frames: int
    unique_tracks: int
    total_events: int
    average_fps: float
    mean_latency_ms: float


class SpatialTrackEngine:
    """End-to-end spatial tracking engine coordinating CV detection, tracking, and analytics."""

    def __init__(self, config: SpatialTrackConfig) -> None:
        """Initialize pipeline engine and instantiate modular sub-components.

        Args:
            config: SpatialTrackConfig container holding model, tracking, and visual options.
        """
        self.config = config
        self.detector = OnnxDetector(config.detection)
        self.tracker = ByteTracker(config.tracking)
        self.annotator = FrameAnnotator(config.visualization)
        self.dashboard = Dashboard(config.visualization)
        self.profiler = StageProfiler()

        self.mapper: CoordinateMapper | None = None
        self.bev_renderer: BevRenderer | None = None
        self.heatmap: SpatialHeatmap | None = None
        self.speed_estimator: SpeedEstimator | None = None
        self.violation_detector: SpeedViolationDetector | None = None

        self._init_spatial_modules()

    def _init_spatial_modules(self) -> None:
        """Initialize homography mapper, BEV renderer, and heatmap if calibration is provided."""
        calib_path = self.config.projection.calibration_path
        if calib_path is not None and Path(calib_path).is_file():
            calib_data = load_calibration(calib_path)
            self.mapper = CoordinateMapper(calib_data)
            bounds = (
                self.mapper.min_world_x,
                self.mapper.max_world_x,
                self.mapper.min_world_y,
                self.mapper.max_world_y,
            )
            self.bev_renderer = BevRenderer(self.config.visualization, world_bounds=bounds)
            if self.config.analytics.enable_heatmap:
                self.heatmap = SpatialHeatmap(
                    world_bounds=bounds,
                    grid_resolution=self.config.analytics.heatmap_resolution,
                    sigma_cells=self.config.analytics.heatmap_sigma,
                )

    def setup_fps(self, fps: float) -> None:
        """Configure frame rate for speed calculation and violation debouncing."""
        valid_fps = max(1.0, float(fps))
        self.speed_estimator = SpeedEstimator(
            fps=valid_fps,
            speed_limit_kmh=self.config.analytics.speed_limit_kmh,
            smoothing_alpha=self.config.projection.speed_smoothing_alpha,
            min_displacement_m=self.config.projection.min_displacement_m,
        )
        self.violation_detector = SpeedViolationDetector(self.config.analytics, fps=valid_fps)

    def process_frame(self, frame: NDArray[np.uint8], frame_index: FrameIndex) -> FrameResult:
        """Execute a full processing cycle for a single frame.

        Args:
            frame: Raw BGR input frame array (H, W, 3).
            frame_index: Zero-based frame counter index.

        Returns:
            FrameResult containing annotated image, tracks, records, and performance metrics.
        """
        detections = self._run_detection(frame, frame_index)
        tracks = self._run_tracking(detections, frame_index)
        records, trails = self._run_projection(tracks, frame_index)
        events = self._run_analytics(records)
        output_frame = self._run_visualization(frame, tracks, records, trails)

        return FrameResult(
            frame_index=frame_index,
            annotated_frame=output_frame,
            detections=detections,
            tracks=tracks,
            spatial_records=records,
            events=events,
            latency_breakdown=self.profiler.get_current_breakdown(),
        )

    def process_stream(
        self,
        reader: VideoReader,
        max_frames: int | None = None,
    ) -> Iterator[FrameResult]:
        """Generator yielding FrameResult for each decoded frame from video reader.

        Args:
            reader: Open VideoReader instance.
            max_frames: Optional upper limit on processed frames.

        Yields:
            FrameResult instances for consecutive frames.
        """
        if self.speed_estimator is None or self.violation_detector is None:
            self.setup_fps(reader.fps)

        count = 0
        for frame in reader:
            if max_frames is not None and count >= max_frames:
                break
            yield self.process_frame(frame, FrameIndex(count))
            count += 1

    def run(
        self,
        source: str | None = None,
        max_frames: int | None = None,
        on_frame: Callable[[FrameResult], None] | None = None,
    ) -> PipelineSummary:
        """Run complete end-to-end video processing with configured input/output sinks.

        Args:
            source: Video source path (overrides config.video_source).
            max_frames: Optional limit on processed frames.
            on_frame: Optional callback invoked for each FrameResult.

        Returns:
            PipelineSummary with aggregated frame, track, event, and speed stats.
        """
        target_source = source or self.config.video_source
        if not target_source:
            raise VideoSourceError("No video source provided to engine.")

        unique_tracks: set[TrackId] = set()
        total_events = 0
        frame_count = 0

        with VideoReader(target_source) as reader:
            self.setup_fps(reader.fps)
            writer, csv_exp, evt_log = self._open_sinks(reader)
            try:
                for result in self.process_stream(reader, max_frames=max_frames):
                    frame_count += 1
                    total_events += len(result.events)
                    unique_tracks.update(t.track_id for t in result.tracks)
                    self._write_sinks(result, writer, csv_exp, evt_log)
                    if on_frame is not None:
                        on_frame(result)
            finally:
                self._close_sinks(writer, csv_exp, evt_log)

        mean_lat = self.profiler.get_current_breakdown().total_ms
        avg_fps = 1000.0 / mean_lat if mean_lat > 0.0 else 0.0
        return PipelineSummary(
            total_frames=frame_count,
            unique_tracks=len(unique_tracks),
            total_events=total_events,
            average_fps=avg_fps,
            mean_latency_ms=mean_lat,
        )

    def _run_detection(self, frame: NDArray[np.uint8], frame_index: FrameIndex) -> list[Detection]:
        """Execute detector on frame inside profiler context."""
        with self.profiler.measure("inference"):
            return self.detector.detect(frame, frame_index)

    def _run_tracking(self, detections: list[Detection], frame_index: FrameIndex) -> list[Track]:
        """Update tracker inside profiler context."""
        with self.profiler.measure("tracking"):
            return self.tracker.update(detections, frame_index)

    def _run_projection(
        self, tracks: list[Track], frame_index: FrameIndex
    ) -> tuple[list[SpatialRecord], dict[int, list[WorldCoord]]]:
        """Project tracks to world coordinates and estimate speeds."""
        records: list[SpatialRecord] = []
        trails: dict[int, list[WorldCoord]] = {}

        if self.mapper is None or self.speed_estimator is None:
            return records, trails

        with self.profiler.measure("projection_speed"):
            for track_obj in tracks:
                world_pos, w_trail = self.mapper.project_track(track_obj)
                record = self.speed_estimator.estimate_speed(track_obj, world_pos, frame_index)
                records.append(record)
                trails[int(track_obj.track_id)] = w_trail

        return records, trails

    def _run_analytics(self, records: list[SpatialRecord]) -> list[SpatialEvent]:
        """Accumulate heatmaps and detect violation events."""
        if self.heatmap is not None and records:
            self.heatmap.accumulate([r.world_position for r in records])

        if self.violation_detector is not None:
            return self.violation_detector.update(records)

        return []

    def _run_visualization(
        self,
        frame: NDArray[np.uint8],
        tracks: list[Track],
        records: list[SpatialRecord],
        trails: dict[int, list[WorldCoord]],
    ) -> NDArray[np.uint8]:
        """Compose visualization output: camera view + BEV map + dashboard banner."""
        with self.profiler.measure("visualization"):
            cam_view = self.annotator.draw_tracks(frame, tracks)
            if self.bev_renderer is None:
                return cam_view

            bev_view = self.bev_renderer.render(records, trails, heatmap=self.heatmap)
            violations = sum(1 for r in records if r.is_speed_violation)
            return self.dashboard.compose(
                cam_view,
                bev_view,
                len(tracks),
                violations,
                self.profiler.get_current_breakdown(),
            )

    def _open_sinks(
        self, reader: VideoReader
    ) -> tuple[VideoWriter | None, TelemetryExporter | None, EventLogger | None]:
        """Initialize configured video, CSV, and event output file sinks."""
        writer: VideoWriter | None = None
        csv_exp: TelemetryExporter | None = None
        evt_log: EventLogger | None = None

        if self.config.output.output_video_path is not None:
            dim_w = reader.width + (400 if self.bev_renderer else 0)
            dim_h = reader.height + (42 if self.bev_renderer else 0)
            writer = VideoWriter(
                self.config.output.output_video_path,
                fps=self.config.output.video_fps or reader.fps,
                frame_size=(dim_w, dim_h),
                codec=self.config.output.video_codec,
            ).open()

        if self.config.output.telemetry_csv_path is not None:
            csv_exp = TelemetryExporter(self.config.output.telemetry_csv_path).open()

        if self.config.output.event_log_path is not None:
            evt_log = EventLogger(self.config.output.event_log_path).open()

        return writer, csv_exp, evt_log

    def _write_sinks(
        self,
        result: FrameResult,
        writer: VideoWriter | None,
        csv_exp: TelemetryExporter | None,
        evt_log: EventLogger | None,
    ) -> None:
        """Write frame results to active sinks."""
        if writer is not None:
            writer.write(result.annotated_frame)
        if csv_exp is not None and result.spatial_records:
            csv_exp.export_records(result.spatial_records)
        if evt_log is not None and result.events:
            evt_log.log_events(result.events)

    def _close_sinks(
        self,
        writer: VideoWriter | None,
        csv_exp: TelemetryExporter | None,
        evt_log: EventLogger | None,
    ) -> None:
        """Flush and release all sink resources."""
        if writer is not None:
            writer.close()
        if csv_exp is not None:
            csv_exp.close()
        if evt_log is not None:
            evt_log.close()
