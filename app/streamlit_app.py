"""Interactive Streamlit Web Application for SpatialTrack."""

import io
import tempfile
from pathlib import Path

import cv2
import streamlit as st

from app.components.config_sidebar import render_sidebar
from app.components.results_display import (
    render_download_section,
    render_event_table,
    render_header,
    render_metrics,
)
from spatialtrack.core.config import SpatialTrackConfig
from spatialtrack.core.types import SpatialEvent, SpatialRecord
from spatialtrack.io.video_reader import VideoReader
from spatialtrack.io.video_writer import VideoWriter
from spatialtrack.pipeline.engine import SpatialTrackEngine


def main() -> None:
    """Main Streamlit execution entry point."""
    st.set_page_config(
        page_title="SpatialTrack - Spatial Telemetry & Speed Engine",
        layout="wide",
        page_icon="🚗",
    )
    render_header()
    config, video_source, max_frames = render_sidebar()

    if st.button("Run Spatial Telemetry Analysis", type="primary"):
        _run_interactive_pipeline(config, video_source, max_frames)


def _run_interactive_pipeline(
    config: SpatialTrackConfig,
    video_source: str,
    max_frames: int,
) -> None:
    """Execute video processing stream and update interactive Streamlit components."""
    engine = SpatialTrackEngine(config)
    progress_bar = st.progress(0, text="Initializing CPU pipeline...")
    frame_placeholder = st.empty()
    metric_placeholder = st.empty()

    accumulated_events: list[SpatialEvent] = []
    accumulated_records: list[SpatialRecord] = []
    temp_video_path = Path(tempfile.gettempdir()) / "st_spatialtrack_out.mp4"

    with VideoReader(video_source) as reader:
        writer = _create_temp_writer(temp_video_path, reader, engine)
        try:
            frame_idx = 0
            for result in engine.process_stream(reader, max_frames=max_frames):
                frame_idx += 1
                progress_bar.progress(
                    min(1.0, frame_idx / max_frames),
                    text=f"Processing frame {frame_idx} of {max_frames} on CPU...",
                )
                accumulated_events.extend(result.events)
                accumulated_records.extend(result.spatial_records)

                if writer is not None:
                    writer.write(result.annotated_frame)

                rgb_frame = cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                violations = sum(1 for r in result.spatial_records if r.is_speed_violation)
                with metric_placeholder.container():
                    render_metrics(
                        fps=result.latency_breakdown.estimated_fps,
                        active_tracks=len(result.tracks),
                        speed_violations=violations,
                        mean_latency_ms=result.latency_breakdown.total_ms,
                    )
        finally:
            if writer is not None:
                writer.close()

    progress_bar.progress(1.0, text=f"Finished processing {frame_idx} frames!")
    render_event_table(accumulated_events)
    _provide_downloads(temp_video_path, accumulated_records, accumulated_events)


def _create_temp_writer(
    out_path: Path,
    reader: VideoReader,
    engine: SpatialTrackEngine,
) -> VideoWriter:
    """Create temporary VideoWriter matching dashboard dimensions."""
    out_w = reader.width + (400 if engine.bev_renderer else 0)
    out_h = reader.height + (42 if engine.bev_renderer else 0)
    return VideoWriter(out_path, fps=reader.fps, frame_size=(out_w, out_h)).open()


def _provide_downloads(
    video_path: Path,
    records: list[SpatialRecord],
    events: list[SpatialEvent],
) -> None:
    """Prepare download payloads and display download buttons."""
    video_bytes = video_path.read_bytes() if video_path.exists() else None

    # CSV data
    csv_buf = io.StringIO()
    csv_buf.write("frame,track_id,pixel_x,pixel_y,world_x,world_y,speed_kmh,violation\n")
    for r in records:
        csv_buf.write(
            f"{r.frame_index},{r.track_id},{r.pixel_position[0]:.1f},{r.pixel_position[1]:.1f},"
            f"{r.world_position[0]:.2f},{r.world_position[1]:.2f},{r.speed_kmh:.1f},"
            f"{1 if r.is_speed_violation else 0}\n"
        )
    csv_str = csv_buf.getvalue()

    # JSONL events data
    jsonl_lines = [
        f'{{"type": "{e.event_type.value}", "track": {e.track_id}, '
        f'"speed": {e.metadata.get("speed_kmh", 0)}}}'
        for e in events
    ]
    jsonl_str = "\n".join(jsonl_lines)

    render_download_section(video_bytes, csv_str, jsonl_str)


if __name__ == "__main__":
    main()
