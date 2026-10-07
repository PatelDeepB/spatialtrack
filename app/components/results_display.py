"""Streamlit display component for metrics, event tables, and download buttons."""

import pandas as pd
import streamlit as st

from spatialtrack.core.types import SpatialEvent


def render_header() -> None:
    """Render top page header with project badges and subtitle."""
    st.title("SpatialTrack - Spatial Telemetry & Speed Estimation")
    st.markdown(
        "**Real-time multi-object tracking, 4-point homography metric speed estimation, "
        "and spatial analytics engine running strictly on CPU.**"
    )
    st.markdown("---")


def render_metrics(
    fps: float,
    active_tracks: int,
    speed_violations: int,
    mean_latency_ms: float,
) -> None:
    """Render 4 KPI metric cards across page columns."""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Pipeline FPS", f"{fps:.1f}")
    col2.metric("Active Tracks", str(active_tracks))
    col3.metric("Speed Violations", str(speed_violations))
    col4.metric("Mean Latency", f"{mean_latency_ms:.1f} ms")


def render_event_table(events: list[SpatialEvent]) -> None:
    """Render structured table of detected spatial and security events."""
    st.subheader("Detected Spatial Events")
    if not events:
        st.info("No spatial events recorded yet.")
        return

    data = [
        {
            "Event": e.event_type.value,
            "Track ID": int(e.track_id),
            "Frame": int(e.frame_index),
            "Timestamp (s)": f"{e.timestamp_sec:.2f}",
            "Position (m)": f"({e.world_position[0]:.1f}, {e.world_position[1]:.1f})",
            "Speed": f"{e.metadata.get('speed_kmh', 0.0)} km/h",
            "Excess": f"{e.metadata.get('excess_kmh', 0.0)} km/h",
        }
        for e in events
    ]
    st.dataframe(pd.DataFrame(data), use_container_width=True)


def render_download_section(
    video_bytes: bytes | None,
    csv_data: str | None,
    jsonl_data: str | None,
) -> None:
    """Render export download buttons and processed video playback."""
    if video_bytes:
        st.subheader("Processed Video Playback")
        st.video(video_bytes)

    st.subheader("Export Artifacts")
    c1, c2, c3 = st.columns(3)

    if video_bytes:
        c1.download_button(
            label="Download Output Video (.mp4)",
            data=video_bytes,
            file_name="spatialtrack_output.mp4",
            mime="video/mp4",
        )
    if csv_data:
        c2.download_button(
            label="Download Telemetry (.csv)",
            data=csv_data,
            file_name="spatialtrack_telemetry.csv",
            mime="text/csv",
        )
    if jsonl_data:
        c3.download_button(
            label="Download Events (.jsonl)",
            data=jsonl_data,
            file_name="spatialtrack_events.jsonl",
            mime="application/json",
        )
