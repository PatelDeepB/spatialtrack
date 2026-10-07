"""Sidebar control component for configuring SpatialTrack parameters in Streamlit."""

import tempfile
from pathlib import Path

import streamlit as st

from spatialtrack.core.config import (
    AnalyticsConfig,
    DetectionConfig,
    OutputConfig,
    ProjectionConfig,
    SpatialTrackConfig,
    TrackingConfig,
    VisualizationConfig,
)


def render_sidebar() -> tuple[SpatialTrackConfig, str, int]:
    """Render interactive sidebar controls and return configured SpatialTrackConfig.

    Returns:
        Tuple of (SpatialTrackConfig, video_source_path, max_frames).
    """
    st.sidebar.title("SpatialTrack Controls")
    video_source = _render_source_selector()
    det_conf, num_threads = _render_detection_controls()
    speed_limit, enable_heatmap = _render_analytics_controls()
    max_frames = st.sidebar.slider(
        "Max Frames to Process", min_value=10, max_value=150, value=60, step=10
    )

    calib_path = Path("app/assets/sample_calibration.json")
    model_path = Path("models/yolov10n_int8.onnx")

    config = SpatialTrackConfig(
        detection=DetectionConfig(
            model_path=model_path,
            confidence_threshold=det_conf,
            num_threads=num_threads,
        ),
        tracking=TrackingConfig(),
        projection=ProjectionConfig(calibration_path=calib_path if calib_path.is_file() else None),
        analytics=AnalyticsConfig(
            speed_limit_kmh=speed_limit,
            enable_heatmap=enable_heatmap,
        ),
        visualization=VisualizationConfig(),
        output=OutputConfig(),
        video_source=video_source,
    )

    return config, video_source, max_frames


def _render_source_selector() -> str:
    """Render video source radio and file uploader."""
    st.sidebar.subheader("Video Source")
    source_type = st.sidebar.radio("Select Input:", ["Demo Clip (Included)", "Upload Video File"])

    sample_path = Path("app/assets/sample_traffic.mp4")
    if source_type == "Demo Clip (Included)":
        return str(sample_path)

    uploaded_file = st.sidebar.file_uploader("Upload MP4 Video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        temp_dir = tempfile.gettempdir()
        temp_path = Path(temp_dir) / f"upload_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return str(temp_path)

    return str(sample_path)


def _render_detection_controls() -> tuple[float, int]:
    """Render detection threshold and thread sliders."""
    st.sidebar.subheader("Detection (INT8 ONNX)")
    conf = st.sidebar.slider(
        "Confidence Threshold", min_value=0.10, max_value=0.90, value=0.25, step=0.05
    )
    threads = st.sidebar.slider("CPU Threads", min_value=1, max_value=8, value=2, step=1)
    return conf, threads


def _render_analytics_controls() -> tuple[float, bool]:
    """Render speed limit and heatmap controls."""
    st.sidebar.subheader("Spatial Analytics")
    speed_limit = st.sidebar.slider(
        "Speed Limit (km/h)", min_value=10.0, max_value=120.0, value=60.0, step=5.0
    )
    enable_heatmap = st.sidebar.checkbox("Enable 2D Density Heatmap", value=True)
    return speed_limit, enable_heatmap
