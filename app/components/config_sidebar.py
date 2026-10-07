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

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent



def render_sidebar() -> tuple[SpatialTrackConfig, str, int | None]:
    """Render interactive sidebar controls and return configured SpatialTrackConfig.

    Returns:
        Tuple of (SpatialTrackConfig, video_source_path, max_frames).
    """
    st.sidebar.title("SpatialTrack Controls")
    video_source = _render_source_selector()
    model_path = _render_model_selector()
    det_conf, num_threads = _render_detection_controls()
    speed_limit, enable_heatmap = _render_analytics_controls()
    max_frames = _render_frame_limit_controls()
    calib_path = _render_calibration_selector(video_source)

    track_high = max(0.20, min(0.40, det_conf))
    config = SpatialTrackConfig(
        detection=DetectionConfig(
            model_path=model_path,
            confidence_threshold=det_conf,
            num_threads=num_threads,
        ),
        tracking=TrackingConfig(
            high_threshold=track_high,
            low_threshold=0.10,
            min_hits=2,
            max_age=30,
        ),
        projection=ProjectionConfig(calibration_path=calib_path),
        analytics=AnalyticsConfig(
            speed_limit_kmh=speed_limit,
            enable_heatmap=enable_heatmap,
        ),
        visualization=VisualizationConfig(),
        output=OutputConfig(),
        video_source=video_source,
    )

    return config, video_source, max_frames


def _render_model_selector() -> Path:
    """Render detection model architecture and precision selector."""
    st.sidebar.subheader("Detection Model")
    model_choice = st.sidebar.selectbox(
        "Model Weights:",
        [
            "YOLOv10n FP32 (High Accuracy - Recommended)",
            "YOLOv10n INT8 (Quantized)",
        ],
        index=0,
    )
    if "INT8" in model_choice:
        return _REPO_ROOT / "models" / "yolov10n_int8.onnx"
    return _REPO_ROOT / "models" / "yolov10n.onnx"


def _render_frame_limit_controls() -> int | None:
    """Render video processing duration controls."""
    st.sidebar.subheader("Processing Duration")
    process_all = st.sidebar.checkbox("Process Entire Video", value=False)
    if process_all:
        return None
    return st.sidebar.slider(
        "Max Frames to Process", min_value=30, max_value=900, value=150, step=30
    )


def _render_calibration_selector(video_source: str) -> Path | None:
    """Render calibration options for perspective projection."""
    sample_calib = _REPO_ROOT / "app" / "assets" / "sample_calibration.json"
    is_demo = "sample_traffic.mp4" in video_source

    if is_demo:
        return sample_calib if sample_calib.is_file() else None

    st.sidebar.subheader("Camera Calibration")
    calib_choice = st.sidebar.radio(
        "Perspective Mapping:",
        ["Highway Sample Grid", "Uncalibrated (Camera-Plane Only)"],
        index=0,
    )
    if calib_choice == "Highway Sample Grid" and sample_calib.is_file():
        return sample_calib
    return None


def _render_source_selector() -> str:
    """Render video source radio and file uploader."""
    st.sidebar.subheader("Video Source")
    source_type = st.sidebar.radio("Select Input:", ["Demo Clip (Included)", "Upload Video File"])

    sample_path = _REPO_ROOT / "app" / "assets" / "sample_traffic.mp4"
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
    st.sidebar.subheader("Detection Settings")
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
