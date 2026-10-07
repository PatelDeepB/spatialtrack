"""Unit tests for Streamlit application helper logic."""

from unittest.mock import MagicMock, patch

from app.components.config_sidebar import _render_analytics_controls, _render_detection_controls
from app.components.results_display import render_header, render_metrics


def test_render_detection_controls() -> None:
    """Verify detection controls return valid slider values."""
    # Arrange & Act
    with patch("streamlit.sidebar.slider", side_effect=[0.35, 4]):
        conf, threads = _render_detection_controls()

    # Assert
    assert conf == 0.35
    assert threads == 4


def test_render_analytics_controls() -> None:
    """Verify analytics controls return speed limit and heatmap toggle."""
    # Arrange & Act
    with (
        patch("streamlit.sidebar.slider", return_value=50.0),
        patch("streamlit.sidebar.checkbox", return_value=True),
    ):
        speed_limit, enable_heatmap = _render_analytics_controls()

    # Assert
    assert speed_limit == 50.0
    assert enable_heatmap is True


def test_render_metrics_calls_metric() -> None:
    """Verify render_metrics populates 4 metrics across layout columns."""
    # Arrange
    mock_col = MagicMock()
    with patch("streamlit.columns", return_value=[mock_col, mock_col, mock_col, mock_col]):
        # Act
        render_metrics(fps=25.0, active_tracks=5, speed_violations=1, mean_latency_ms=40.0)

    # Assert
    assert mock_col.metric.call_count == 4


def test_render_header() -> None:
    """Verify render_header calls streamlit title and markdown."""
    # Arrange & Act
    with patch("streamlit.title") as mock_title, patch("streamlit.markdown") as mock_md:
        render_header()

    # Assert
    assert mock_title.called
    assert mock_md.called
