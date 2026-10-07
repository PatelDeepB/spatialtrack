"""Unit tests for calibration CLI wizard."""

import json
from pathlib import Path

import pytest

from spatialtrack.cli.calibration_cli import parse_points_str, run_calibration_wizard
from spatialtrack.core.exceptions import CalibrationError


def test_parse_points_str_valid() -> None:
    """Verify parsing valid semicolon-separated point string."""
    # Arrange
    raw = "100,200; 300,200; 400,500; 50,500"

    # Act
    pts = parse_points_str(raw)

    # Assert
    assert len(pts) == 4
    assert pts[0] == (100.0, 200.0)
    assert pts[1] == (300.0, 200.0)
    assert pts[2] == (400.0, 500.0)
    assert pts[3] == (50.0, 500.0)


def test_parse_points_str_invalid_count_raises() -> None:
    """Verify error raised when fewer or more than 4 points provided."""
    # Arrange & Act & Assert
    with pytest.raises(CalibrationError, match="Expected 4 coordinate pairs"):
        parse_points_str("100,200; 300,200; 400,500")


def test_parse_points_str_invalid_token_raises() -> None:
    """Verify error raised when token format is not x,y."""
    # Arrange & Act & Assert
    with pytest.raises(CalibrationError, match="Invalid coordinate token pair"):
        parse_points_str("100; 300,200; 400,500; 50,500")


def test_run_calibration_wizard_with_points_str(tmp_path: Path) -> None:
    """Verify running calibration wizard saves valid JSON calibration file."""
    # Arrange
    video_path = Path("app/assets/sample_traffic.mp4")
    out_calib = tmp_path / "test_calib.json"
    src_str = "523,412; 758,412; 980,680; 300,680"
    dst_str = "0,0; 12,0; 12,40; 0,40"

    # Act
    run_calibration_wizard(
        source=str(video_path),
        output_path=out_calib,
        src_points_str=src_str,
        dst_points_str=dst_str,
        description="Unit test calibration",
    )

    # Assert
    assert out_calib.exists()
    with open(out_calib, encoding="utf-8") as f:
        data = json.load(f)
    assert len(data["source_points_px"]) == 4
    assert len(data["target_points_m"]) == 4
    assert data["frame_width"] == 1280
    assert data["frame_height"] == 720
    assert data["description"] == "Unit test calibration"
