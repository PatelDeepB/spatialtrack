"""Unit tests for the context-managed VideoWriter."""

from pathlib import Path

import numpy as np
import pytest

from spatialtrack.core.exceptions import ExportError
from spatialtrack.io.video_reader import VideoReader
from spatialtrack.io.video_writer import VideoWriter


def test_video_writer_invalid_dimensions_raises(tmp_path: Path) -> None:
    """Verify that non-positive dimensions raise ExportError."""
    # Arrange
    out_file = tmp_path / "test.mp4"

    # Act & Assert
    with pytest.raises(ExportError, match="Invalid video frame size"):
        VideoWriter(out_file, fps=30.0, frame_size=(0, 480))


def test_video_writer_creates_and_writes_frames(tmp_path: Path) -> None:
    """Verify that VideoWriter creates a playable video with expected frames."""
    # Arrange
    out_file = tmp_path / "output_test.mp4"
    frame_h, frame_w = 240, 320
    test_frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    test_frame[50:100, 50:100] = (0, 255, 0)  # Green square

    # Act
    with VideoWriter(out_file, fps=25.0, frame_size=(frame_w, frame_h)) as writer:
        for _ in range(10):
            writer.write(test_frame)

    # Assert
    assert out_file.exists()
    assert out_file.stat().st_size > 0
    assert writer.frame_count == 10
    assert not writer.is_opened

    # Verify by reading back
    with VideoReader(out_file) as reader:
        assert reader.width == frame_w
        assert reader.height == frame_h
        assert reader.total_frames == 10


def test_video_writer_auto_resizes_mismatched_frames(tmp_path: Path) -> None:
    """Verify that frames with mismatched dimensions are resized before writing."""
    # Arrange
    out_file = tmp_path / "resized_test.mp4"
    target_w, target_h = 320, 240
    mismatched_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Act
    with VideoWriter(out_file, fps=30.0, frame_size=(target_w, target_h)) as writer:
        writer.write(mismatched_frame)

    # Assert
    assert writer.frame_count == 1
    with VideoReader(out_file) as reader:
        assert reader.width == target_w
        assert reader.height == target_h
