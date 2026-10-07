"""Unit tests for the threaded VideoReader module."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from spatialtrack.core.exceptions import VideoSourceError
from spatialtrack.io.video_reader import VideoReader


def create_synthetic_video(
    video_path: Path, frame_count: int = 10, width: int = 320, height: int = 240
) -> None:
    """Helper to generate a short synthetic MP4 test video."""
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (width, height))
    try:
        for i in range(frame_count):
            frame = np.full((height, width, 3), (i * 10) % 250, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def test_video_reader_invalid_source_raises_error() -> None:
    """Verify that attempting to open a non-existent video path raises VideoSourceError."""
    # Arrange & Act & Assert
    with pytest.raises(VideoSourceError):
        VideoReader("path_that_does_not_exist_12345.mp4")


def test_video_reader_reads_all_frames(tmp_path: Path) -> None:
    """Verify threaded video reader correctly reads all frames in sequence."""
    # Arrange
    test_video = tmp_path / "test_feed.mp4"
    frame_count = 15
    create_synthetic_video(test_video, frame_count=frame_count, width=320, height=240)

    # Act
    read_frames: list[NDArray[np.uint8]] = []
    with VideoReader(test_video) as reader:
        assert reader.width == 320
        assert reader.height == 240
        assert reader.fps > 0.0
        for frame in reader:
            read_frames.append(frame)

    # Assert
    assert len(read_frames) == frame_count
    assert read_frames[0].shape == (240, 320, 3)
