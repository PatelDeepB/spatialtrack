"""Unit tests for the preprocessing module."""

import numpy as np

from spatialtrack.detection.preprocessing import (
    apply_letterbox,
    calculate_letterbox_geometry,
    preprocess_frame,
)


def test_calculate_letterbox_geometry_widescreen() -> None:
    """Verify geometry calculation for 16:9 widescreen input (1280x720) to 640x640."""
    # Arrange
    height, width = 720, 1280
    target_size = 640

    # Act
    ratio, new_w, new_h, pad_left, pad_top = calculate_letterbox_geometry(
        height, width, target_size
    )

    # Assert
    assert ratio == 0.5
    assert new_w == 640
    assert new_h == 360
    assert pad_left == 0
    assert pad_top == 140  # (640 - 360) // 2


def test_calculate_letterbox_geometry_square() -> None:
    """Verify geometry calculation for square input without padding needed."""
    # Arrange
    height, width = 640, 640
    target_size = 640

    # Act
    ratio, new_w, new_h, pad_left, pad_top = calculate_letterbox_geometry(
        height, width, target_size
    )

    # Assert
    assert ratio == 1.0
    assert new_w == 640
    assert new_h == 640
    assert pad_left == 0
    assert pad_top == 0


def test_apply_letterbox_canvas_shape() -> None:
    """Verify output canvas dimensions from letterbox application."""
    # Arrange
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)

    # Act
    canvas, metadata = apply_letterbox(dummy_image, target_size=640)

    # Assert
    assert canvas.shape == (640, 640, 3)
    assert metadata.original_height == 480
    assert metadata.original_width == 640


def test_preprocess_frame_tensor_format() -> None:
    """Verify final preprocessed tensor shape, dtype, and normalized value bounds."""
    # Arrange
    sample_frame = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)

    # Act
    tensor, metadata = preprocess_frame(sample_frame, target_size=640)

    # Assert
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0
    assert metadata.original_width == 1280
    assert metadata.original_height == 720
