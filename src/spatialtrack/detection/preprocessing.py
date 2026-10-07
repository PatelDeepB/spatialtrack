"""Preprocessing utilities including aspect-preserving letterbox resizing and tensor conversion."""

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from spatialtrack.core.constants import DEFAULT_INPUT_SIZE, LETTERBOX_FILL_COLOR


@dataclass(frozen=True, slots=True)
class PreprocessMetadata:
    """Metadata required to invert letterbox transformations during post-processing."""

    scale_ratio: float
    pad_left: float
    pad_top: float
    original_height: int
    original_width: int


def calculate_letterbox_geometry(
    original_height: int,
    original_width: int,
    target_size: int,
) -> tuple[float, int, int, int, int]:
    """Calculate scaling ratio, scaled dimensions, and centered padding offsets.

    Args:
        original_height: Source image height in pixels.
        original_width: Source image width in pixels.
        target_size: Target square dimension in pixels.

    Returns:
        Tuple containing (scale_ratio, new_width, new_height, pad_left, pad_top).
    """
    scale_ratio = min(target_size / original_height, target_size / original_width)
    new_width = int(round(original_width * scale_ratio))
    new_height = int(round(original_height * scale_ratio))

    pad_horizontal = target_size - new_width
    pad_vertical = target_size - new_height

    pad_left = pad_horizontal // 2
    pad_top = pad_vertical // 2

    return scale_ratio, new_width, new_height, pad_left, pad_top


def apply_letterbox(
    image: NDArray[np.uint8],
    target_size: int = DEFAULT_INPUT_SIZE,
) -> tuple[NDArray[np.uint8], PreprocessMetadata]:
    """Resize image to target square dimensions preserving aspect ratio with centered padding.

    Args:
        image: Input image array of shape (height, width, channels).
        target_size: Output square canvas size.

    Returns:
        Tuple of (padded_image, metadata).
    """
    orig_h, orig_w = image.shape[:2]
    ratio, new_w, new_h, pad_left, pad_top = calculate_letterbox_geometry(
        orig_h, orig_w, target_size
    )

    if (orig_w, orig_h) != (new_w, new_h):
        resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized_img = image.copy()

    canvas = np.full(
        (target_size, target_size, 3),
        LETTERBOX_FILL_COLOR,
        dtype=np.uint8,
    )
    canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized_img

    metadata = PreprocessMetadata(
        scale_ratio=ratio,
        pad_left=float(pad_left),
        pad_top=float(pad_top),
        original_height=orig_h,
        original_width=orig_w,
    )
    return canvas, metadata


def preprocess_frame(
    frame: NDArray[np.uint8],
    target_size: int = DEFAULT_INPUT_SIZE,
) -> tuple[NDArray[np.float32], PreprocessMetadata]:
    """Execute preprocessing: letterbox, BGR to RGB, normalize, and transpose to NCHW.

    Args:
        frame: BGR frame array from video capture.
        target_size: Target square model input resolution.

    Returns:
        Tuple of batch tensor of shape (1, 3, target_size, target_size) and metadata.
    """
    canvas, metadata = apply_letterbox(frame, target_size=target_size)

    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    # Convert to float32 and normalize [0, 1]
    normalized_image = rgb_image.astype(np.float32) / 255.0

    # Transpose HWC -> CHW and add batch dimension -> (1, C, H, W)
    chw_tensor = np.transpose(normalized_image, (2, 0, 1))
    batch_tensor = np.expand_dims(chw_tensor, axis=0)

    return batch_tensor, metadata
