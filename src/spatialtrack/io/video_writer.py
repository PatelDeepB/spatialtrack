"""Robust context-managed video writer wrapping OpenCV VideoWriter."""

from pathlib import Path
from types import TracebackType
from typing import Self

import cv2
import numpy as np
from numpy.typing import NDArray

from spatialtrack.core.exceptions import ExportError


class VideoWriter:
    """Context-managed video writer producing compressed MP4/AVI recordings."""

    def __init__(
        self,
        output_path: str | Path,
        fps: float,
        frame_size: tuple[int, int],
        codec: str = "mp4v",
    ) -> None:
        """Initialize video writer settings.

        Args:
            output_path: Target video file path.
            fps: Frame rate for the output video container.
            frame_size: Video frame dimensions as (width, height).
            codec: FourCC codec string (default: 'mp4v').
        """
        self.output_path = Path(output_path)
        self.fps = max(1.0, float(fps))
        self.width, self.height = frame_size
        self.codec = codec

        if self.width <= 0 or self.height <= 0:
            raise ExportError(f"Invalid video frame size: {frame_size}")

        self._writer: cv2.VideoWriter | None = None
        self._frame_count = 0

    @property
    def is_opened(self) -> bool:
        """Check whether underlying OpenCV video writer stream is active."""
        return self._writer is not None and bool(self._writer.isOpened())

    @property
    def frame_count(self) -> int:
        """Total number of frames successfully written to container."""
        return self._frame_count

    def open(self) -> Self:
        """Initialize the video file on disk and open the OpenCV writer stream."""
        if self.is_opened:
            return self

        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter.fourcc(*self.codec)
            self._writer = cv2.VideoWriter(
                str(self.output_path),
                fourcc,
                self.fps,
                (self.width, self.height),
            )
            if not self._writer.isOpened():
                raise ExportError(f"OpenCV failed to open writer for: {self.output_path}")
            return self
        except Exception as err:
            raise ExportError(f"Failed to open video writer at {self.output_path}: {err}") from err

    def write(self, frame: NDArray[np.uint8]) -> None:
        """Write a single BGR image frame to the video container.

        Args:
            frame: Numpy BGR image array with shape (height, width, 3).
        """
        if not self.is_opened or self._writer is None:
            self.open()

        h, w = frame.shape[:2]
        if (w, h) != (self.width, self.height):
            resized = cv2.resize(frame, (self.width, self.height))
            assert self._writer is not None
            self._writer.write(resized)
        else:
            assert self._writer is not None
            self._writer.write(frame)

        self._frame_count += 1

    def close(self) -> None:
        """Flush and release the underlying video writer handle."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> Self:
        """Context manager entry point."""
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Context manager exit point releasing writer resources."""
        self.close()
