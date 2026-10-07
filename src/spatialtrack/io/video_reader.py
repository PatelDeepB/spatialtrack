"""Threaded video reader decoupling frame decoding from pipeline inference."""

import threading
from collections.abc import Iterator
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Self

import cv2
import numpy as np
from numpy.typing import NDArray

from spatialtrack.core.exceptions import VideoSourceError


class VideoReader:
    """Threaded video frame reader for local files, RTSP streams, and USB webcams."""

    def __init__(
        self,
        source: str | Path | int,
        queue_size: int = 64,
        timeout_seconds: float = 2.0,
    ) -> None:
        """Initialize video capture and worker thread structures.

        Args:
            source: Path to video file, RTSP stream URL, or integer webcam index.
            queue_size: Maximum buffer capacity for decoded frames.
            timeout_seconds: Frame wait timeout before raising or stopping.

        Raises:
            VideoSourceError: If the underlying video stream cannot be opened.
        """
        self._source_identifier = str(source)
        self._queue: Queue[NDArray[np.uint8] | None] = Queue(maxsize=queue_size)
        self._timeout_seconds = timeout_seconds
        self._is_running = False
        self._worker_thread: threading.Thread | None = None

        self._capture = self._initialize_capture(source)
        self._width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 30.0)
        self._total_frames = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))

    @staticmethod
    def _initialize_capture(source: str | Path | int) -> cv2.VideoCapture:
        """Open cv2.VideoCapture with validation."""
        capture_source = (
            int(source)
            if isinstance(source, int) or (isinstance(source, str) and source.isdigit())
            else str(source)
        )
        cap = cv2.VideoCapture(capture_source)
        if not cap.isOpened():
            raise VideoSourceError(f"Could not open video stream from source: {source}")
        return cap

    @property
    def width(self) -> int:
        """Get native video horizontal resolution."""
        return self._width

    @property
    def height(self) -> int:
        """Get native video vertical resolution."""
        return self._height

    @property
    def fps(self) -> float:
        """Get native video frames per second."""
        return self._fps

    @property
    def total_frames(self) -> int:
        """Get total number of frames in video file, or -1 if live stream."""
        return self._total_frames

    def start(self) -> Self:
        """Start background frame decoding thread."""
        if self._is_running:
            return self

        self._is_running = True
        self._worker_thread = threading.Thread(target=self._decode_worker, daemon=True)
        self._worker_thread.start()
        return self

    def _decode_worker(self) -> None:
        """Background thread worker continuously reading frames into queue."""
        while self._is_running:
            success, frame = self._capture.read()
            if not success or frame is None:
                # Video ended or disconnected
                self._enqueue_with_retry(None)
                break

            frame_uint8: NDArray[np.uint8] = np.asarray(frame, dtype=np.uint8)
            if not self._enqueue_with_retry(frame_uint8):
                break

    def _enqueue_with_retry(self, item: NDArray[np.uint8] | None) -> bool:
        """Attempt to push item into queue with retry loop."""
        while self._is_running:
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except Full:
                continue
        return False

    def read_frame(self) -> NDArray[np.uint8] | None:
        """Retrieve next decoded frame from buffer, returning None at stream termination."""
        if not self._is_running:
            self.start()

        try:
            frame = self._queue.get(timeout=self._timeout_seconds)
            return frame
        except Empty:
            return None

    def __iter__(self) -> Iterator[NDArray[np.uint8]]:
        """Yield frames sequentially until source is exhausted."""
        self.start()
        while True:
            frame = self.read_frame()
            if frame is None:
                break
            yield frame

    def close(self) -> None:
        """Stop background worker and release video capture resources."""
        self._is_running = False
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        if self._capture.isOpened():
            self._capture.release()

    def __enter__(self) -> Self:
        """Context manager entry point."""
        return self.start()

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Context manager exit point releasing resources."""
        self.close()
