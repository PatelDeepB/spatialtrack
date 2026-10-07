"""Frame annotation utilities for drawing bounding boxes, labels, and tracking metadata."""

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from spatialtrack.core.config import VisualizationConfig
from spatialtrack.core.types import Detection, PixelCoord, Track
from spatialtrack.visualization.color_palette import get_color_for_id, get_color_for_label


class FrameAnnotator:
    """Renderer for bounding boxes, tracks, trails, and labels on camera image frames."""

    def __init__(self, config: VisualizationConfig) -> None:
        """Initialize renderer with visual formatting parameters."""
        self.config = config

    def draw_detections(
        self,
        frame: NDArray[np.uint8],
        detections: Sequence[Detection],
    ) -> NDArray[np.uint8]:
        """Draw bounding boxes and class confidence tags on a frame copy."""
        output_frame = frame.copy()
        for detection in detections:
            self._draw_single_detection(output_frame, detection)
        return output_frame

    def draw_tracks(
        self,
        frame: NDArray[np.uint8],
        tracks: Sequence[Track],
    ) -> NDArray[np.uint8]:
        """Draw tracked bounding boxes, persistent IDs, and trajectory trails on a frame copy.

        Args:
            frame: Raw BGR input frame.
            tracks: Sequence of confirmed Track objects.

        Returns:
            New annotated BGR image frame copy.
        """
        output_frame = frame.copy()
        for track in tracks:
            self._draw_single_track(output_frame, track)
        return output_frame

    def _draw_single_detection(
        self,
        canvas: NDArray[np.uint8],
        detection: Detection,
    ) -> None:
        """Draw an individual detection box with colored badge label."""
        bbox = detection.bbox
        x1, y1 = int(round(bbox.x1)), int(round(bbox.y1))
        x2, y2 = int(round(bbox.x2)), int(round(bbox.y2))

        label_text = f"{detection.object_class.name.lower()} {detection.confidence:.2f}"
        color = get_color_for_label(detection.object_class.name)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, self.config.line_thickness)
        self._draw_label_badge(canvas, label_text, (x1, y1), color)

    def _draw_single_track(
        self,
        canvas: NDArray[np.uint8],
        track: Track,
    ) -> None:
        """Draw a tracked object with bounding box, ID label, and trajectory trail."""
        color = get_color_for_id(int(track.track_id))
        bbox = track.bbox
        x1, y1 = int(round(bbox.x1)), int(round(bbox.y1))
        x2, y2 = int(round(bbox.x2)), int(round(bbox.y2))

        # 1. Draw trajectory trail
        if self.config.show_trails and len(track.pixel_trail) > 1:
            self._draw_trajectory_trail(canvas, track.pixel_trail, color)

        # 2. Draw ground contact dot
        cx, cy = int(round(bbox.bottom_center[0])), int(round(bbox.bottom_center[1]))
        cv2.circle(canvas, (cx, cy), 4, color, -1)

        # 3. Draw bounding rectangle
        if self.config.show_bboxes:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, self.config.line_thickness)

        # 4. Draw persistent identity badge
        label_text = f"#{track.track_id} {track.object_class.name.lower()}"
        self._draw_label_badge(canvas, label_text, (x1, y1), color)

    def _draw_trajectory_trail(
        self,
        canvas: NDArray[np.uint8],
        trail: Sequence[PixelCoord],
        color: tuple[int, int, int],
    ) -> None:
        """Draw a progressive fading trajectory trail connecting historical positions."""
        num_points = len(trail)
        for i in range(1, num_points):
            pt1 = (int(round(trail[i - 1][0])), int(round(trail[i - 1][1])))
            pt2 = (int(round(trail[i][0])), int(round(trail[i][1])))

            if self.config.trail_fade:
                # Thickness increases toward the recent head
                thickness = max(1, int(round((i / num_points) * self.config.line_thickness * 1.5)))
            else:
                thickness = self.config.line_thickness

            cv2.line(canvas, pt1, pt2, color, thickness, cv2.LINE_AA)

    def _draw_label_badge(
        self,
        canvas: NDArray[np.uint8],
        text: str,
        origin: tuple[int, int],
        badge_color: tuple[int, int, int],
    ) -> None:
        """Draw filled label background badge with contrasting white text."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self.config.font_scale
        thickness = 1

        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x, y = origin

        badge_y1 = max(0, y - text_h - baseline - 4)
        badge_y2 = y
        badge_x2 = min(canvas.shape[1], x + text_w + 6)

        cv2.rectangle(canvas, (x, badge_y1), (badge_x2, badge_y2), badge_color, -1)
        text_origin = (x + 3, max(text_h + 2, y - baseline - 2))
        cv2.putText(
            canvas,
            text,
            text_origin,
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
