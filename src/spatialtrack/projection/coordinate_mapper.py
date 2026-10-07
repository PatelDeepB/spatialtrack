"""Coordinate mapper providing bidirectional conversions between pixel and metric space."""

from collections.abc import Sequence

import numpy as np

from spatialtrack.core.types import CalibrationData, PixelCoord, Track, WorldCoord
from spatialtrack.projection.homography import compute_homography_matrices, project_points


class CoordinateMapper:
    """Manages calibrated coordinate transformations between camera pixels and world meters."""

    def __init__(self, calibration: CalibrationData) -> None:
        """Initialize mapper by computing forward and inverse homography matrices.

        Args:
            calibration: Validated 4-point CalibrationData instance.
        """
        self.calibration = calibration
        self.forward_matrix, self.inverse_matrix = compute_homography_matrices(
            calibration.source_points_px,
            calibration.target_points_m,
        )

        target_arr = np.array(calibration.target_points_m, dtype=np.float32)
        self.min_world_x = float(np.min(target_arr[:, 0]))
        self.max_world_x = float(np.max(target_arr[:, 0]))
        self.min_world_y = float(np.min(target_arr[:, 1]))
        self.max_world_y = float(np.max(target_arr[:, 1]))

    @property
    def world_width_m(self) -> float:
        """Get span of calibrated world area along X axis in meters."""
        return max(1.0, self.max_world_x - self.min_world_x)

    @property
    def world_height_m(self) -> float:
        """Get span of calibrated world area along Y axis in meters."""
        return max(1.0, self.max_world_y - self.min_world_y)

    def pixel_to_world(self, pixel_coord: PixelCoord) -> WorldCoord:
        """Project pixel coordinate (u, v) into metric world coordinates (X, Y)."""
        input_arr = np.array([[pixel_coord[0], pixel_coord[1]]], dtype=np.float32)
        world_pts = project_points(input_arr, self.forward_matrix)
        return WorldCoord((float(world_pts[0, 0]), float(world_pts[0, 1])))

    def world_to_pixel(self, world_coord: WorldCoord) -> PixelCoord:
        """Project a metric world coordinate (X, Y) back into camera pixel coordinates (u, v)."""
        input_arr = np.array([[world_coord[0], world_coord[1]]], dtype=np.float32)
        pixel_pts = project_points(input_arr, self.inverse_matrix)
        return PixelCoord((float(pixel_pts[0, 0]), float(pixel_pts[0, 1])))

    def project_trail(self, pixel_trail: Sequence[PixelCoord]) -> list[WorldCoord]:
        """Transform a sequence of pixel trail coordinates into metric world coordinates."""
        if not pixel_trail:
            return []
        input_arr = np.array(pixel_trail, dtype=np.float32)
        world_pts = project_points(input_arr, self.forward_matrix)
        return [WorldCoord((float(pt[0]), float(pt[1]))) for pt in world_pts]

    def project_track(self, track: Track) -> tuple[WorldCoord, list[WorldCoord]]:
        """Project track current contact position and historical trail to world coordinates."""
        current_world_pos = self.pixel_to_world(track.bbox.bottom_center)
        world_trail = self.project_trail(track.pixel_trail)
        return current_world_pos, world_trail
