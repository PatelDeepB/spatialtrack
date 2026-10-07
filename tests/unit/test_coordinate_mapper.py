"""Unit tests for the CoordinateMapper and calibration storage."""

from pathlib import Path

from spatialtrack.core.types import (
    BoundingBox,
    CalibrationData,
    Confidence,
    ObjectClass,
    PixelCoord,
    Track,
    TrackId,
    WorldCoord,
)
from spatialtrack.projection.calibration import load_calibration, save_calibration
from spatialtrack.projection.coordinate_mapper import CoordinateMapper


def test_coordinate_mapper_conversions(tmp_path: Path) -> None:
    """Verify coordinate mapper maps coordinates bidirectionally and serializes accurately."""
    # Arrange
    calib = CalibrationData(
        source_points_px=[
            PixelCoord((0.0, 0.0)),
            PixelCoord((200.0, 0.0)),
            PixelCoord((200.0, 200.0)),
            PixelCoord((0.0, 200.0)),
        ],
        target_points_m=[
            WorldCoord((0.0, 0.0)),
            WorldCoord((20.0, 0.0)),
            WorldCoord((20.0, 20.0)),
            WorldCoord((0.0, 20.0)),
        ],
        frame_width=640,
        frame_height=480,
        description="Test calibration",
    )

    # Test serialization
    calib_file = tmp_path / "test_calib.json"
    save_calibration(calib, calib_file)
    loaded_calib = load_calibration(calib_file)
    assert loaded_calib.frame_width == 640

    # Act
    mapper = CoordinateMapper(loaded_calib)
    world_pos = mapper.pixel_to_world(PixelCoord((100.0, 100.0)))
    pixel_pos = mapper.world_to_pixel(world_pos)

    # Assert
    assert abs(world_pos[0] - 10.0) < 1e-2
    assert abs(world_pos[1] - 10.0) < 1e-2
    assert abs(pixel_pos[0] - 100.0) < 1e-2
    assert abs(pixel_pos[1] - 100.0) < 1e-2


def test_coordinate_mapper_project_track() -> None:
    """Verify track projection transforms current bottom-center point and trajectory trail."""
    # Arrange
    calib = CalibrationData(
        source_points_px=[
            PixelCoord((0.0, 0.0)),
            PixelCoord((100.0, 0.0)),
            PixelCoord((100.0, 100.0)),
            PixelCoord((0.0, 100.0)),
        ],
        target_points_m=[
            WorldCoord((0.0, 0.0)),
            WorldCoord((10.0, 0.0)),
            WorldCoord((10.0, 10.0)),
            WorldCoord((0.0, 10.0)),
        ],
        frame_width=100,
        frame_height=100,
    )
    mapper = CoordinateMapper(calib)
    track = Track(
        track_id=TrackId(1),
        bbox=BoundingBox(x1=20.0, y1=20.0, x2=40.0, y2=80.0),
        object_class=ObjectClass.CAR,
        confidence=Confidence(0.9),
        age=3,
        time_since_update=0,
        pixel_trail=[PixelCoord((30.0, 60.0)), PixelCoord((30.0, 80.0))],
    )

    # Act
    world_pos, world_trail = mapper.project_track(track)

    # Assert: bottom_center is (30.0, 80.0) -> world (3.0, 8.0)
    assert abs(world_pos[0] - 3.0) < 1e-2
    assert abs(world_pos[1] - 8.0) < 1e-2
    assert len(world_trail) == 2
