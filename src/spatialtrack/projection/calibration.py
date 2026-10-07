"""Perspective calibration storage, validation, and JSON serialization."""

import json
from pathlib import Path

from spatialtrack.core.exceptions import CalibrationError
from spatialtrack.core.types import CalibrationData, PixelCoord, WorldCoord


def validate_calibration_points(
    source_points: list[PixelCoord],
    target_points: list[WorldCoord],
) -> None:
    """Validate that source and target calibration point sets satisfy projective requirements.

    Args:
        source_points: Exactly 4 non-collinear pixel coordinates.
        target_points: Exactly 4 corresponding metric world coordinates.

    Raises:
        CalibrationError: If point counts differ from 4 or points are degenerate.
    """
    if len(source_points) != 4 or len(target_points) != 4:
        raise CalibrationError(
            f"Calibration requires exactly 4 point correspondences. "
            f"Got {len(source_points)} source and {len(target_points)} target points."
        )

    # Check for duplicate source points
    unique_sources = set(source_points)
    if len(unique_sources) < 4:
        raise CalibrationError("Calibration source points must all be distinct.")

    # Check for duplicate target points
    unique_targets = set(target_points)
    if len(unique_targets) < 4:
        raise CalibrationError("Calibration target points must all be distinct.")


def save_calibration(calibration: CalibrationData, file_path: str | Path) -> None:
    """Serialize calibration data to JSON format.

    Args:
        calibration: CalibrationData instance to save.
        file_path: Destination file path.
    """
    validate_calibration_points(calibration.source_points_px, calibration.target_points_m)
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "source_points_px": [list(pt) for pt in calibration.source_points_px],
        "target_points_m": [list(pt) for pt in calibration.target_points_m],
        "frame_width": calibration.frame_width,
        "frame_height": calibration.frame_height,
        "description": calibration.description,
    }

    with open(path, "w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)


def load_calibration(file_path: str | Path) -> CalibrationData:
    """Load and validate calibration data from a JSON file.

    Args:
        file_path: Path to target calibration JSON file.

    Returns:
        Validated CalibrationData instance.

    Raises:
        CalibrationError: If file is missing or contains invalid calibration data.
    """
    path = Path(file_path)
    if not path.is_file():
        raise CalibrationError(f"Calibration file does not exist at: {path}")

    try:
        with open(path, encoding="utf-8") as stream:
            data = json.load(stream)

        src_raw = data["source_points_px"]
        dst_raw = data["target_points_m"]

        sources = [PixelCoord((float(p[0]), float(p[1]))) for p in src_raw]
        targets = [WorldCoord((float(p[0]), float(p[1]))) for p in dst_raw]

        validate_calibration_points(sources, targets)

        return CalibrationData(
            source_points_px=sources,
            target_points_m=targets,
            frame_width=int(data["frame_width"]),
            frame_height=int(data["frame_height"]),
            description=str(data.get("description", "")),
        )
    except Exception as err:
        if isinstance(err, CalibrationError):
            raise
        raise CalibrationError(f"Failed to parse calibration file at {path}: {err}") from err
