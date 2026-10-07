"""Interactive camera ground calibration wizard and homography solver."""

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from rich.console import Console

from spatialtrack.core.exceptions import CalibrationError
from spatialtrack.core.types import CalibrationData, PixelCoord, WorldCoord
from spatialtrack.io.video_reader import VideoReader
from spatialtrack.projection.calibration import save_calibration, validate_calibration_points
from spatialtrack.projection.homography import compute_homography_matrices

console = Console()


def parse_points_str(points_str: str) -> list[tuple[float, float]]:
    """Parse semi-colon separated coordinate pairs (e.g. 'x1,y1;x2,y2;x3,y3;x4,y4')."""
    pairs = [pair.strip() for pair in points_str.split(";") if pair.strip()]
    if len(pairs) != 4:
        raise CalibrationError(f"Expected 4 coordinate pairs, got {len(pairs)} from '{points_str}'")

    parsed: list[tuple[float, float]] = []
    for pair in pairs:
        tokens = [t.strip() for t in pair.split(",") if t.strip()]
        if len(tokens) != 2:
            raise CalibrationError(f"Invalid coordinate token pair '{pair}'")
        parsed.append((float(tokens[0]), float(tokens[1])))
    return parsed


def run_calibration_wizard(
    source: str,
    output_path: Path,
    src_points_str: str | None = None,
    dst_points_str: str | None = None,
    description: str = "Standard 4-point ground calibration",
) -> None:
    """Execute calibration procedure and save homography data to JSON file."""
    console.print("[bold cyan]SpatialTrack[/bold cyan] - Camera Calibration Tool")
    frame, width, height = _load_first_frame(source)

    if src_points_str and dst_points_str:
        src_raw = parse_points_str(src_points_str)
        dst_raw = parse_points_str(dst_points_str)
        src_pts = [PixelCoord(pt) for pt in src_raw]
        dst_pts = [WorldCoord(pt) for pt in dst_raw]
    else:
        src_pts, dst_pts = _interactive_point_picker(frame)

    validate_calibration_points(src_pts, dst_pts)
    forward_h, _ = compute_homography_matrices(src_pts, dst_pts)

    calib_data = CalibrationData(
        source_points_px=src_pts,
        target_points_m=dst_pts,
        frame_width=width,
        frame_height=height,
        description=description,
    )
    save_calibration(calib_data, output_path)

    console.print(f"[bold green]Successfully saved calibration to: {output_path}[/bold green]")
    console.print(f"Condition Number: {np.linalg.cond(forward_h):.2f}")


def _load_first_frame(source: str) -> tuple[NDArray[np.uint8], int, int]:
    """Retrieve first frame from video stream or static image file."""
    src_path = Path(source)
    if src_path.is_file() and src_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        img = cv2.imread(str(src_path))
        if img is None:
            raise CalibrationError(f"Failed to read image at: {source}")
        bgr_img: NDArray[np.uint8] = np.asarray(img, dtype=np.uint8)
        return bgr_img, int(img.shape[1]), int(img.shape[0])

    with VideoReader(source) as reader:
        for frame in reader:
            return frame, int(reader.width), int(reader.height)

    raise CalibrationError(f"No frames could be extracted from: {source}")


def _interactive_point_picker(
    frame: NDArray[np.uint8],
) -> tuple[list[PixelCoord], list[WorldCoord]]:
    """Capture 4 ground points via mouse clicks and terminal metric coordinates."""
    clicked_pixels: list[tuple[float, float]] = []

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked_pixels) < 4:
            clicked_pixels.append((float(x), float(y)))
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(
                frame,
                f"P{len(clicked_pixels)}",
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
            cv2.imshow("SpatialTrack - Calibration Wizard", frame)

    try:
        cv2.imshow("SpatialTrack - Calibration Wizard", frame)
        cv2.setMouseCallback("SpatialTrack - Calibration Wizard", on_mouse)
        console.print("[yellow]Click 4 points on the ground plane in the OpenCV window.[/yellow]")

        while len(clicked_pixels) < 4:
            key = cv2.waitKey(20) & 0xFF
            if key == 27 or key == ord("q"):
                raise CalibrationError("Calibration aborted by user.")

        cv2.destroyAllWindows()
    except cv2.error as err:
        raise CalibrationError(
            "GUI display not supported in current environment. "
            "Please use --src-points and --dst-points flags."
        ) from err

    dst_pts: list[WorldCoord] = []
    for i, pt in enumerate(clicked_pixels, 1):
        console.print(f"Point {i} clicked at pixel: ({pt[0]:.1f}, {pt[1]:.1f})")
        val_x = float(console.input(f"Enter real-world X in meters for Point {i}: "))
        val_y = float(console.input(f"Enter real-world Y in meters for Point {i}: "))
        dst_pts.append(WorldCoord((val_x, val_y)))

    return [PixelCoord(p) for p in clicked_pixels], dst_pts
