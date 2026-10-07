"""Download official YOLOv10n ONNX model and prepare optimized model weights."""

import sys
import urllib.request
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TimeRemainingColumn

console = Console()

MODEL_URL = "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.onnx"
DEFAULT_MODELS_DIR = Path("models")
FP32_MODEL_PATH = DEFAULT_MODELS_DIR / "yolov10n.onnx"
INT8_MODEL_PATH = DEFAULT_MODELS_DIR / "yolov10n_int8.onnx"


def download_file(url: str, destination: Path) -> None:
    """Download file from URL with a visual progress bar.

    Args:
        url: Direct download link.
        destination: Destination file path on local filesystem.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        console.print(f"[green]Model already present at:[/green] {destination}")
        return

    console.print(f"[cyan]Downloading model from:[/cyan] {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 SpatialTrack/0.1"})

    with urllib.request.urlopen(request) as response:
        total_size = int(response.headers.get("content-length", 0))

        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TimeRemainingColumn(),
        )

        with progress:
            task_id = progress.add_task("Downloading", total=total_size)
            with open(destination, "wb") as output_file:
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    output_file.write(chunk)
                    progress.update(task_id, advance=len(chunk))

    console.print(f"[bold green]Saved to:[/bold green] {destination}")


def quantize_model(input_path: Path, output_path: Path) -> None:
    """Apply dynamic INT8 quantization to an ONNX model for CPU speedup.

    Args:
        input_path: Path to FP32 ONNX model.
        output_path: Path to target quantized INT8 ONNX model.
    """
    if output_path.is_file() and output_path.stat().st_size > 0:
        console.print(f"[green]Quantized model already exists at:[/green] {output_path}")
        return

    console.print(f"[cyan]Quantizing model to INT8:[/cyan] {output_path}")
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QUInt8,
    )
    console.print(f"[bold green]Successfully generated INT8 model:[/bold green] {output_path}")


def main() -> None:
    """Download model and generate INT8 quantized weights."""
    try:
        download_file(MODEL_URL, FP32_MODEL_PATH)
        quantize_model(FP32_MODEL_PATH, INT8_MODEL_PATH)
    except Exception as err:
        console.print(f"[bold red]Failed to download or quantize model:[/bold red] {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
