"""Script for static and dynamic INT8 quantization of ONNX models for CPU execution."""

from pathlib import Path

import typer
from rich.console import Console

console = Console()
app = typer.Typer(help="Quantize ONNX models to INT8 for CPU deployment.")


@app.command()
def quantize(
    input_model: Path = typer.Option(
        Path("models/yolov10n.onnx"),
        "--input",
        "-i",
        help="Path to FP32 ONNX model.",
    ),
    output_model: Path = typer.Option(
        Path("models/yolov10n_int8.onnx"),
        "--output",
        "-o",
        help="Target INT8 model path.",
    ),
) -> None:
    """Quantize an ONNX model to INT8 representation for accelerated CPU throughput."""
    if not input_model.is_file():
        console.print(f"[bold red]Input model not found:[/bold red] {input_model}")
        raise typer.Exit(code=1)

    console.print("[cyan]Applying dynamic INT8 quantization...[/cyan]")
    from onnxruntime.quantization import QuantType, quantize_dynamic

    output_model.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(input_model),
        model_output=str(output_model),
        weight_type=QuantType.QUInt8,
    )

    original_size_mb = input_model.stat().st_size / (1024 * 1024)
    quantized_size_mb = output_model.stat().st_size / (1024 * 1024)

    console.print(f"[green]Original model size:[/green] {original_size_mb:.2f} MB")
    console.print(f"[green]Quantized model size:[/green] {quantized_size_mb:.2f} MB")
    console.print(f"[bold green]Saved to:[/bold green] {output_model}")


if __name__ == "__main__":
    app()
