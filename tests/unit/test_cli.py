"""Unit tests for the Typer CLI application."""

from typer.testing import CliRunner

from spatialtrack import __version__
from spatialtrack.cli.main import app

runner = CliRunner()


def test_cli_version_command() -> None:
    """Verify that the version command returns the correct package version string."""
    # Arrange & Act
    result = runner.invoke(app, ["version"])

    # Assert
    assert result.exit_code == 0
    assert "SpatialTrack version:" in result.stdout
    assert __version__ in result.stdout


def test_cli_detect_missing_source_fails() -> None:
    """Verify that omitting required --source argument causes CLI to exit with an error code."""
    # Arrange & Act
    result = runner.invoke(app, ["detect"])

    # Assert
    assert result.exit_code == 2


def test_cli_track_missing_source_fails() -> None:
    """Verify that omitting required --source argument for track exits with error."""
    # Arrange & Act
    result = runner.invoke(app, ["track"])

    # Assert
    assert result.exit_code == 2


def test_cli_telemetry_missing_source_fails() -> None:
    """Verify that omitting required --source argument for telemetry exits with error."""
    # Arrange & Act
    result = runner.invoke(app, ["telemetry"])

    # Assert
    assert result.exit_code == 2


def test_cli_run_missing_source_fails() -> None:
    """Verify that omitting required --source argument for run exits with error."""
    # Arrange & Act
    result = runner.invoke(app, ["run"])

    # Assert
    assert result.exit_code == 2


def test_cli_calibrate_missing_source_fails() -> None:
    """Verify that omitting required --source argument for calibrate exits with error."""
    # Arrange & Act
    result = runner.invoke(app, ["calibrate"])

    # Assert
    assert result.exit_code == 2


def test_cli_benchmark_missing_source_fails() -> None:
    """Verify that omitting required --source argument for benchmark exits with error."""
    # Arrange & Act
    result = runner.invoke(app, ["benchmark"])

    # Assert
    assert result.exit_code == 2


def test_cli_run_executes_short_clip() -> None:
    """Verify that running CLI run command processes frames and exits 0."""
    # Arrange & Act
    result = runner.invoke(
        app,
        [
            "run",
            "--source",
            "app/assets/sample_traffic.mp4",
            "--calibration",
            "app/assets/sample_calibration.json",
            "--max-frames",
            "3",
        ],
    )

    # Assert
    assert result.exit_code == 0
    assert "Finished 3 frames" in result.stdout
