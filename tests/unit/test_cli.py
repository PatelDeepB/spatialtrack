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


def test_cli_telemetry_missing_source_fails() -> None:
    """Verify that omitting required --source argument for telemetry exits with error."""
    # Arrange & Act
    result = runner.invoke(app, ["telemetry"])

    # Assert
    assert result.exit_code == 2
