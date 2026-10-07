"""Unit tests for the color palette generator."""

from spatialtrack.visualization.color_palette import (
    COLOR_PALETTE,
    get_color_for_id,
    get_color_for_label,
)


def test_get_color_for_id_determinism() -> None:
    """Verify that identical IDs always yield identical BGR colors."""
    # Arrange
    track_id = 42

    # Act
    color_first = get_color_for_id(track_id)
    color_second = get_color_for_id(track_id)

    # Assert
    assert color_first == color_second
    assert len(color_first) == 3
    assert color_first in COLOR_PALETTE


def test_get_color_for_label_determinism() -> None:
    """Verify that identical label strings always yield identical colors."""
    # Arrange
    label = "car"

    # Act
    color_first = get_color_for_label(label)
    color_second = get_color_for_label(label)

    # Assert
    assert color_first == color_second
    assert len(color_first) == 3
    assert color_first in COLOR_PALETTE
