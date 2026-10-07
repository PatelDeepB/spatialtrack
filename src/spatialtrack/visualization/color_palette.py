"""Deterministic color generation for consistent visual tracking and labeling."""

# Distinct, high-contrast BGR colors for visualization
COLOR_PALETTE: tuple[tuple[int, int, int], ...] = (
    (255, 56, 56),  # Red
    (255, 157, 151),  # Coral
    (255, 112, 31),  # Orange
    (255, 178, 29),  # Amber
    (207, 210, 49),  # Lime
    (72, 249, 10),  # Bright Green
    (146, 204, 23),  # Olive Green
    (61, 219, 134),  # Seafoam
    (26, 147, 65),  # Forest Green
    (0, 212, 187),  # Teal
    (44, 153, 168),  # Cyan
    (0, 194, 255),  # Sky Blue
    (52, 69, 255),  # Electric Blue
    (100, 115, 255),  # Periwinkle
    (0, 24, 236),  # Deep Blue
    (132, 56, 255),  # Violet
    (82, 0, 133),  # Dark Purple
    (203, 56, 255),  # Magenta
    (255, 149, 200),  # Rose
    (255, 56, 132),  # Hot Pink
)


def get_color_for_id(identifier: int) -> tuple[int, int, int]:
    """Retrieve a deterministic BGR color based on integer ID hash.

    Args:
        identifier: Integer object ID or index.

    Returns:
        BGR color tuple (Blue, Green, Red).
    """
    index = abs(int(identifier)) % len(COLOR_PALETTE)
    return COLOR_PALETTE[index]


def get_color_for_label(label: str) -> tuple[int, int, int]:
    """Retrieve a deterministic BGR color based on text label string hash.

    Args:
        label: Class name string.

    Returns:
        BGR color tuple (Blue, Green, Red).
    """
    hash_value = sum(ord(char) for char in label)
    return get_color_for_id(hash_value)
