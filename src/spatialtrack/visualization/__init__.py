"""Visualization package for overlay rendering, BEV maps, and dashboards."""

from spatialtrack.visualization.annotator import FrameAnnotator
from spatialtrack.visualization.bev_renderer import BevRenderer
from spatialtrack.visualization.color_palette import (
    COLOR_PALETTE,
    get_color_for_id,
    get_color_for_label,
)
from spatialtrack.visualization.dashboard import Dashboard

__all__ = [
    "COLOR_PALETTE",
    "BevRenderer",
    "Dashboard",
    "FrameAnnotator",
    "get_color_for_id",
    "get_color_for_label",
]
