"""
bin_packer_3d - A 3D Bin Packing solver using heuristic algorithms.

This package provides efficient algorithms for solving the 3D Bin Packing Problem (3D-BPP),
a classic NP-hard optimization problem with applications in logistics, warehousing,
and container loading.

Features:
    - Multiple packing algorithms (FFD, Shelf-based, etc.)
    - Interactive 3D visualization with Plotly
    - Support for box rotations (6 orientations)
    - CSV/Excel data loading
    - Comprehensive metrics calculation

Example:
    >>> from bin_packer_3d import Box, Bin, PackerConfig
    >>> from bin_packer_3d.algorithms import FirstFitDecreasingPacker
    >>>
    >>> boxes = [Box(id="box1", width=100, height=50, length=80)]
    >>> config = PackerConfig(bin_length=860, bin_width=890, bin_height=1040)
    >>> packer = FirstFitDecreasingPacker(config)
    >>> result = packer.pack(boxes)
"""

from bin_packer_3d.models.box import Box
from bin_packer_3d.models.bin import Bin
from bin_packer_3d.models.placement import Placement, PlacementResult
from bin_packer_3d.config import PackerConfig, VisualizationConfig, DataConfig, Settings

__version__ = "0.2.0.dev0"
__author__ = "Bruno Ghiberto"

__all__ = [
    "__version__",
    "Box",
    "Bin",
    "Placement",
    "PlacementResult",
    "PackerConfig",
    "VisualizationConfig", 
    "DataConfig",
    "Settings",
]
