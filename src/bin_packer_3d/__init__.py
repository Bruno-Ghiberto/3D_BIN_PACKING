"""bin_packer_3d - A 3D Bin Packing solver using heuristic algorithms.

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

from bin_packer_3d.algorithms import ALGORITHMS, get_strategies, register
from bin_packer_3d.benchmark import BenchmarkResult
from bin_packer_3d.config import DataConfig, PackerConfig, Settings, VisualizationConfig
from bin_packer_3d.data.loaders import ColumnMapping
from bin_packer_3d.models.bin import Bin
from bin_packer_3d.models.box import Box
from bin_packer_3d.models.metadata import AlgorithmMetadata
from bin_packer_3d.models.placement import Placement, PlacementResult
from bin_packer_3d.models.result import LoadReport, PackingResult, RejectedRow
from bin_packer_3d.observability import StructuredAdapter, get_logger

__version__ = "0.2.0.dev0"
__author__ = "Bruno Ghiberto"

# Attach NullHandler to the package logger at import time so consumers
# who have not configured logging never see "No handlers could be found
# for logger bin_packer_3d" warnings. See Constitution §V and
# docs/adr/0008-observability.md.
get_logger("")

__all__ = [
    "ALGORITHMS",
    "AlgorithmMetadata",
    "BenchmarkResult",
    "Bin",
    "Box",
    "ColumnMapping",
    "DataConfig",
    "LoadReport",
    "PackerConfig",
    "PackingResult",
    "Placement",
    "PlacementResult",
    "RejectedRow",
    "Settings",
    "StructuredAdapter",
    "VisualizationConfig",
    "__version__",
    "get_logger",
    "get_strategies",
    "register",
]
