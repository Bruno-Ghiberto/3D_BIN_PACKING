"""Packing algorithms for 3D bin packing.

This module provides various algorithms for solving the 3D-BPP:
- First-Fit Decreasing (FFD)
- Best-Fit Decreasing (BFD) 
- Shelf-based packing
- Extreme Points (planned)
"""

from bin_packer_3d.algorithms.base import PackerBase
from bin_packer_3d.algorithms.ffd import FirstFitDecreasingPacker
from bin_packer_3d.algorithms.shelf import ShelfPacker

__all__ = [
    "PackerBase",
    "FirstFitDecreasingPacker",
    "ShelfPacker",
]
