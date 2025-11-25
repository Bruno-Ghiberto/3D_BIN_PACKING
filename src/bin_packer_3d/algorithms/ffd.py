"""First-Fit Decreasing (FFD) packing algorithm.

This module implements the classic FFD heuristic where boxes are
sorted by volume and placed in the first bin that can accommodate them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bin_packer_3d.algorithms.base import PackerBase
from bin_packer_3d.models.bin import Bin
from bin_packer_3d.models.placement import Placement, PlacementResult

if TYPE_CHECKING:
    from bin_packer_3d.models.box import Box
    from bin_packer_3d.config import PackerConfig


class FirstFitDecreasingPacker(PackerBase):
    """First-Fit Decreasing packing algorithm.
    
    Boxes are sorted by volume (largest first) and each box is placed
    in the first bin that has space for it. New bins are created as needed.
    
    This is a simple but effective baseline algorithm for 3D-BPP.
    
    Complexity:
        Time: O(n log n) for sorting + O(n * m) for placement
        Space: O(n + m) where n = boxes, m = bins
    
    Example:
        >>> config = PackerConfig(bin_length=100, bin_width=100, bin_height=100)
        >>> packer = FirstFitDecreasingPacker(config)
        >>> boxes = [Box(id="1", width=50, height=50, length=50)]
        >>> result = packer.pack(boxes)
        >>> print(result.utilization_percent)
    """

    def __init__(self, config: "PackerConfig") -> None:
        super().__init__(config)
        self._bins: list[Bin] = []
        self._bin_spaces: list[list[tuple[float, float, float, float, float, float]]] = []

    @property
    def name(self) -> str:
        return "First-Fit Decreasing (FFD)"

    def _pack_impl(self, boxes: list["Box"]) -> PlacementResult:
        """Implement FFD packing.
        
        For each box (sorted by volume desc):
        1. Try to place in existing bins
        2. If no fit, create new bin
        3. Track unpacked boxes that don't fit anywhere
        """
        self._bins = []
        self._bin_spaces = []
        unpacked: list["Box"] = []
        
        for box in boxes:
            placed = False
            
            # Try existing bins first
            for bin_idx, bin_obj in enumerate(self._bins):
                placement = self._try_place_in_bin(box, bin_idx, bin_obj)
                if placement is not None:
                    bin_obj.add_placement(placement)
                    placed = True
                    break
            
            # Create new bin if needed
            if not placed:
                new_bin = Bin(
                    id=len(self._bins) + 1,
                    length=self.config.bin_length,
                    width=self.config.bin_width,
                    height=self.config.bin_height,
                    max_weight=self.config.max_weight,
                )
                
                # Initialize space tracking for new bin
                self._bins.append(new_bin)
                self._bin_spaces.append([
                    (0, 0, 0, 
                     self.config.bin_length, 
                     self.config.bin_width, 
                     self.config.bin_height)
                ])
                
                placement = self._try_place_in_bin(box, len(self._bins) - 1, new_bin)
                if placement is not None:
                    new_bin.add_placement(placement)
                else:
                    # Box doesn't fit in empty bin (too large)
                    self._bins.pop()
                    self._bin_spaces.pop()
                    unpacked.append(box)
        
        return PlacementResult(
            bins=self._bins,
            unpacked_boxes=unpacked,
        )

    def _try_place_in_bin(
        self,
        box: "Box",
        bin_idx: int,
        bin_obj: Bin,
    ) -> Placement | None:
        """Try to place box in a specific bin.
        
        Uses simple space tracking with guillotine-style splitting.
        """
        # Check weight constraint
        if not bin_obj.can_fit_weight(box.weight):
            return None
        
        spaces = self._bin_spaces[bin_idx]
        
        for space_idx, (sx0, sy0, sz0, sx1, sy1, sz1) in enumerate(spaces):
            space_w = sx1 - sx0
            space_h = sz1 - sz0  # Height is Z
            space_l = sy1 - sy0  # Length/depth is Y
            
            fits, orientation = self._can_fit_box(box, space_w, space_h, space_l)
            
            if fits and orientation is not None:
                w, h, l = orientation
                
                # Create placement
                placement = Placement(
                    box=box,
                    bin_id=bin_obj.id,
                    x0=sx0,
                    y0=sy0,
                    z0=sz0,
                    x1=sx0 + w,
                    y1=sy0 + l,
                    z1=sz0 + h,
                )
                
                # Update spaces (guillotine split)
                self._split_space(bin_idx, space_idx, w, l, h)
                
                return placement
        
        return None

    def _split_space(
        self,
        bin_idx: int,
        space_idx: int,
        used_w: float,
        used_l: float,
        used_h: float,
    ) -> None:
        """Split space after placing a box using guillotine cuts.
        
        Creates up to 3 new spaces:
        - Right of placed box
        - Behind placed box  
        - Above placed box
        """
        spaces = self._bin_spaces[bin_idx]
        sx0, sy0, sz0, sx1, sy1, sz1 = spaces[space_idx]
        
        # Remove used space
        del spaces[space_idx]
        
        # Right space (along X)
        remaining_w = sx1 - sx0 - used_w
        if remaining_w > 0:
            spaces.append((
                sx0 + used_w, sy0, sz0,
                sx1, sy1, sz1
            ))
        
        # Back space (along Y) 
        remaining_l = sy1 - sy0 - used_l
        if remaining_l > 0:
            spaces.append((
                sx0, sy0 + used_l, sz0,
                sx0 + used_w, sy1, sz1
            ))
        
        # Top space (along Z)
        remaining_h = sz1 - sz0 - used_h
        if remaining_h > 0:
            spaces.append((
                sx0, sy0, sz0 + used_h,
                sx0 + used_w, sy0 + used_l, sz1
            ))
