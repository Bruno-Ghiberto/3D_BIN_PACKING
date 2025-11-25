"""Shelf-based packing algorithm.

This module implements shelf-based 2D packing extended to 3D,
where items are placed on horizontal "shelves" within each bin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from bin_packer_3d.algorithms.base import PackerBase
from bin_packer_3d.models.bin import Bin
from bin_packer_3d.models.placement import Placement, PlacementResult

if TYPE_CHECKING:
    from bin_packer_3d.models.box import Box
    from bin_packer_3d.config import PackerConfig


@dataclass
class FreeRectangle:
    """A 2D free rectangle in the shelf plane."""
    x: float
    y: float
    width: float
    depth: float


@dataclass
class Shelf:
    """A horizontal shelf layer within a bin."""
    z_offset: float
    bin_length: float
    bin_width: float
    bin_height: float  # Total bin height (for default max_height)
    current_height: float = 0.0
    max_height: float | None = None  # Ceiling height (set when next shelf created)
    is_ceiling_locked: bool = False  # Once True, max_height won't change
    free_rects: list[FreeRectangle] = field(default_factory=list)
    placements: list[Placement] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.free_rects:
            self.free_rects = [
                FreeRectangle(x=0, y=0, width=self.bin_length, depth=self.bin_width)
            ]
        # Default max_height is the remaining space to bin top
        if self.max_height is None:
            self.max_height = self.bin_height - self.z_offset

    @property
    def available_height(self) -> float:
        """Return maximum allowed height for boxes on this shelf."""
        return self.max_height if self.max_height is not None else (self.bin_height - self.z_offset)

    def lock_ceiling(self, new_shelf_z_offset: float) -> None:
        """Lock the ceiling height when a new shelf is created above this one."""
        if not self.is_ceiling_locked:
            self.max_height = new_shelf_z_offset - self.z_offset
            self.is_ceiling_locked = True


class ShelfPacker(PackerBase):
    """Shelf-based packing algorithm.
    
    Items are placed on horizontal shelves within bins. Each shelf
    has a height determined by the tallest item placed on it.
    
    This approach uses 2D rectangle packing (guillotine cuts) within
    each shelf, extended to 3D by stacking shelves vertically.
    
    Complexity:
        Time: O(n * s * r) where s = shelves, r = rectangles per shelf
        Space: O(n + s * r)
    """

    def __init__(self, config: "PackerConfig") -> None:
        super().__init__(config)
        self._bins: list[Bin] = []
        self._bin_shelves: list[list[Shelf]] = []

    @property
    def name(self) -> str:
        return "Shelf-Based Packer"

    def _pack_impl(self, boxes: list["Box"]) -> PlacementResult:
        """Implement shelf-based packing."""
        self._bins = []
        self._bin_shelves = []
        unpacked: list["Box"] = []
        
        for box in boxes:
            placed = self._place_box(box)
            if not placed:
                unpacked.append(box)
        
        # Collect all placements into bins
        for bin_idx, bin_obj in enumerate(self._bins):
            for shelf in self._bin_shelves[bin_idx]:
                for placement in shelf.placements:
                    bin_obj.add_placement(placement)
        
        return PlacementResult(
            bins=self._bins,
            unpacked_boxes=unpacked,
        )

    def _place_box(self, box: "Box") -> bool:
        """Try to place box in existing bins/shelves or create new."""
        # Try existing bins
        for bin_idx in range(len(self._bins)):
            if self._try_place_in_bin(box, bin_idx):
                return True
        
        # Create new bin
        return self._create_new_bin_and_place(box)

    def _try_place_in_bin(self, box: "Box", bin_idx: int) -> bool:
        """Try to place box in existing shelves or create new shelf."""
        bin_obj = self._bins[bin_idx]
        
        # Check weight
        if not bin_obj.can_fit_weight(box.weight):
            return False
        
        shelves = self._bin_shelves[bin_idx]
        
        # Try existing shelves
        for shelf in shelves:
            if self._try_place_on_shelf(box, bin_idx, shelf):
                return True
        
        # Try creating new shelf
        return self._try_create_shelf(box, bin_idx)

    def _try_place_on_shelf(
        self,
        box: "Box",
        bin_idx: int,
        shelf: Shelf,
    ) -> bool:
        """Try to place box on a specific shelf."""
        for rect_idx, rect in enumerate(shelf.free_rects):
            for w, h, depth in box.orientations():
                # Check if fits in rectangle and height fits within shelf's ceiling
                if (w <= rect.width and
                    depth <= rect.depth and
                    h <= shelf.available_height):
                    
                    # Place the box
                    placement = Placement(
                        box=box,
                        bin_id=self._bins[bin_idx].id,
                        shelf_id=self._bin_shelves[bin_idx].index(shelf) + 1,
                        x0=rect.x,
                        y0=rect.y,
                        z0=shelf.z_offset,
                        x1=rect.x + w,
                        y1=rect.y + depth,
                        z1=shelf.z_offset + h,
                    )
                    shelf.placements.append(placement)
                    
                    # Update shelf height
                    if h > shelf.current_height:
                        shelf.current_height = h
                    
                    # Split the rectangle
                    self._split_rectangle(shelf, rect_idx, w, depth)
                    return True
        
        return False

    def _try_create_shelf(self, box: "Box", bin_idx: int) -> bool:
        """Try to create a new shelf and place box on it."""
        shelves = self._bin_shelves[bin_idx]

        # Calculate current used height as the maximum z1 of all placements
        # This prevents overlap issues when boxes of varying heights are placed
        used_height = 0.0
        for shelf in shelves:
            for placement in shelf.placements:
                if placement.z1 > used_height:
                    used_height = placement.z1

        # Check if there's room for a new shelf
        for w, h, depth in box.orientations():
            if used_height + h <= self.config.bin_height:
                # Lock all existing shelves' ceilings at the new shelf's floor
                # This prevents future placements from overlapping with the new shelf
                for existing_shelf in shelves:
                    existing_shelf.lock_ceiling(used_height)

                # Create new shelf
                new_shelf = Shelf(
                    z_offset=used_height,
                    bin_length=self.config.bin_length,
                    bin_width=self.config.bin_width,
                    bin_height=self.config.bin_height,
                )
                shelves.append(new_shelf)

                # Try placing on new shelf
                if self._try_place_on_shelf(box, bin_idx, new_shelf):
                    return True
                else:
                    # Remove empty shelf (ceiling locks remain - they're permanent)
                    shelves.pop()

        return False

    def _create_new_bin_and_place(self, box: "Box") -> bool:
        """Create a new bin and place box in it."""
        new_bin = Bin(
            id=len(self._bins) + 1,
            length=self.config.bin_length,
            width=self.config.bin_width,
            height=self.config.bin_height,
            max_weight=self.config.max_weight,
        )
        self._bins.append(new_bin)
        self._bin_shelves.append([])
        
        bin_idx = len(self._bins) - 1
        if self._try_create_shelf(box, bin_idx):
            return True
        else:
            # Box doesn't fit in empty bin
            self._bins.pop()
            self._bin_shelves.pop()
            return False

    def _split_rectangle(
        self,
        shelf: Shelf,
        rect_idx: int,
        used_w: float,
        used_d: float,
    ) -> None:
        """Split a rectangle after placing a box (guillotine cut)."""
        rect = shelf.free_rects[rect_idx]
        del shelf.free_rects[rect_idx]
        
        # Right space
        remaining_w = rect.width - used_w
        if remaining_w > 0:
            shelf.free_rects.append(FreeRectangle(
                x=rect.x + used_w,
                y=rect.y,
                width=remaining_w,
                depth=rect.depth,
            ))
        
        # Top space (in 2D plane)
        remaining_d = rect.depth - used_d
        if remaining_d > 0:
            shelf.free_rects.append(FreeRectangle(
                x=rect.x,
                y=rect.y + used_d,
                width=used_w,
                depth=remaining_d,
            ))
