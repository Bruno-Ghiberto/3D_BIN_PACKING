"""Bin/Container model for packing.

This module defines the Bin class representing containers
that hold packed boxes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bin_packer_3d.models.placement import Placement


@dataclass
class Bin:
    """A container/bin for packing boxes.
    
    Attributes:
        id: Unique identifier for the bin.
        length: Length dimension in mm (X-axis).
        width: Width dimension in mm (Y-axis).
        height: Height dimension in mm (Z-axis).
        max_weight: Maximum weight capacity in kg.
        placements: List of box placements in this bin.
    
    Example:
        >>> bin = Bin(id=1, length=860, width=890, height=1040)
        >>> bin.volume
        795544000.0
        >>> bin.utilization_percent  # Initially 0%
        0.0
    """

    id: int
    length: float
    width: float
    height: float
    max_weight: float | None = None
    placements: list["Placement"] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate bin dimensions."""
        if self.length <= 0 or self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"All dimensions must be positive. Got: "
                f"l={self.length}, w={self.width}, h={self.height}"
            )

    @property
    def volume(self) -> float:
        """Calculate bin volume in cubic mm."""
        return self.length * self.width * self.height

    @property
    def volume_m3(self) -> float:
        """Calculate bin volume in cubic meters."""
        return self.volume / 1_000_000_000

    @property
    def used_volume(self) -> float:
        """Calculate total volume of placed boxes in cubic mm."""
        return sum(p.volume for p in self.placements)

    @property
    def remaining_volume(self) -> float:
        """Calculate remaining available volume in cubic mm."""
        return self.volume - self.used_volume

    @property
    def utilization_percent(self) -> float:
        """Calculate space utilization as percentage (0-100)."""
        if self.volume == 0:
            return 0.0
        return (self.used_volume / self.volume) * 100

    @property
    def total_weight(self) -> float:
        """Calculate total weight of placed boxes in kg."""
        return sum(p.box.weight for p in self.placements)

    @property
    def remaining_weight_capacity(self) -> float | None:
        """Calculate remaining weight capacity in kg."""
        if self.max_weight is None:
            return None
        return self.max_weight - self.total_weight

    @property
    def box_count(self) -> int:
        """Return number of boxes placed in this bin."""
        return len(self.placements)

    def can_fit_weight(self, weight: float) -> bool:
        """Check if bin can accommodate additional weight.
        
        Args:
            weight: Weight to add in kg.
        
        Returns:
            True if weight fits or no weight limit is set.
        """
        if self.max_weight is None:
            return True
        return self.total_weight + weight <= self.max_weight

    def add_placement(self, placement: "Placement") -> None:
        """Add a placement to this bin.
        
        Args:
            placement: The placement to add.
        """
        self.placements.append(placement)

    def __repr__(self) -> str:
        return (
            f"Bin(id={self.id}, "
            f"dims=({self.length}x{self.width}x{self.height}), "
            f"boxes={self.box_count}, "
            f"util={self.utilization_percent:.1f}%)"
        )
