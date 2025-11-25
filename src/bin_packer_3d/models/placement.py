"""Placement models for tracking box positions.

This module defines classes for tracking where boxes are placed
within bins, including 3D coordinates and packing results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bin_packer_3d.models.box import Box
    from bin_packer_3d.models.bin import Bin


@dataclass
class Placement:
    """A box placement with 3D coordinates.
    
    Represents a box placed at a specific position within a bin,
    with coordinates defining the bounding box corners.
    
    Attributes:
        box: The box that was placed.
        bin_id: ID of the bin containing this placement.
        shelf_id: ID of the shelf within the bin (for shelf-based packing).
        x0, y0, z0: Minimum corner coordinates in mm.
        x1, y1, z1: Maximum corner coordinates in mm.
    
    The coordinate system:
        - X: Length axis (0 to bin_length)
        - Y: Width axis (0 to bin_width)
        - Z: Height axis (0 to bin_height, bottom to top)
    """

    box: "Box"
    bin_id: int
    x0: float
    y0: float
    z0: float
    x1: float
    y1: float
    z1: float
    shelf_id: int = 0

    def __post_init__(self) -> None:
        """Validate coordinate ordering."""
        if self.x0 > self.x1 or self.y0 > self.y1 or self.z0 > self.z1:
            raise ValueError(
                f"Invalid coordinates: min must be <= max. "
                f"Got: ({self.x0},{self.y0},{self.z0}) to ({self.x1},{self.y1},{self.z1})"
            )

    @property
    def width(self) -> float:
        """Placed width (x1 - x0)."""
        return self.x1 - self.x0

    @property
    def depth(self) -> float:
        """Placed depth (y1 - y0)."""
        return self.y1 - self.y0

    @property
    def height(self) -> float:
        """Placed height (z1 - z0)."""
        return self.z1 - self.z0

    @property
    def volume(self) -> float:
        """Calculate placed volume in cubic mm."""
        return self.width * self.depth * self.height

    @property
    def center(self) -> tuple[float, float, float]:
        """Return center point coordinates."""
        return (
            (self.x0 + self.x1) / 2,
            (self.y0 + self.y1) / 2,
            (self.z0 + self.z1) / 2,
        )

    def overlaps_with(self, other: "Placement") -> bool:
        """Check if this placement overlaps with another.
        
        Args:
            other: Another placement to check against.
        
        Returns:
            True if the placements overlap in 3D space.
        """
        # No overlap if separated on any axis
        if self.x1 <= other.x0 or other.x1 <= self.x0:
            return False
        if self.y1 <= other.y0 or other.y1 <= self.y0:
            return False
        if self.z1 <= other.z0 or other.z1 <= self.z0:
            return False
        return True

    def to_dict(self) -> dict[str, float | int | str]:
        """Convert placement to dictionary for DataFrame export."""
        return {
            "BOX_ID": self.box.id,
            "BIN_ID": self.bin_id,
            "SHELF_ID": self.shelf_id,
            "x0": self.x0,
            "y0": self.y0,
            "z0": self.z0,
            "x1": self.x1,
            "y1": self.y1,
            "z1": self.z1,
            "CAJA": self.box.box_type,
            "DESCRIPCION": self.box.description,
            "CANTIDAD": self.box.quantity,
        }

    def __repr__(self) -> str:
        return (
            f"Placement(box={self.box.id}, bin={self.bin_id}, "
            f"pos=({self.x0:.0f},{self.y0:.0f},{self.z0:.0f})-"
            f"({self.x1:.0f},{self.y1:.0f},{self.z1:.0f}))"
        )


@dataclass
class PlacementResult:
    """Result of a packing operation.
    
    Contains all information about a completed packing run including
    placed boxes, bins used, and comprehensive metrics.
    
    Attributes:
        bins: List of bins with placements.
        unpacked_boxes: Boxes that couldn't be placed.
        algorithm: Name of algorithm used.
        elapsed_time_ms: Time taken in milliseconds.
    """

    bins: list["Bin"] = field(default_factory=list)
    unpacked_boxes: list["Box"] = field(default_factory=list)
    algorithm: str = ""
    elapsed_time_ms: float = 0.0

    @property
    def total_boxes(self) -> int:
        """Total number of boxes attempted."""
        return self.placed_count + len(self.unpacked_boxes)

    @property
    def placed_count(self) -> int:
        """Number of successfully placed boxes."""
        return sum(bin.box_count for bin in self.bins)

    @property
    def bins_used(self) -> int:
        """Number of bins used."""
        return len(self.bins)

    @property
    def total_box_volume(self) -> float:
        """Total volume of all placed boxes in cubic mm."""
        return sum(bin.used_volume for bin in self.bins)

    @property
    def total_bin_volume(self) -> float:
        """Total volume of all bins used in cubic mm."""
        return sum(bin.volume for bin in self.bins)

    @property
    def utilization_percent(self) -> float:
        """Overall space utilization percentage (0-100)."""
        if self.total_bin_volume == 0:
            return 0.0
        return (self.total_box_volume / self.total_bin_volume) * 100

    @property
    def success_rate(self) -> float:
        """Percentage of boxes successfully placed (0-100)."""
        if self.total_boxes == 0:
            return 100.0
        return (self.placed_count / self.total_boxes) * 100

    def all_placements(self) -> list[Placement]:
        """Return flat list of all placements across bins."""
        placements: list[Placement] = []
        for bin in self.bins:
            placements.extend(bin.placements)
        return placements

    def summary(self) -> str:
        """Generate human-readable summary."""
        return (
            f"Packing Result ({self.algorithm}):\n"
            f"  Boxes: {self.placed_count}/{self.total_boxes} placed "
            f"({self.success_rate:.1f}%)\n"
            f"  Bins: {self.bins_used} used\n"
            f"  Utilization: {self.utilization_percent:.1f}%\n"
            f"  Time: {self.elapsed_time_ms:.2f}ms"
        )

    def __repr__(self) -> str:
        return (
            f"PlacementResult(placed={self.placed_count}, "
            f"bins={self.bins_used}, util={self.utilization_percent:.1f}%)"
        )
