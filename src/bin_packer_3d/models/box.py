"""Box model representing items to be packed.

This module defines the Box class which represents rectangular items
with support for 6 rotation orientations.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class Box:
    """A rectangular box with dimensions and rotation support.
    
    Attributes:
        id: Unique identifier for the box.
        width: Width dimension in mm (W).
        height: Height dimension in mm (H).
        length: Length dimension in mm (L).
        weight: Weight in kg (optional).
        box_type: Type/category of box for grouping.
        description: Human-readable description.
        quantity: Number of this box type (for tracking).
        metadata: Additional arbitrary data.
    
    Example:
        >>> box = Box(id="B001", width=100, height=50, length=80)
        >>> box.volume
        400000.0
        >>> list(box.orientations())
        [(100, 50, 80), (100, 80, 50), (50, 100, 80), ...]
    """

    id: str
    width: float
    height: float
    length: float
    weight: float = 0.0
    box_type: str = ""
    description: str = ""
    quantity: int = 1
    metadata: dict[str, str | int | float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate dimensions after initialization."""
        if self.width <= 0 or self.height <= 0 or self.length <= 0:
            raise ValueError(
                f"All dimensions must be positive. Got: "
                f"w={self.width}, h={self.height}, l={self.length}"
            )
        if self.weight < 0:
            raise ValueError(f"Weight cannot be negative. Got: {self.weight}")

    @property
    def volume(self) -> float:
        """Calculate box volume in cubic mm."""
        return self.width * self.height * self.length

    @property
    def volume_m3(self) -> float:
        """Calculate box volume in cubic meters."""
        return self.volume / 1_000_000_000

    @property
    def footprint(self) -> float:
        """Calculate base footprint (width * length) in sq mm."""
        return self.width * self.length

    @property
    def dimensions(self) -> tuple[float, float, float]:
        """Return dimensions as (width, height, length) tuple."""
        return (self.width, self.height, self.length)

    def orientations(self) -> Iterator[tuple[float, float, float]]:
        """Generate all 6 possible rotation orientations.
        
        Each orientation is a (width, height, length) tuple representing
        the box dimensions when rotated to that position.
        
        Yields:
            Tuples of (w, h, l) for each unique orientation.
        
        Note:
            Duplicate orientations (for boxes with equal dimensions)
            are automatically filtered using set().
        """
        dims = [self.width, self.height, self.length]
        # Use set to remove duplicates for boxes with equal dimensions
        unique_perms = set(itertools.permutations(dims, 3))
        yield from unique_perms

    def orientation_count(self) -> int:
        """Return number of unique orientations (1-6)."""
        return len(set(itertools.permutations(self.dimensions, 3)))

    def fits_in(
        self,
        container_w: float,
        container_h: float,
        container_l: float,
        allow_rotation: bool = True,
    ) -> bool:
        """Check if box can fit in given container dimensions.
        
        Args:
            container_w: Container width.
            container_h: Container height.
            container_l: Container length.
            allow_rotation: Whether to try all rotations.
        
        Returns:
            True if box fits in any orientation (or fixed if no rotation).
        """
        if allow_rotation:
            for w, h, l in self.orientations():
                if w <= container_w and h <= container_h and l <= container_l:
                    return True
            return False
        else:
            return (
                self.width <= container_w
                and self.height <= container_h
                and self.length <= container_l
            )

    def __repr__(self) -> str:
        return (
            f"Box(id={self.id!r}, "
            f"w={self.width}, h={self.height}, l={self.length}, "
            f"vol={self.volume:.0f}mm³)"
        )
