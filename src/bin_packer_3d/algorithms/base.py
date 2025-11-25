"""Abstract base class for packing algorithms.

This module defines the interface that all packing algorithms must implement,
enabling easy comparison and swapping of different strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import time

if TYPE_CHECKING:
    from bin_packer_3d.models.box import Box
    from bin_packer_3d.models.placement import PlacementResult
    from bin_packer_3d.config import PackerConfig


class PackerBase(ABC):
    """Abstract base class for all packing algorithms.
    
    All packing algorithms should inherit from this class and implement
    the `_pack_impl` method.
    
    Attributes:
        config: Packer configuration with bin dimensions and options.
        name: Human-readable name of the algorithm.
    
    Example:
        >>> class MyPacker(PackerBase):
        ...     @property
        ...     def name(self) -> str:
        ...         return "My Custom Packer"
        ...     
        ...     def _pack_impl(self, boxes):
        ...         # Implementation here
        ...         pass
    """

    def __init__(self, config: "PackerConfig") -> None:
        """Initialize packer with configuration.
        
        Args:
            config: Packer configuration object.
        """
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Return human-readable name of the algorithm."""
        pass

    @abstractmethod
    def _pack_impl(self, boxes: list["Box"]) -> "PlacementResult":
        """Implement the actual packing logic.
        
        Args:
            boxes: List of boxes to pack.
        
        Returns:
            PlacementResult with bins and placements.
        """
        pass

    def pack(self, boxes: list["Box"]) -> "PlacementResult":
        """Pack boxes into bins.
        
        This is the main entry point that wraps the implementation
        with timing and preprocessing.
        
        Args:
            boxes: List of boxes to pack.
        
        Returns:
            PlacementResult containing bins, placements, and metrics.
        """
        if not boxes:
            from bin_packer_3d.models.placement import PlacementResult
            return PlacementResult(algorithm=self.name)
        
        # Preprocess: sort by volume descending (FFD principle)
        sorted_boxes = self._preprocess(boxes)
        
        # Time the packing operation
        start_time = time.perf_counter()
        result = self._pack_impl(sorted_boxes)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        # Set metadata
        result.algorithm = self.name
        result.elapsed_time_ms = elapsed_ms
        
        return result

    def _preprocess(self, boxes: list["Box"]) -> list["Box"]:
        """Preprocess boxes before packing.
        
        Default implementation sorts by volume descending (FFD).
        Subclasses can override for different sorting strategies.
        
        Args:
            boxes: Original box list.
        
        Returns:
            Preprocessed (sorted) box list.
        """
        return sorted(boxes, key=lambda b: b.volume, reverse=True)

    def _can_fit_box(
        self,
        box: "Box",
        space_w: float,
        space_h: float,
        space_l: float,
    ) -> tuple[bool, tuple[float, float, float] | None]:
        """Check if box fits in space with any rotation.
        
        Args:
            box: Box to check.
            space_w: Available width.
            space_h: Available height.
            space_l: Available length.
        
        Returns:
            Tuple of (fits, best_orientation) where orientation is (w, h, l)
            or None if no fit found.
        """
        if not self.config.allow_rotation:
            if (box.width <= space_w and 
                box.height <= space_h and 
                box.length <= space_l):
                return True, (box.width, box.height, box.length)
            return False, None
        
        # Try all orientations
        for w, h, l in box.orientations():
            if w <= space_w and h <= space_h and l <= space_l:
                return True, (w, h, l)
        
        return False, None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"
