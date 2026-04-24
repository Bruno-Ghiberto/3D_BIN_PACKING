"""Abstract base class for packing algorithms.

This module defines the interface that all packing algorithms must implement,
enabling easy comparison and swapping of different strategies.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from bin_packer_3d.config import PackerConfig
    from bin_packer_3d.models.bin import Bin
    from bin_packer_3d.models.box import Box
    from bin_packer_3d.models.placement import Placement, PlacementResult


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

    complexity: ClassVar[str] = "unknown"
    description: ClassVar[str] = "unknown"

    def __init__(self, config: PackerConfig) -> None:
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
    def _pack_impl(self, boxes: list[Box]) -> PlacementResult:
        """Implement the actual packing logic.

        Args:
            boxes: List of boxes to pack.

        Returns:
            PlacementResult with bins and placements.
        """
        pass

    def pack(self, boxes: list[Box]) -> PlacementResult:
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

    def _preprocess(self, boxes: list[Box]) -> list[Box]:
        """Preprocess boxes before packing.

        Default implementation sorts by volume descending (FFD).
        Subclasses can override for different sorting strategies.

        Args:
            boxes: Original box list.

        Returns:
            Preprocessed (sorted) box list.
        """
        return sorted(boxes, key=lambda b: b.volume, reverse=True)

    def _check_constraints(
        self,
        placement: Placement,
        box: Box,
        bin: Bin,
        existing_placements: list[Placement],
    ) -> bool:
        """Consult the constraint registry before accepting a placement.

        Iterates over ``self.config.constraints`` (added in US7 via
        :mod:`bin_packer_3d.constraints`) and returns ``True`` only if
        every constraint's ``check`` returns a passing
        :class:`~bin_packer_3d.constraints.base.ConstraintResult`.

        In Phase A the ``constraints`` field does not yet exist on
        ``PackerConfig`` — we use :func:`getattr` with an empty-list
        fallback so this hook is a forward-compatible no-op until US7's
        T131 adds the field. When the list is empty, this always
        returns ``True`` and behaviour is unchanged.

        Args:
            placement: The candidate placement under consideration.
            box: The box being placed.
            bin: The bin the placement would live in.
            existing_placements: Placements already accepted in this bin.

        Returns:
            ``True`` when every registered constraint passes (or when
            no constraints are configured). ``False`` when at least one
            constraint rejects the placement.
        """
        constraints = getattr(self.config, "constraints", [])
        for constraint in constraints:
            result = constraint.check(placement, box, bin, existing_placements)
            if not result.ok:
                return False
        return True

    def _can_fit_box(
        self,
        box: Box,
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
            if box.width <= space_w and box.height <= space_h and box.length <= space_l:
                return True, (box.width, box.height, box.length)
            return False, None

        # Try all orientations
        for w, h, length in box.orientations():
            if w <= space_w and h <= space_h and length <= space_l:
                return True, (w, h, length)

        return False, None

    def __repr__(self) -> str:
        """Return debug repr including the bound config."""
        return f"{self.__class__.__name__}(config={self.config})"
