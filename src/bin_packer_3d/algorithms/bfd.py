"""Best-Fit Decreasing (BFD) packing algorithm (T092, FR-040).

BFD differs from First-Fit Decreasing on a single decision: when a box
could fit in more than one already-opened bin, FFD takes the lowest
indexed bin (first-fit) while BFD picks the bin with the **smallest
total remaining free volume** that still accommodates the box. The
heuristic concentrates new placements into bins that are already
filling up, leaving the freshest bins for boxes that genuinely need
the space — a textbook variant covered in Coffman, Garey & Johnson
1996 §"Bin packing approximation algorithms".

Both algorithms share the same fitting test (``_can_fit_box``) and the
same guillotine-style space split, so any difference in
``volume_utilisation`` or ``n_bins_used`` between FFD and BFD on the
same instance is purely the bin-selection heuristic. SC-006 measures
exactly this delta against the BR1 reference instance and gates v0.3.0
on at least one new algorithm achieving ≥ 5 percentage points
utilisation OR ≥ 1 bin delta vs FFD.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from bin_packer_3d.algorithms.base import PackerBase
from bin_packer_3d.algorithms.registry import register
from bin_packer_3d.models.bin import Bin
from bin_packer_3d.models.placement import Placement, PlacementResult

if TYPE_CHECKING:
    from bin_packer_3d.config import PackerConfig
    from bin_packer_3d.models.box import Box


@register("bfd")
class BestFitDecreasingPacker(PackerBase):
    """Best-Fit Decreasing packer — concentrates placements into fuller bins.

    Sort boxes by volume descending (inherited :meth:`_preprocess`), then
    for each box:

    1. Scan every already-opened bin and check whether the box fits in
       any of its remaining spaces (no mutation — pure inspection).
    2. Among bins where it fits, pick the one with the smallest summed
       free-space volume — the "tightest" bin still capable of holding
       the box.
    3. Place there. Open a new bin only when no existing bin fits.

    The constraint hook (``_check_constraints``) is consulted before
    every placement is accepted, mirroring FFD's contract so US7's
    constraint framework drops in without algorithm edits.

    Complexity:
        Time: O(n log n) sort + O(n × m × s) placement, where ``m`` is
        the number of opened bins and ``s`` the average free-space
        count per bin. Same asymptotic bound as FFD; the constant is
        slightly larger because BFD evaluates every existing bin
        before placing.
        Space: O(n + m × s) for placements and free-space lists.
    """

    complexity: ClassVar[str] = "O(n log n)"
    description: ClassVar[str] = "Best Fit Decreasing — concentrate fills"

    def __init__(self, config: PackerConfig) -> None:
        """Initialise the BFD packer with the given configuration."""
        super().__init__(config)
        self._bins: list[Bin] = []
        self._bin_spaces: list[list[tuple[float, float, float, float, float, float]]] = []

    @property
    def name(self) -> str:
        """Return the human-readable algorithm name."""
        return "Best-Fit Decreasing (BFD)"

    def _pack_impl(self, boxes: list[Box]) -> PlacementResult:
        """Run BFD over the pre-sorted box list."""
        self._bins = []
        self._bin_spaces = []
        unpacked: list[Box] = []

        for box in boxes:
            best_bin_idx = self._select_best_existing_bin(box)

            if best_bin_idx is not None:
                bin_obj = self._bins[best_bin_idx]
                placement = self._try_place_in_bin(box, best_bin_idx, bin_obj)
                if placement is not None and self._check_constraints(
                    placement, box, bin_obj, bin_obj.placements
                ):
                    bin_obj.add_placement(placement)
                    continue
                # If a constraint rejected the otherwise-fitting placement,
                # fall through to open a fresh bin (matches FFD semantics).

            new_bin = Bin(
                id=len(self._bins) + 1,
                length=self.config.bin_length,
                width=self.config.bin_width,
                height=self.config.bin_height,
                max_weight=self.config.max_weight,
            )
            self._bins.append(new_bin)
            self._bin_spaces.append(
                [
                    (
                        0.0,
                        0.0,
                        0.0,
                        self.config.bin_length,
                        self.config.bin_width,
                        self.config.bin_height,
                    )
                ]
            )

            placement = self._try_place_in_bin(box, len(self._bins) - 1, new_bin)
            if placement is not None and self._check_constraints(
                placement, box, new_bin, new_bin.placements
            ):
                new_bin.add_placement(placement)
            else:
                self._bins.pop()
                self._bin_spaces.pop()
                unpacked.append(box)

        return PlacementResult(bins=self._bins, unpacked_boxes=unpacked)

    def _select_best_existing_bin(self, box: Box) -> int | None:
        """Return the index of the tightest existing bin that fits ``box``.

        "Tightest" = smallest sum of free-space volumes among bins that
        also pass the weight check and have at least one space large
        enough to hold the box in some orientation. Returns ``None`` if
        no existing bin fits.
        """
        best_idx: int | None = None
        best_remaining = float("inf")

        for bin_idx, bin_obj in enumerate(self._bins):
            if not bin_obj.can_fit_weight(box.weight):
                continue
            if not self._has_fitting_space(box, bin_idx):
                continue
            remaining = sum(
                (sx1 - sx0) * (sy1 - sy0) * (sz1 - sz0)
                for sx0, sy0, sz0, sx1, sy1, sz1 in self._bin_spaces[bin_idx]
            )
            if remaining < best_remaining:
                best_remaining = remaining
                best_idx = bin_idx

        return best_idx

    def _has_fitting_space(self, box: Box, bin_idx: int) -> bool:
        """Return ``True`` iff any free space in bin ``bin_idx`` accommodates ``box``."""
        for sx0, sy0, sz0, sx1, sy1, sz1 in self._bin_spaces[bin_idx]:
            space_w = sx1 - sx0
            space_h = sz1 - sz0
            space_l = sy1 - sy0
            fits, _ = self._can_fit_box(box, space_w, space_h, space_l)
            if fits:
                return True
        return False

    def _try_place_in_bin(
        self,
        box: Box,
        bin_idx: int,
        bin_obj: Bin,
    ) -> Placement | None:
        """Try to place box in a specific bin using guillotine-style splits."""
        if not bin_obj.can_fit_weight(box.weight):
            return None

        spaces = self._bin_spaces[bin_idx]

        for space_idx, (sx0, sy0, sz0, sx1, sy1, sz1) in enumerate(spaces):
            space_w = sx1 - sx0
            space_h = sz1 - sz0
            space_l = sy1 - sy0

            fits, orientation = self._can_fit_box(box, space_w, space_h, space_l)
            if fits and orientation is not None:
                w, h, length = orientation
                placement = Placement(
                    box=box,
                    bin_id=bin_obj.id,
                    x0=sx0,
                    y0=sy0,
                    z0=sz0,
                    x1=sx0 + w,
                    y1=sy0 + length,
                    z1=sz0 + h,
                )
                self._split_space(bin_idx, space_idx, w, length, h)
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
        """Split the just-used free space into up to three guillotine-cut sub-spaces."""
        spaces = self._bin_spaces[bin_idx]
        sx0, sy0, sz0, sx1, sy1, sz1 = spaces[space_idx]
        del spaces[space_idx]

        remaining_w = sx1 - sx0 - used_w
        if remaining_w > 0:
            spaces.append((sx0 + used_w, sy0, sz0, sx1, sy1, sz1))

        remaining_l = sy1 - sy0 - used_l
        if remaining_l > 0:
            spaces.append((sx0, sy0 + used_l, sz0, sx0 + used_w, sy1, sz1))

        remaining_h = sz1 - sz0 - used_h
        if remaining_h > 0:
            spaces.append((sx0, sy0, sz0 + used_h, sx0 + used_w, sy0 + used_l, sz1))
