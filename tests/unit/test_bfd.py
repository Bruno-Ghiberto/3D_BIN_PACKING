"""Unit tests for Best-Fit Decreasing packer (T086, FR-040).

T086 lives in its own module rather than appending to
``tests/unit/test_algorithms.py`` so the test-first ``ImportError``
from the missing ``bin_packer_3d.algorithms.bfd`` does NOT take down
the existing TestFirstFitDecreasingPacker / TestShelfPacker classes
during the red phase. Once T092 lands the BFD implementation, all
tests in this module turn green together (no manipulation of the
sibling FFD test class). Resolved per implement-context §7 step 4
(low-stakes test-design default).

The hand-computed example uses the same fitting logic FFD relies on
(``_can_fit_box`` + guillotine splits) so any divergence between FFD
and BFD is purely the bin-selection heuristic — best-remaining-volume
vs first-fit. SC-006's ≥5pp utilisation OR ≥1 bin delta is the
business outcome BFD must clear; this test only verifies the
correctness floor (all boxes placed, invariants hold).
"""

from __future__ import annotations

from bin_packer_3d.algorithms.bfd import BestFitDecreasingPacker
from bin_packer_3d.config import PackerConfig
from bin_packer_3d.models.box import Box


class TestBestFitDecreasingPacker:
    """Tests for the BFD algorithm — correctness floor + name + registration."""

    def test_pack_single_box(self) -> None:
        """A single small box lands in one bin with 100% success."""
        config = PackerConfig(bin_length=200, bin_width=200, bin_height=200)
        packer = BestFitDecreasingPacker(config)

        result = packer.pack([Box(id="b1", width=50, height=50, length=50)])

        assert result.placed_count == 1
        assert result.bins_used == 1
        assert result.success_rate == 100.0

    def test_pack_empty_list(self) -> None:
        """Empty input yields an empty, success-100% result (spec §Edge Cases)."""
        config = PackerConfig(bin_length=100, bin_width=100, bin_height=100)
        packer = BestFitDecreasingPacker(config)

        result = packer.pack([])

        assert result.placed_count == 0
        assert result.bins_used == 0
        assert result.success_rate == 100.0

    def test_pack_oversized_box_unpacked(self) -> None:
        """A box larger than the bin in every orientation is reported unpacked."""
        config = PackerConfig(bin_length=100, bin_width=100, bin_height=100)
        packer = BestFitDecreasingPacker(config)

        oversized = Box(id="big", width=200, height=200, length=200)
        result = packer.pack([oversized])

        assert result.placed_count == 0
        assert len(result.unpacked_boxes) == 1
        assert result.success_rate == 0.0

    def test_pack_hand_computed_filling(self) -> None:
        """BFD packs the canonical four-box example without leaving any unpacked.

        Bin 100x100x100; boxes A(50³) B(50³) C(40³) D(30³). All four
        fit comfortably (combined volume 175 000 << 1 000 000) so a
        correct BFD implementation places everything; bins_used >= 1.
        """
        config = PackerConfig(bin_length=100, bin_width=100, bin_height=100)
        packer = BestFitDecreasingPacker(config)

        boxes = [
            Box(id="A", width=50, height=50, length=50),
            Box(id="B", width=50, height=50, length=50),
            Box(id="C", width=40, height=40, length=40),
            Box(id="D", width=30, height=30, length=30),
        ]
        result = packer.pack(boxes)

        assert result.placed_count == 4
        assert len(result.unpacked_boxes) == 0
        assert result.bins_used >= 1
        assert result.utilization_percent > 0

    def test_algorithm_name_includes_best_fit(self) -> None:
        """The algorithm name advertises the strategy ('Best-Fit')."""
        config = PackerConfig(bin_length=100, bin_width=100, bin_height=100)
        packer = BestFitDecreasingPacker(config)

        assert "Best-Fit" in packer.name

    def test_registered_under_bfd_key(self) -> None:
        """The packer registers itself under the 'bfd' key in ALGORITHMS."""
        from bin_packer_3d.algorithms import ALGORITHMS

        assert "bfd" in ALGORITHMS
        assert ALGORITHMS["bfd"] is BestFitDecreasingPacker

    def test_complexity_classvar_set(self) -> None:
        """The ``complexity`` classvar is set to a non-empty string for ``bin-packer info``."""
        assert BestFitDecreasingPacker.complexity != "unknown"
        assert BestFitDecreasingPacker.complexity.strip() != ""

    def test_description_classvar_set(self) -> None:
        """The ``description`` classvar is set to a non-empty string."""
        assert BestFitDecreasingPacker.description != "unknown"
        assert BestFitDecreasingPacker.description.strip() != ""
