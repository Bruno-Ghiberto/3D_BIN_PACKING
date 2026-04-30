"""Unit tests for data models."""

import pytest

from bin_packer_3d.models.bin import Bin
from bin_packer_3d.models.box import Box
from bin_packer_3d.models.placement import Placement, PlacementResult
from bin_packer_3d.models.result import LoadReport, PackingResult, RejectedRow


class TestBox:
    """Tests for Box model."""

    def test_box_creation(self, sample_box: Box) -> None:
        """Test basic box creation."""
        assert sample_box.id == "test_box_1"
        assert sample_box.width == 100
        assert sample_box.height == 50
        assert sample_box.length == 80

    def test_box_volume(self, sample_box: Box) -> None:
        """Test volume calculation."""
        expected = 100 * 50 * 80
        assert sample_box.volume == expected

    def test_box_volume_m3(self, sample_box: Box) -> None:
        """Test volume in cubic meters."""
        expected = (100 * 50 * 80) / 1_000_000_000
        assert sample_box.volume_m3 == expected

    def test_box_footprint(self, sample_box: Box) -> None:
        """Test footprint calculation."""
        expected = 100 * 80
        assert sample_box.footprint == expected

    def test_box_dimensions_tuple(self, sample_box: Box) -> None:
        """Test dimensions property."""
        assert sample_box.dimensions == (100, 50, 80)

    def test_box_orientations(self, sample_box: Box) -> None:
        """Test that box generates correct orientations."""
        orientations = list(sample_box.orientations())

        # All dimensions unique, so should have 6 orientations
        assert len(orientations) == 6

        # Original orientation should be included
        assert (100, 50, 80) in orientations

    def test_box_orientations_with_duplicates(self) -> None:
        """Test orientations for cube (all equal dimensions)."""
        cube = Box(id="cube", width=50, height=50, length=50)
        orientations = list(cube.orientations())

        # Cube only has 1 unique orientation
        assert len(orientations) == 1
        assert orientations[0] == (50, 50, 50)

    def test_box_fits_in_container(self, sample_box: Box) -> None:
        """Test fits_in method."""
        # Should fit in larger container
        assert sample_box.fits_in(200, 200, 200, allow_rotation=True)

        # Should not fit in smaller container
        assert not sample_box.fits_in(10, 10, 10, allow_rotation=True)

    def test_box_fits_with_rotation(self) -> None:
        """Test that rotation helps fitting."""
        box = Box(id="tall", width=50, height=200, length=50)

        # Won't fit without rotation (height too tall)
        assert not box.fits_in(100, 100, 100, allow_rotation=False)

        # Will fit with rotation (turn on side)
        assert box.fits_in(200, 100, 100, allow_rotation=True)

    def test_box_validation_negative_dimensions(self) -> None:
        """Test that negative dimensions raise error."""
        with pytest.raises(ValueError):
            Box(id="bad", width=-10, height=50, length=80)

    def test_box_validation_zero_dimension(self) -> None:
        """Test that zero dimensions raise error."""
        with pytest.raises(ValueError):
            Box(id="bad", width=0, height=50, length=80)

    def test_weight_unknown_distinguishable(self) -> None:
        """Box.weight None (unknown) is distinguishable from 0.0 (known zero) per FR-005 (T019)."""
        unknown = Box(id="unknown", width=10, height=10, length=10, weight=None)
        zero = Box(id="zero", width=10, height=10, length=10, weight=0.0)

        assert unknown.weight is None
        assert zero.weight == 0.0
        assert unknown.weight != zero.weight

    def test_weight_default_is_none(self) -> None:
        """Box() with no explicit weight defaults to None (unknown), not 0.0 (FR-005, T019)."""
        box = Box(id="default", width=10, height=10, length=10)

        assert box.weight is None


class TestBin:
    """Tests for Bin model."""

    def test_bin_creation(self, sample_bin: Bin) -> None:
        """Test basic bin creation."""
        assert sample_bin.id == 1
        assert sample_bin.length == 860
        assert sample_bin.width == 890
        assert sample_bin.height == 1040

    def test_bin_volume(self, sample_bin: Bin) -> None:
        """Test bin volume calculation."""
        expected = 860 * 890 * 1040
        assert sample_bin.volume == expected

    def test_bin_initial_utilization(self, sample_bin: Bin) -> None:
        """Test empty bin has 0% utilization."""
        assert sample_bin.utilization_percent == 0.0
        assert sample_bin.box_count == 0

    def test_bin_add_placement(self, sample_bin: Bin, sample_box: Box) -> None:
        """Test adding placement to bin."""
        placement = Placement(
            box=sample_box,
            bin_id=sample_bin.id,
            x0=0,
            y0=0,
            z0=0,
            x1=100,
            y1=80,
            z1=50,
        )

        sample_bin.add_placement(placement)

        assert sample_bin.box_count == 1
        assert sample_bin.utilization_percent > 0

    def test_bin_weight_capacity(self) -> None:
        """Test weight capacity checking."""
        bin_with_weight = Bin(id=1, length=100, width=100, height=100, max_weight=10.0)

        assert bin_with_weight.can_fit_weight(5.0)
        assert bin_with_weight.can_fit_weight(10.0)
        assert not bin_with_weight.can_fit_weight(15.0)


class TestPlacement:
    """Tests for Placement model."""

    def test_placement_creation(self, sample_box: Box) -> None:
        """Test basic placement creation."""
        placement = Placement(
            box=sample_box,
            bin_id=1,
            x0=0,
            y0=0,
            z0=0,
            x1=100,
            y1=80,
            z1=50,
        )

        assert placement.box == sample_box
        assert placement.bin_id == 1
        assert placement.width == 100
        assert placement.depth == 80
        assert placement.height == 50

    def test_placement_volume(self, sample_box: Box) -> None:
        """Test placement volume calculation."""
        placement = Placement(
            box=sample_box,
            bin_id=1,
            x0=0,
            y0=0,
            z0=0,
            x1=100,
            y1=80,
            z1=50,
        )

        assert placement.volume == 100 * 80 * 50

    def test_placement_overlap_detection(self, sample_box: Box) -> None:
        """Test overlap detection between placements."""
        p1 = Placement(box=sample_box, bin_id=1, x0=0, y0=0, z0=0, x1=100, y1=100, z1=100)
        p2 = Placement(box=sample_box, bin_id=1, x0=50, y0=50, z0=50, x1=150, y1=150, z1=150)
        p3 = Placement(box=sample_box, bin_id=1, x0=200, y0=200, z0=200, x1=300, y1=300, z1=300)

        assert p1.overlaps_with(p2)
        assert p2.overlaps_with(p1)
        assert not p1.overlaps_with(p3)

    def test_placement_to_dict(self, sample_box: Box) -> None:
        """Test conversion to dictionary."""
        placement = Placement(
            box=sample_box,
            bin_id=1,
            x0=0,
            y0=0,
            z0=0,
            x1=100,
            y1=80,
            z1=50,
        )

        d = placement.to_dict()

        assert d["BOX_ID"] == sample_box.id
        assert d["BIN_ID"] == 1
        assert d["x0"] == 0


class TestPlacementResult:
    """Tests for PlacementResult model."""

    def test_empty_result(self) -> None:
        """Test empty result metrics."""
        result = PlacementResult()

        assert result.total_boxes == 0
        assert result.placed_count == 0
        assert result.bins_used == 0
        assert result.utilization_percent == 0.0
        assert result.success_rate == 100.0  # No boxes = 100% success

    def test_result_summary(self) -> None:
        """Test summary generation."""
        result = PlacementResult(algorithm="Test Algorithm")
        summary = result.summary()

        assert "Test Algorithm" in summary
        assert "Boxes:" in summary


class TestPackingResult:
    """Tests for the new pydantic-based PackingResult (data-model.md §new entities)."""

    def _make_placement(self, box: Box, bin_id: int = 1) -> Placement:
        """Helper — construct a placement at the origin sized to the box."""
        return Placement(
            box=box,
            bin_id=bin_id,
            x0=0.0,
            y0=0.0,
            z0=0.0,
            x1=box.width,
            y1=box.length,
            z1=box.height,
        )

    def test_empty_result_has_100_percent_success(self) -> None:
        """Empty input edge case: success_rate defaults to 100.0.

        Spec §Edge Cases: "an empty box list passed to the packing entry
        point returns an empty result with success_rate == 100.0".
        """
        result = PackingResult()
        assert result.placements == []
        assert result.unpacked_boxes == []
        assert result.bins_used == 0
        assert result.volume_utilisation == 0.0
        assert result.success_rate == 100.0
        assert result.algorithm == ""
        assert result.runtime_seconds == 0.0

    def test_box_count_conservation(self) -> None:
        """Placements + unpacked count == input count (FR-046)."""
        boxes = [Box(id=f"B{i}", width=10, height=10, length=10) for i in range(5)]
        placed = [self._make_placement(b) for b in boxes[:3]]
        unpacked = boxes[3:]

        result = PackingResult(
            placements=placed,
            unpacked_boxes=unpacked,
            bins_used=1,
            algorithm="ffd",
        )
        total = len(result.placements) + len(result.unpacked_boxes)
        assert total == len(boxes), (
            f"box-count conservation violated: placed={len(result.placements)} "
            f"+ unpacked={len(result.unpacked_boxes)} != input={len(boxes)}"
        )

    def test_volume_conservation(self) -> None:
        """Sum of placed + unpacked box volumes == sum of input volumes (FR-046)."""
        boxes = [Box(id=f"B{i}", width=10.0, height=10.0, length=10.0) for i in range(4)]
        placed = [self._make_placement(b) for b in boxes[:2]]
        unpacked = boxes[2:]

        result = PackingResult(
            placements=placed,
            unpacked_boxes=unpacked,
            bins_used=1,
            algorithm="ffd",
        )
        total_input_volume = sum(b.volume for b in boxes)
        total_placed_volume = sum(p.box.volume for p in result.placements)
        total_unpacked_volume = sum(b.volume for b in result.unpacked_boxes)
        assert total_placed_volume + total_unpacked_volume == pytest.approx(total_input_volume), (
            "volume conservation violated"
        )

    def test_bins_used_must_be_non_negative(self) -> None:
        """Pydantic validator rejects bins_used < 0."""
        with pytest.raises(ValueError):
            PackingResult(bins_used=-1)

    def test_volume_utilisation_bounded(self) -> None:
        """Pydantic validator enforces volume_utilisation in [0.0, 1.0]."""
        with pytest.raises(ValueError):
            PackingResult(volume_utilisation=1.5)
        with pytest.raises(ValueError):
            PackingResult(volume_utilisation=-0.1)

    def test_success_rate_bounded(self) -> None:
        """Pydantic validator enforces success_rate in [0.0, 100.0]."""
        with pytest.raises(ValueError):
            PackingResult(success_rate=101.0)
        with pytest.raises(ValueError):
            PackingResult(success_rate=-1.0)

    def test_runtime_seconds_non_negative(self) -> None:
        """runtime_seconds cannot be negative."""
        with pytest.raises(ValueError):
            PackingResult(runtime_seconds=-0.001)


class TestLoadReport:
    """Tests for LoadReport (FR-053 — malformed rows surfaced via structured report)."""

    def test_empty_load_report(self) -> None:
        """LoadReport defaults are all empty collections."""
        report = LoadReport()
        assert report.boxes == []
        assert report.rejected_rows == []
        assert report.warnings == []

    def test_load_report_accepts_rejected_rows(self) -> None:
        """LoadReport carries RejectedRow instances with structured reason codes."""
        rr = RejectedRow(
            row_number=5,
            reason="missing_dimension",
            raw={"id": "B001", "width": 10},
        )
        report = LoadReport(rejected_rows=[rr], warnings=["row 5 dropped"])
        assert len(report.rejected_rows) == 1
        assert report.rejected_rows[0].reason == "missing_dimension"
        assert report.rejected_rows[0].row_number == 5


class TestRejectedRow:
    """Tests for RejectedRow frozen dataclass."""

    def test_rejected_row_is_immutable(self) -> None:
        """RejectedRow is frozen — fields cannot be reassigned."""
        from dataclasses import FrozenInstanceError

        rr = RejectedRow(row_number=1, reason="x", raw={})
        with pytest.raises(FrozenInstanceError):
            rr.row_number = 2  # type: ignore[misc]

    def test_rejected_row_fields(self) -> None:
        """RejectedRow carries row_number, reason, raw payload."""
        rr = RejectedRow(
            row_number=3,
            reason="schema_error",
            raw={"bogus": "payload"},
        )
        assert rr.row_number == 3
        assert rr.reason == "schema_error"
        assert rr.raw == {"bogus": "payload"}
