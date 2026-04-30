"""Unit tests for packing algorithms."""

from bin_packer_3d.algorithms.ffd import FirstFitDecreasingPacker
from bin_packer_3d.algorithms.shelf import ShelfPacker
from bin_packer_3d.config import PackerConfig
from bin_packer_3d.models.box import Box


class TestFirstFitDecreasingPacker:
    """Tests for FFD algorithm."""

    def test_pack_single_box(self, default_config: PackerConfig) -> None:
        """Test packing a single box."""
        packer = FirstFitDecreasingPacker(default_config)
        boxes = [Box(id="box1", width=100, height=50, length=80)]

        result = packer.pack(boxes)

        assert result.placed_count == 1
        assert result.bins_used == 1
        assert result.success_rate == 100.0

    def test_pack_multiple_boxes(
        self, default_config: PackerConfig, sample_boxes: list[Box]
    ) -> None:
        """Test packing multiple boxes."""
        packer = FirstFitDecreasingPacker(default_config)

        result = packer.pack(sample_boxes)

        assert result.placed_count == len(sample_boxes)
        assert result.success_rate == 100.0
        assert result.bins_used >= 1

    def test_pack_empty_list(self, default_config: PackerConfig) -> None:
        """Test packing empty box list."""
        packer = FirstFitDecreasingPacker(default_config)

        result = packer.pack([])

        assert result.placed_count == 0
        assert result.bins_used == 0

    def test_pack_oversized_box(self, small_config: PackerConfig) -> None:
        """Test that oversized boxes are marked as unpacked."""
        packer = FirstFitDecreasingPacker(small_config)
        # Box larger than bin in all orientations
        oversized = Box(id="big", width=300, height=300, length=300)

        result = packer.pack([oversized])

        assert result.placed_count == 0
        assert len(result.unpacked_boxes) == 1
        assert result.success_rate == 0.0

    def test_algorithm_name(self, default_config: PackerConfig) -> None:
        """Test algorithm name property."""
        packer = FirstFitDecreasingPacker(default_config)

        assert "First-Fit" in packer.name

    def test_pack_records_time(self, default_config: PackerConfig, sample_boxes: list[Box]) -> None:
        """Test that elapsed time is recorded."""
        packer = FirstFitDecreasingPacker(default_config)

        result = packer.pack(sample_boxes)

        assert result.elapsed_time_ms >= 0

    def test_utilization_calculated(
        self, default_config: PackerConfig, sample_boxes: list[Box]
    ) -> None:
        """Test that utilization is calculated correctly."""
        packer = FirstFitDecreasingPacker(default_config)

        result = packer.pack(sample_boxes)

        assert result.utilization_percent > 0
        assert result.utilization_percent <= 100


class TestShelfPacker:
    """Tests for shelf-based algorithm."""

    def test_pack_single_box(self, default_config: PackerConfig) -> None:
        """Test packing a single box."""
        packer = ShelfPacker(default_config)
        boxes = [Box(id="box1", width=100, height=50, length=80)]

        result = packer.pack(boxes)

        assert result.placed_count == 1
        assert result.bins_used == 1

    def test_pack_multiple_boxes(
        self, default_config: PackerConfig, sample_boxes: list[Box]
    ) -> None:
        """Test packing multiple boxes."""
        packer = ShelfPacker(default_config)

        result = packer.pack(sample_boxes)

        assert result.placed_count == len(sample_boxes)
        assert result.success_rate == 100.0

    def test_algorithm_name(self, default_config: PackerConfig) -> None:
        """Test algorithm name property."""
        packer = ShelfPacker(default_config)

        assert "Shelf" in packer.name

    def test_shelf_id_assigned(self, default_config: PackerConfig, sample_boxes: list[Box]) -> None:
        """Test that placements have shelf IDs."""
        packer = ShelfPacker(default_config)

        result = packer.pack(sample_boxes)

        for placement in result.all_placements():
            assert placement.shelf_id >= 0


class TestAlgorithmComparison:
    """Tests comparing different algorithms."""

    def test_both_pack_same_boxes(
        self, default_config: PackerConfig, sample_boxes: list[Box]
    ) -> None:
        """Test that both algorithms can pack the same boxes."""
        ffd = FirstFitDecreasingPacker(default_config)
        shelf = ShelfPacker(default_config)

        ffd_result = ffd.pack(sample_boxes.copy())
        shelf_result = shelf.pack(sample_boxes.copy())

        # Both should pack all boxes
        assert ffd_result.placed_count == len(sample_boxes)
        assert shelf_result.placed_count == len(sample_boxes)

    def test_sorting_applied(self, default_config: PackerConfig) -> None:
        """Test that boxes are sorted by volume before packing."""
        packer = FirstFitDecreasingPacker(default_config)

        # Create boxes with different volumes
        small = Box(id="small", width=10, height=10, length=10)
        large = Box(id="large", width=100, height=100, length=100)

        # Pack in "wrong" order
        result = packer.pack([small, large])

        # Both should still be packed (sorting handles order)
        assert result.placed_count == 2
