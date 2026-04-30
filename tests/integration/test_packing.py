"""Integration tests for complete packing workflow."""

import pytest

from bin_packer_3d.algorithms.ffd import FirstFitDecreasingPacker
from bin_packer_3d.config import PackerConfig
from bin_packer_3d.models.box import Box
from bin_packer_3d.utils.metrics import calculate_metrics
from bin_packer_3d.visualization.plotter import Plotter3D


class TestPackingWorkflow:
    """Integration tests for complete packing workflow."""

    def test_complete_workflow(self, default_config: PackerConfig, sample_boxes: list[Box]) -> None:
        """Test complete workflow: pack -> metrics -> visualize."""
        # Pack
        packer = FirstFitDecreasingPacker(default_config)
        result = packer.pack(sample_boxes)

        # Calculate metrics
        metrics = calculate_metrics(result)

        assert metrics.placed_boxes == len(sample_boxes)
        assert metrics.success_rate == 100.0
        assert metrics.utilization_percent > 0

        # Visualization (don't actually create files in test)
        plotter = Plotter3D()
        for bin_obj in result.bins:
            fig = plotter.plot_bin(bin_obj, title="Test")
            assert fig is not None

    def test_no_overlapping_placements(
        self, default_config: PackerConfig, sample_boxes: list[Box]
    ) -> None:
        """Test that placements don't overlap."""
        packer = FirstFitDecreasingPacker(default_config)
        result = packer.pack(sample_boxes)

        for bin_obj in result.bins:
            placements = bin_obj.placements
            for i, p1 in enumerate(placements):
                for p2 in placements[i + 1 :]:
                    assert not p1.overlaps_with(p2), f"Overlap detected: {p1} and {p2}"

    def test_placements_within_bin_bounds(
        self, default_config: PackerConfig, sample_boxes: list[Box]
    ) -> None:
        """Test that all placements are within bin boundaries."""
        packer = FirstFitDecreasingPacker(default_config)
        result = packer.pack(sample_boxes)

        for bin_obj in result.bins:
            for placement in bin_obj.placements:
                assert placement.x0 >= 0
                assert placement.y0 >= 0
                assert placement.z0 >= 0
                assert placement.x1 <= bin_obj.length
                assert placement.y1 <= bin_obj.width
                assert placement.z1 <= bin_obj.height

    @pytest.mark.slow
    def test_large_dataset(self, default_config: PackerConfig) -> None:
        """Test with a larger dataset."""
        # Generate 100 random boxes
        boxes = [
            Box(
                id=f"box_{i}",
                width=50 + (i % 10) * 10,
                height=30 + (i % 8) * 10,
                length=40 + (i % 6) * 10,
            )
            for i in range(100)
        ]

        packer = FirstFitDecreasingPacker(default_config)
        result = packer.pack(boxes)

        assert result.placed_count == 100
        assert result.success_rate == 100.0
        assert result.elapsed_time_ms < 5000  # Should complete in < 5 seconds
