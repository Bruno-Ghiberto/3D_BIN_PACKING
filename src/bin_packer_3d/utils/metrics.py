"""Metrics calculation for packing results.

This module provides comprehensive metrics for evaluating
packing algorithm performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bin_packer_3d.models.placement import PlacementResult


@dataclass
class PackingMetrics:
    """Comprehensive metrics for a packing result.

    Attributes:
        total_boxes: Total boxes attempted.
        placed_boxes: Successfully placed boxes.
        unpacked_boxes: Boxes that couldn't be placed.
        bins_used: Number of bins used.
        total_box_volume: Volume of all placed boxes (mm³).
        total_bin_volume: Volume of all bins (mm³).
        utilization_percent: Overall space utilization (%).
        success_rate: Percentage of boxes placed (%).
        avg_bin_utilization: Average utilization per bin (%).
        elapsed_time_ms: Time taken (ms).
        algorithm: Algorithm name.
    """

    total_boxes: int
    placed_boxes: int
    unpacked_boxes: int
    bins_used: int
    total_box_volume: float
    total_bin_volume: float
    utilization_percent: float
    success_rate: float
    avg_bin_utilization: float
    elapsed_time_ms: float
    algorithm: str

    def __str__(self) -> str:
        """Render a multi-line human-readable metrics summary."""
        return (
            f"Packing Metrics ({self.algorithm})\n"
            f"{'=' * 40}\n"
            f"Boxes:        {self.placed_boxes}/{self.total_boxes} "
            f"({self.success_rate:.1f}% success)\n"
            f"Bins used:    {self.bins_used}\n"
            f"Utilization:  {self.utilization_percent:.1f}% overall\n"
            f"              {self.avg_bin_utilization:.1f}% avg per bin\n"
            f"Time:         {self.elapsed_time_ms:.2f}ms\n"
        )


def calculate_metrics(result: PlacementResult) -> PackingMetrics:
    """Calculate comprehensive metrics from packing result.

    Args:
        result: PlacementResult from a packing operation.

    Returns:
        PackingMetrics with all calculated values.
    """
    total_boxes = result.total_boxes
    placed_boxes = result.placed_count
    unpacked_boxes = len(result.unpacked_boxes)
    bins_used = result.bins_used

    total_box_volume = result.total_box_volume
    total_bin_volume = result.total_bin_volume

    utilization = result.utilization_percent
    success_rate = result.success_rate

    # Average utilization per bin
    avg_util = 0.0
    if result.bins:
        avg_util = sum(b.utilization_percent for b in result.bins) / len(result.bins)

    return PackingMetrics(
        total_boxes=total_boxes,
        placed_boxes=placed_boxes,
        unpacked_boxes=unpacked_boxes,
        bins_used=bins_used,
        total_box_volume=total_box_volume,
        total_bin_volume=total_bin_volume,
        utilization_percent=utilization,
        success_rate=success_rate,
        avg_bin_utilization=avg_util,
        elapsed_time_ms=result.elapsed_time_ms,
        algorithm=result.algorithm,
    )
