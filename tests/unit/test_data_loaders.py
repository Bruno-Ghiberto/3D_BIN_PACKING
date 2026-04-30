"""Unit tests for CSV/Excel loaders — column mapping and LoadReport (T018).

Exercises FR-004 (honour declared column mapping, tolerate missing
optional columns) and FR-053 (malformed rows surfaced via LoadReport
rather than silently dropped). See also spec §Edge Cases "Optional
columns missing" and data-model.md §ColumnMapping.
"""

from __future__ import annotations

from pathlib import Path

from bin_packer_3d.data.loaders import ColumnMapping, load_boxes_from_csv


class TestLoaderColumnMapping:
    """FR-004: declared mapping authoritative; missing optional columns tolerated."""

    def test_missing_identifier_column_no_keyerror(self, tmp_path: Path) -> None:
        """CSV with only required columns (length/width/height) loads without KeyError."""
        csv = tmp_path / "minimal.csv"
        csv.write_text("length,width,height\n100,50,80\n200,100,150\n")

        report = load_boxes_from_csv(csv)

        assert len(report.boxes) == 2
        assert report.rejected_rows == []

    def test_missing_weight_column_yields_none(self, tmp_path: Path) -> None:
        """CSV without a weight column yields Box.weight is None (FR-005)."""
        csv = tmp_path / "noweight.csv"
        csv.write_text("length,width,height\n100,50,80\n")

        report = load_boxes_from_csv(csv)

        assert len(report.boxes) == 1
        assert report.boxes[0].weight is None

    def test_custom_mapping_renames_columns(self, tmp_path: Path) -> None:
        """A custom ColumnMapping renames on-disk columns to Box fields."""
        csv = tmp_path / "custom.csv"
        csv.write_text("L,W,H,ID,Mass\n100,50,80,A-1,2.5\n")

        mapping = ColumnMapping(
            length="L",
            width="W",
            height="H",
            identifier="ID",
            weight="Mass",
        )
        report = load_boxes_from_csv(csv, mapping=mapping)

        assert len(report.boxes) == 1
        box = report.boxes[0]
        assert box.length == 100.0
        assert box.width == 50.0
        assert box.height == 80.0
        assert box.id == "A-1"
        assert box.weight == 2.5
