"""Data loading utilities for CSV and Excel files.

This module provides functions to load box data from various file formats
into :class:`~bin_packer_3d.models.box.Box` objects ready for packing.

The authoritative mapping from on-disk column names to
:class:`Box` fields is the :class:`ColumnMapping` dataclass (FR-004,
data-model.md §ColumnMapping). Legacy
:class:`~bin_packer_3d.config.DataConfig` column-name fields continue
to be honoured via :func:`_mapping_from_config` for backward
compatibility with pre-0.2 call sites (mainly the CLI).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from bin_packer_3d.config import DataConfig
from bin_packer_3d.models.box import Box
from bin_packer_3d.models.result import LoadReport, RejectedRow
from bin_packer_3d.observability import get_logger

if TYPE_CHECKING:
    from bin_packer_3d.models.placement import PlacementResult


_logger = get_logger("data.loaders")


@dataclass
class ColumnMapping:
    """Declared mapping from CSV / Excel column names to ``Box`` fields.

    Required fields (``length``, ``width``, ``height``) MUST map to
    columns that exist in the input file. Optional fields
    (``identifier``, ``description``, ``weight``) may map to
    non-existent columns — the loader silently omits them and leaves
    the corresponding :class:`Box` attribute at its default (``None``
    for weight, an auto-generated row id for identifier) per FR-004.

    Defaults follow ``data-model.md`` §ColumnMapping.
    """

    length: str = "length"
    width: str = "width"
    height: str = "height"
    identifier: str | None = "identifier"
    description: str | None = "description"
    weight: str | None = "weight"


def _mapping_from_config(config: DataConfig) -> ColumnMapping:
    """Derive a :class:`ColumnMapping` from the legacy :class:`DataConfig`.

    Preserves the pre-0.2 column names (``W``/``H``/``L``/``ITEM``,
    plus the hardcoded ``DESCRIPCION``/``PESO``) so callers that
    depended on those — most notably the CLI — keep working after the
    ColumnMapping introduction.
    """
    return ColumnMapping(
        length=config.length_column,
        width=config.width_column,
        height=config.height_column,
        identifier=config.item_column,
        description="DESCRIPCION",
        weight="PESO",
    )


def load_boxes_from_csv(
    file_path: Path | str,
    mapping: ColumnMapping | None = None,
    config: DataConfig | None = None,
) -> LoadReport:
    """Load boxes from a CSV file.

    Args:
        file_path: Path to CSV file.
        mapping: Column mapping. When ``None``, uses
            :class:`ColumnMapping` defaults — unless ``config`` is
            provided, in which case column names are derived from it
            for backward compatibility.
        config: Legacy :class:`DataConfig` source of column names.
            Retained for callers that pre-date ColumnMapping.

    Returns:
        :class:`LoadReport` carrying successfully parsed
        :class:`Box` objects, structured :class:`RejectedRow` records
        for every row that failed to parse, and human-readable
        ``warnings`` mirroring the rejection log (FR-053).

    Example:
        >>> report = load_boxes_from_csv("data.csv")
        >>> print(f"Loaded {len(report.boxes)} boxes; rejected {len(report.rejected_rows)}")
    """
    effective_mapping, quantity_column = _resolve_mapping(mapping, config)
    df = pd.read_csv(file_path)
    return _dataframe_to_report(df, effective_mapping, quantity_column=quantity_column)


def load_boxes_from_excel(
    file_path: Path | str,
    sheet_name: str | int = 0,
    mapping: ColumnMapping | None = None,
    config: DataConfig | None = None,
) -> LoadReport:
    """Load boxes from an Excel file.

    Args:
        file_path: Path to Excel file.
        sheet_name: Sheet name or index to load.
        mapping: Column mapping; see :func:`load_boxes_from_csv`.
        config: Legacy :class:`DataConfig`; see :func:`load_boxes_from_csv`.

    Returns:
        :class:`LoadReport`; see :func:`load_boxes_from_csv`.
    """
    effective_mapping, quantity_column = _resolve_mapping(mapping, config)
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return _dataframe_to_report(df, effective_mapping, quantity_column=quantity_column)


def _resolve_mapping(
    mapping: ColumnMapping | None,
    config: DataConfig | None,
) -> tuple[ColumnMapping, str]:
    """Resolve the effective mapping and quantity-column for a loader call.

    Precedence:
    1. Explicit ``mapping`` wins.
    2. Else, if ``config`` is explicitly provided, derive from it
       (pre-0.2 backcompat).
    3. Else, :class:`ColumnMapping` defaults — the new canonical API.

    Also returns the quantity column name sourced from ``config`` when
    supplied, otherwise the historical default ``"CANTIDAD"`` so
    quantity-expansion behaviour survives without coupling to
    ``DataConfig`` in the default path.
    """
    if mapping is not None:
        effective_mapping = mapping
    elif config is not None:
        effective_mapping = _mapping_from_config(config)
    else:
        effective_mapping = ColumnMapping()
    quantity_column = config.quantity_column if config is not None else "CANTIDAD"
    return effective_mapping, quantity_column


def _dataframe_to_report(
    df: pd.DataFrame,
    mapping: ColumnMapping,
    quantity_column: str = "CANTIDAD",
) -> LoadReport:
    """Convert a DataFrame to a :class:`LoadReport` using ``mapping``.

    Required columns (``mapping.length/width/height``) raise KeyError
    per row when missing — the row is recorded as a
    :class:`RejectedRow` and a ``logging.WARNING`` is emitted via the
    ``bin_packer_3d.data.loaders`` namespaced logger (FR-053,
    Constitution §I + §V). Optional columns are pre-checked with
    ``in row.index`` so their absence never triggers KeyError.
    """
    df.columns = df.columns.str.strip()

    boxes: list[Box] = []
    rejected_rows: list[RejectedRow] = []
    warnings: list[str] = []

    for idx, row in df.iterrows():
        row_number = int(idx) + 1 if isinstance(idx, int) else 0
        try:
            length = float(row[mapping.length])
            width = float(row[mapping.width])
            height = float(row[mapping.height])

            if (
                mapping.identifier
                and mapping.identifier in row.index
                and pd.notna(row[mapping.identifier])
            ):
                item_id = str(row[mapping.identifier])
            else:
                item_id = f"row_{row_number}"

            weight: float | None = None
            if mapping.weight and mapping.weight in row.index and pd.notna(row[mapping.weight]):
                weight = float(row[mapping.weight])

            description = ""
            if (
                mapping.description
                and mapping.description in row.index
                and pd.notna(row[mapping.description])
            ):
                description = str(row[mapping.description])

            box_type = ""
            if "CAJA" in row.index and pd.notna(row["CAJA"]):
                box_type = str(row["CAJA"])

            quantity = 1
            if quantity_column in row.index and pd.notna(row[quantity_column]):
                quantity = int(row[quantity_column])

            for i in range(quantity):
                box = Box(
                    id=f"{item_id}_{i + 1}" if quantity > 1 else item_id,
                    width=width,
                    height=height,
                    length=length,
                    weight=weight,
                    box_type=box_type,
                    description=description,
                    quantity=1,
                )
                boxes.append(box)

        except (KeyError, ValueError) as exc:
            reason = "missing_column" if isinstance(exc, KeyError) else "invalid_value"
            rejected_rows.append(
                RejectedRow(
                    row_number=row_number,
                    reason=reason,
                    raw={str(k): v for k, v in row.to_dict().items()},
                )
            )
            message = f"row {row_number} rejected ({reason}): {exc}"
            warnings.append(message)
            _logger.warning(message)
            continue

    return LoadReport(boxes=boxes, rejected_rows=rejected_rows, warnings=warnings)


def save_placements_to_csv(
    result: PlacementResult,
    output_path: Path | str,
) -> None:
    """Save packing result to CSV file.

    Args:
        result: PlacementResult to save.
        output_path: Path for output CSV.
    """
    records = [p.to_dict() for p in result.all_placements()]
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(records)} placements to: {output_path}")
