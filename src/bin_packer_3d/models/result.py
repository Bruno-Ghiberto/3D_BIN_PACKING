"""Result entities for packing runs and data loading.

Introduces three new entities per ``data-model.md`` §new entities:

- :class:`PackingResult` — pydantic model replacing the legacy
  :class:`~bin_packer_3d.models.placement.PlacementResult`. Carries
  explicit `placements`, `unpacked_boxes`, `bins_used`,
  `volume_utilisation`, `success_rate`, `algorithm`, `runtime_seconds`,
  and `metadata` fields with invariant-enforcing validators.
- :class:`LoadReport` — pydantic model returned by the data loaders
  carrying parsed boxes, rejected rows (structured), and human-readable
  warnings (FR-053 preparation).
- :class:`RejectedRow` — frozen dataclass describing a single rejected
  row from a CSV or XLSX input.

PackingResult intentionally coexists with the legacy
:class:`PlacementResult` during the Phase A transition. Algorithm
implementations will migrate to emit :class:`PackingResult` during
US1's contract-honesty work; :class:`PlacementResult` is scheduled for
removal once all call sites have migrated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from bin_packer_3d.models.box import Box
from bin_packer_3d.models.placement import Placement


@dataclass(frozen=True)
class RejectedRow:
    """A single rejected row from a data loader.

    Frozen dataclass — immutable value object. The structured `reason`
    code enables downstream consumers to filter or aggregate
    rejections programmatically without parsing log messages.

    Attributes:
        row_number: 1-based row index matching spreadsheet convention.
        reason: Machine-readable code, e.g. ``"missing_dimension"``,
            ``"schema_error"``, ``"invalid_weight"``.
        raw: Original row payload as returned by the reader (kept for
            debugging; opaque to the loader).
    """

    row_number: int
    reason: str
    raw: dict[str, Any]


class PackingResult(BaseModel):
    """Outcome of a packing-algorithm run.

    Fields mirror ``data-model.md`` §PackingResult exactly. Pydantic
    validators enforce the numeric invariants documented there:

    - ``bins_used >= 0``
    - ``volume_utilisation`` in ``[0.0, 1.0]``
    - ``success_rate`` in ``[0.0, 100.0]``
    - ``runtime_seconds >= 0``

    Box-count conservation
    (``len(placements) + len(unpacked_boxes) == len(input_boxes)``) and
    volume conservation are algorithm-level invariants verified by the
    property-based test suite under ``tests/property/`` — not enforced
    at construction time because PackingResult does not retain a
    reference to the original input.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    placements: list[Placement] = Field(default_factory=list)
    unpacked_boxes: list[Box] = Field(default_factory=list)
    bins_used: int = Field(default=0, ge=0)
    volume_utilisation: float = Field(default=0.0, ge=0.0, le=1.0)
    success_rate: float = Field(default=100.0, ge=0.0, le=100.0)
    algorithm: str = ""
    runtime_seconds: float = Field(default=0.0, ge=0.0)
    # TODO(T096): tighten `Any` to `AlgorithmMetadata | None` once the
    # metadata model ships in Phase B.
    metadata: Any = None


class LoadReport(BaseModel):
    """Structured outcome of loading boxes from CSV or XLSX.

    Every malformed row that the loader skips MUST appear in
    :attr:`rejected_rows` AND in :attr:`warnings` (via
    :func:`bin_packer_3d.observability.get_logger`). Silent ``continue``
    is forbidden per Constitution §I (Contract Honesty) and FR-053.

    Attributes:
        boxes: Successfully parsed boxes.
        rejected_rows: Structured per-row rejection records.
        warnings: Non-fatal human-readable messages surfaced alongside.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    boxes: list[Box] = Field(default_factory=list)
    rejected_rows: list[RejectedRow] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# Resolve forward references now that Box and Placement are imported. Both
# use `from __future__ import annotations`, so their class references become
# strings at class-definition time. model_rebuild() resolves them in this
# module's namespace so pydantic can validate at construction.
PackingResult.model_rebuild()
LoadReport.model_rebuild()


__all__ = ["LoadReport", "PackingResult", "RejectedRow"]
