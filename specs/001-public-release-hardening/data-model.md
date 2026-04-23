# Phase 1 Data Model: Public-Release Hardening

**Feature**: `001-public-release-hardening`
**Date**: 2026-04-23
**Input**: [spec.md §Key Entities](./spec.md), [plan.md §Technical Context](./plan.md), [research.md](./research.md)

This document enumerates the entities in scope for spec-01, their fields,
invariants, validation rules, relationships, and state transitions.
Existing entities (`Box`, `Bin`, `Placement`, `PackerConfig`) are carried
forward from the current codebase and extended; new entities are introduced
cleanly. Phase annotations in the "Introduced" column indicate when the
entity is added or materially modified.

---

## Entity inventory

| Entity | File | Introduced | Kind |
|---|---|---|---|
| `Box` | `models/box.py` | existing — extended Phase A/B | pydantic model |
| `Bin` | `models/bin.py` | existing — extended Phase A | pydantic model |
| `Placement` | `models/placement.py` | existing | pydantic model |
| `PackingResult` | `models/result.py` | NEW Phase A | pydantic model |
| `LoadReport` | `models/result.py` | NEW Phase A | pydantic model |
| `PackerConfig` | `config.py` | existing — extended Phase A+B | pydantic settings |
| `ColumnMapping` | `data/loaders.py` | NEW Phase A | dataclass |
| `Constraint` | `constraints/base.py` | NEW Phase B | ABC |
| `AllowedOrientations` | `constraints/orientation.py` | NEW Phase B | pydantic model |
| `SupportedWeight` | `constraints/weight.py` | NEW Phase B | pydantic model |
| `AlgorithmMetadata` | `models/metadata.py` | NEW Phase B | dataclass (frozen) |
| `BenchmarkResult` | `benchmark/results.py` | NEW Phase B | dataclass |
| `BenchmarkInstance` | `benchmark/instances.py` | NEW Phase B | dataclass |
| `ConstraintResult` | `constraints/base.py` | NEW Phase B | dataclass (frozen) |

---

## Existing entities (extended)

### `Box`

A rectangular object to pack.

**Fields**:

| Name | Type | Constraints | Introduced |
|---|---|---|---|
| `length` | `float` | `> 0` | existing |
| `width` | `float` | `> 0` | existing |
| `height` | `float` | `> 0` | existing |
| `identifier` | `str \| None` | optional | existing |
| `description` | `str \| None` | optional | existing |
| `weight` | `float \| None` | `>= 0` when not None; **None = "unknown"** | Phase A (FR-005) |
| `allowed_orientations` | `frozenset[Orientation] \| None` | non-empty if not None | Phase B (FR-061) |
| `max_supported_weight` | `float \| None` | `>= 0` when not None | Phase B (FR-062) |

**Orientation** is a string enum: `{"ltw_wth", "ltw_wtl", "htw_ltw",
"htw_wtl", "wtl_htw", "wtl_ltw"}` — the six axis permutations of a
non-cube box. A cube yields one orientation only (existing behaviour
preserved as regression test, spec §Edge Cases).

**Invariants**:
- `length > 0 and width > 0 and height > 0` (validated by pydantic).
- `weight is None` is distinct from `weight == 0.0` — the former means
  "unknown", the latter means "known to be zero" (FR-005).
- `allowed_orientations is None` means "all six" (legacy `allow_rotation`
  flag reduces to a special case: `{"ltw_wth"}` when false, `None` when
  true). Phase B also accepts a `set` which pydantic coerces to
  `frozenset` for hashability.

**State transitions**: none. `Box` is an immutable value object after
construction.

**Backwards compatibility**: `weight` type changes from `float` (default
`0.0`) to `Optional[float]` (default `None`). This is a BREAKING change
per Constitution §Public API stability — documented in `CHANGELOG.md`
under `[0.2.0]` with a migration note (`Box(weight=0.0)` → keep as-is for
known-zero; `Box()` → now means unknown).

---

### `Bin`

A rectangular container.

**Fields**:

| Name | Type | Constraints | Introduced |
|---|---|---|---|
| `length` | `float` | `> 0` | existing |
| `width` | `float` | `> 0` | existing |
| `height` | `float` | `> 0` | existing |
| `identifier` | `str \| None` | optional | existing |
| `max_weight` | `float \| None` | `>= 0` when not None; **None = "unlimited"** | Phase A (edge case) |
| `placements` | `list[Placement]` | default empty | existing |

**Invariants**:
- `max_weight is None` means "unlimited capacity" (explicitly documented
  per spec §Edge Cases — distinguished from "zero capacity").
- `sum(p.box.weight for p in placements if p.box.weight is not None) <=
  max_weight` when `max_weight is not None` (enforced by packer, not by
  model).

**State transitions**: `placements` grows as the packer accepts boxes. No
state machine — it is a list append.

---

### `Placement`

A box positioned and oriented inside a bin.

**Fields** (all existing):

| Name | Type | Constraints |
|---|---|---|
| `box` | `Box` | required |
| `x` | `float` | `>= 0` |
| `y` | `float` | `>= 0` |
| `z` | `float` | `>= 0` |
| `orientation` | `Orientation` | required |
| `bin_identifier` | `str \| None` | parent bin's ID |

**Invariants** (enforced by placement-time validation):
- The placement lies within its bin's bounds: `x + effective_length <=
  bin.length`, and analogously for y/width and z/height. **Where
  `effective_length / effective_width / effective_height` are derived
  from `box` dimensions and the chosen `orientation`**.
- No two placements in the same bin overlap (interior volume
  intersection is empty). Enforced by the packer's `_accept` loop
  (Principle I).

These invariants are asserted in `tests/property/test_invariants.py` for
every registered algorithm (FR-046).

---

### `PackerConfig`

Configuration for a packing run.

**Fields**:

| Name | Type | Default | Introduced |
|---|---|---|---|
| `strategy` | `str` | `"ffd"` | existing (behaviour change — Phase A) |
| `bin_dimensions` | `tuple[float, float, float]` | required | existing |
| `allow_rotation` | `bool` | `True` | existing (deprecated — Phase B adds `allowed_orientations`) |
| `constraints` | `list[Constraint]` | `[]` | Phase B (FR-060) |
| `column_mapping` | `ColumnMapping \| None` | `None` (infer) | Phase A (FR-004) |
| `seed` | `int \| None` | `None` | Phase B (FR-043) |

**Validators**:
- `strategy` — `@field_validator("strategy")` that asserts the value is a
  key in `bin_packer_3d.algorithms.ALGORITHMS`. Error message lists all
  registered names (FR-047).
- `bin_dimensions` — every element `> 0`.

**Versioning** (per `contracts/config-schema.md`): every field has a
`Since:` annotation marking the release that introduced or materially
changed it. Any field rename or type change is a breaking change
requiring a major bump.

---

## New entities

### `PackingResult`

The outcome of running a packing algorithm.

**Fields**:

| Name | Type | Constraints |
|---|---|---|
| `placements` | `list[Placement]` | may be empty |
| `unpacked_boxes` | `list[Box]` | boxes that did not fit |
| `bins_used` | `int` | `>= 0` |
| `volume_utilisation` | `float` | `[0.0, 1.0]` — placed volume / total bin volume |
| `success_rate` | `float` | `[0.0, 100.0]` — percent of input boxes placed |
| `algorithm` | `str` | registry key of the algorithm that produced this |
| `runtime_seconds` | `float` | `>= 0` |
| `metadata` | `AlgorithmMetadata` | provenance |

**Invariants**:
- `len(placements) + len(unpacked_boxes) == len(input_boxes)` — box-count
  conservation (FR-046).
- Total placed volume + total unpacked-box volume == total input box
  volume — volume conservation (FR-046).
- Empty input yields `success_rate == 100.0` (spec §Edge Cases).

**Introduced**: Phase A (replaces the current loose `dict` return
surface).

---

### `LoadReport`

The outcome of loading boxes from CSV or Excel.

**Fields**:

| Name | Type | Constraints |
|---|---|---|
| `boxes` | `list[Box]` | parsed boxes |
| `rejected_rows` | `list[RejectedRow]` | structured rejections |
| `warnings` | `list[str]` | non-fatal messages |

Where `RejectedRow` is:

```python
@dataclass(frozen=True)
class RejectedRow:
    row_number: int       # 1-based, matches spreadsheet row
    reason: str           # machine-readable code (e.g. "missing_dimension")
    raw: dict[str, Any]   # original row for debugging
```

**Invariants**:
- Every malformed row appears in `rejected_rows` AND produces a WARNING
  log (FR-053). No silent `continue` (Principle I + V).

**Introduced**: Phase A.

---

### `ColumnMapping`

Declared mapping from CSV / Excel column names to `Box` fields.

**Fields**:

| Name | Type | Default |
|---|---|---|
| `length` | `str` | `"length"` |
| `width` | `str` | `"width"` |
| `height` | `str` | `"height"` |
| `identifier` | `str \| None` | `"identifier"` |
| `description` | `str \| None` | `"description"` |
| `weight` | `str \| None` | `"weight"` |

**Invariants**:
- Required fields (`length`, `width`, `height`) map to columns that MUST
  exist in the input file.
- Optional fields (`identifier`, `description`, `weight`) may map to
  non-existent columns — the loader silently omits them and sets the
  `Box` attribute to `None` (FR-004, spec §Edge Cases "Optional columns
  missing").

**Introduced**: Phase A (FR-004 repair).

---

### `Constraint` (ABC)

Base abstraction for constraint framework.

**Interface**:

```python
class Constraint(ABC):
    @abstractmethod
    def check(
        self,
        placement: Placement,
        box: Box,
        bin: Bin,
        existing_placements: Sequence[Placement],
    ) -> ConstraintResult:
        ...
```

**Introduced**: Phase B.

---

### `ConstraintResult`

Frozen dataclass returned by every `Constraint.check` call.

**Fields**:

| Name | Type | Default |
|---|---|---|
| `ok` | `bool` | required |
| `reason` | `str` | `""` (empty when `ok=True`) |

**Invariants**: `ok is True` ⇒ `reason == ""`; `ok is False` ⇒ `reason`
is non-empty.

**Introduced**: Phase B.

---

### `AllowedOrientations` (concrete `Constraint`)

Rejects placements whose orientation is not in the box's allowed set.

**Fields**: inherits `Constraint`. No additional state — reads
`box.allowed_orientations`.

**Behaviour**:
- If `box.allowed_orientations is None` → always pass (no restriction).
- Else → `placement.orientation in box.allowed_orientations` required.
- Rejection reason: `"orientation {orient} not in allowed set
  {allowed} for box {id}"`.

**Introduced**: Phase B (FR-061).

---

### `SupportedWeight` (concrete `Constraint`)

Rejects placements above a box whose cumulative overhead weight would
exceed the support limit.

**Fields**: inherits `Constraint`.

**Behaviour**:
- For each placement `p` in `existing_placements` directly below the
  candidate placement (z-axis supporting relation):
  - If `p.box.max_supported_weight is None` → no limit from this
    supporter (pass).
  - Else → compute cumulative weight above `p` if `box` is placed.
    If `cumulative > p.box.max_supported_weight` → reject.
- Rejection reason: `"placement would exceed max_supported_weight
  {limit} on supporting box {id}"`.

**Supporting relation**: `p` supports the candidate iff
`p.z + p.effective_height == candidate.z` AND the 2D (x, y) footprint
of the candidate intersects `p`'s footprint.

**Introduced**: Phase B (FR-062).

---

### `AlgorithmMetadata`

Provenance record captured at the start of each packing run.

**Fields** (frozen dataclass):

| Name | Type |
|---|---|
| `name` | `str` — registry key |
| `version` | `str` — `bin_packer_3d.__version__` |
| `parameters` | `dict[str, Any]` — captured from `PackerConfig` |
| `seed` | `int \| None` |
| `timestamp` | `datetime` — UTC, set at `packing.start` |

**Invariants**: immutable; serialises via `dataclasses.asdict` for
inclusion in `PackingResult` and `BenchmarkResult`.

**Introduced**: Phase B.

---

### `BenchmarkResult`

Per-(algorithm, instance) benchmark measurement.

**Fields** (dataclass):

| Name | Type | Constraints |
|---|---|---|
| `algorithm` | `str` | registry key |
| `instance` | `str` | e.g. `"BR1"` |
| `n_boxes` | `int` | `>= 0` |
| `n_bins_used` | `int` | `>= 0` |
| `volume_utilisation` | `float` | `[0.0, 1.0]` |
| `success_rate` | `float` | `[0.0, 100.0]` |
| `elapsed_seconds` | `float` | `>= 0` |
| `metadata` | `AlgorithmMetadata` | required |
| `notes` | `str` | default `""` |

**Serialisation**: JSON via `dataclasses.asdict`. Schema documented in
`contracts/benchmark-format.md`.

**Introduced**: Phase B.

---

### `BenchmarkInstance`

Describes a benchmark reference instance (e.g. BR1).

**Fields**:

| Name | Type |
|---|---|
| `name` | `str` — e.g. `"BR1"` |
| `boxes` | `list[Box]` |
| `bin_dimensions` | `tuple[float, float, float]` |
| `citation` | `str` — literature reference |
| `license` | `str \| None` — redistribution license |

**Invariants**:
- `license is None` ⇒ the instance is fetched by
  `benchmark/download_instances.py`, not bundled.
- `license is not None` ⇒ the instance is committed under
  `benchmark/instances/` alongside the license file.

**Introduced**: Phase B.

---

## Relationships

```text
              ┌────────────────┐
              │  PackerConfig  │
              └───┬────────────┘
                  │ 0..*
                  ▼
              ┌────────────────┐
              │   Constraint   │(ABC)
              └───▲────────────┘
                  │
         ┌────────┴────────┐
         │                 │
  ┌──────┴──────┐   ┌──────┴────────┐
  │ AllowedOri. │   │ SupportedWt.  │
  └─────────────┘   └───────────────┘

    ┌─────┐    1    ┌───────────┐
    │ Box │◄────────┤ Placement │
    └─────┘         └──┬────────┘
                       │ *
                       ▼ 1
                     ┌─────┐
                     │ Bin │
                     └─────┘

┌──────────────┐   ┌────────────────────┐
│ PackingResult├──►│ AlgorithmMetadata  │
├──────────────┤   └────────────────────┘
│  placements  │         ▲
│  unpacked    │         │
└──────┬───────┘   ┌─────┴──────────┐
       │  n..1     │ BenchmarkResult│
       ▼           └────────────────┘
  ┌──────────┐
  │ LoadReport│
  └──────────┘
```

---

## Versioning posture

All entities listed here are **public** (exported from
`bin_packer_3d.__init__`) unless marked "internal" in the "Kind" column of
the inventory. Every public entity's field list constitutes a versioned
contract (`contracts/api.md`). Field additions with defaults are MINOR;
field renames, type changes, or removal of a default are MAJOR.

All breaking changes introduced in Phase A (e.g. `Box.weight` becoming
`Optional[float]`) are documented in `CHANGELOG.md` under `[0.2.0]` with a
migration note. `v1.0.0` (Phase C) then stabilises the surface — post-1.0
breaking changes follow the stricter MAJOR amendment workflow.
