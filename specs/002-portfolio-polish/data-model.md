# Data Model: Portfolio Polish of `bin-packer-3d`

**Phase**: 1 (design) | **Feeds**: `plan.md` § Project Structure, `contracts/*.md` | **Date**: 2026-05-13

This document specifies the data model additions and augmentations introduced by spec-02. Existing entities from spec-01 (`Box`, `Bin`, `Placement`, `PackerConfig`, `PackingResult`, `LoadReport`, `RejectedRow`, `ColumnMapping`, `AlgorithmMetadata`, `BenchmarkResult`, `ConstraintVisitor` and concretions, `StructuredAdapter`) are carried forward unchanged.

---

## Carried-Forward Entities (no spec-02 changes)

The following entities are unmodified by spec-02. Refer to `specs/001-public-release-hardening/data-model.md` for their full specification.

- `Box` — geometric box dataclass (length / width / height / weight / identifier / description)
- `Bin` — packed bin (dimensions, placements list, weight constraint)
- `Placement` — single box-in-bin placement record (will be augmented in spec-02; see §Augmented below)
- `PackerConfig` — Pydantic v2 settings (strategy, bin dimensions, allow_rotation, seed, constraints)
- `PackingResult` — outcome record (algorithm, boxes placed, bins used, utilisation, elapsed time)
- `LoadReport` — loader outcome (loaded rows, rejected rows, warnings)
- `RejectedRow` — single row rejection record from loader
- `ColumnMapping` — CSV / Excel column mapping configuration
- `AlgorithmMetadata` — `@register`-decorator metadata for an algorithm (name, complexity, description)
- `BenchmarkResult` — benchmark run record (algorithm, n_boxes, utilisation, runtime, metadata) — US5 territory; spec-02 does NOT modify
- `ConstraintVisitor` (ABC) + `WeightConstraint`, `RotationLockConstraint` — constraint framework
- `StructuredAdapter` — `logging.LoggerAdapter` subclass for structured-event emission

---

## New Entities

### `VisualisationStyle`

**Module**: `src/bin_packer_3d/visualization/theme.py`
**Kind**: Frozen dataclass.
**Public**: Yes — re-exported from `bin_packer_3d` top-level.
**Introduced by**: spec-02 (US2 — Visualisation polish).

```python
from dataclasses import dataclass
from typing import Any, Literal

@dataclass(frozen=True)
class VisualisationStyle:
    """Branded Plotly theme configuration applied to every emitted figure.

    Attributes:
        name: Identifier; one of "bin_packer_3d_dark" or "bin_packer_3d_light".
        palette: 12 hex-string colours from ColorBrewer Set3, deterministic,
            colourblind-safe per FR-037 (verified empirically by ADR-012).
        axis_label_format: f-string template for axis titles, e.g. "{name} (mm)".
        hover_template: Plotly hover string with %{customdata.*} fields.
        title_format: f-string template for figure title, e.g. "{algorithm} — {dataset}".
        stats_overlay_layout: Plotly annotation block defining the stats panel layout
            (algorithm, boxes placed, utilisation %, runtime ms).
    """
    name: Literal["bin_packer_3d_dark", "bin_packer_3d_light"]
    palette: tuple[str, ...]
    axis_label_format: str
    hover_template: str
    title_format: str
    stats_overlay_layout: dict[str, Any]
```

**Constants exported alongside**:

- `BIN_PACKER_3D_DARK: VisualisationStyle` — dark-variant theme constant.
- `BIN_PACKER_3D_LIGHT: VisualisationStyle` — light-variant theme constant.

**Public function exported alongside**:

```python
def apply_theme(
    fig: plotly.graph_objects.Figure,
    variant: Literal["dark", "light"] = "dark",
) -> plotly.graph_objects.Figure:
    """Apply the bin-packer-3d branded theme to a Plotly figure.

    Returns the same figure object (mutated in place) for fluent chaining.
    """
```

**Validation rules**:

- `palette` MUST contain exactly 12 entries (ColorBrewer Set3 constant); each MUST be a 7-char hex string (`#RRGGBB`).
- `name` MUST match a registered variant.
- `hover_template` MUST be a non-empty string containing at least one `%{...}` placeholder.

**Lifecycle**: Immutable. Constants are module-level singletons; never reassigned at runtime.

**Contract reference**: `contracts/visualisation-theme.md`.

---

### `DemoArtifact`

**Module**: `src/bin_packer_3d/visualization/demo.py` (new) OR inline in `cli.py`'s `demo` subcommand handler.
**Kind**: Frozen dataclass.
**Public**: Internal only — not re-exported. Materialised on disk as `summary.json` per algorithm; the in-memory dataclass is a serialisation helper.
**Introduced by**: spec-02 (US5 — Demo command).

```python
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

@dataclass(frozen=True)
class DemoArtifact:
    """Per-algorithm output record from a single `bin-packer demo` invocation.

    Attributes:
        source_dataset: Path to the input CSV (defaults to examples/headline.csv).
        algorithm: Strategy key from the ALGORITHMS registry, e.g. "bfd".
        html_paths: Per-bin interactive Plotly HTML files emitted.
        static_export_paths: Per-bin PNG (default) or SVG files emitted.
        placements_csv: Path to the per-placement CSV output.
        summary_json: Path to the JSON manifest written by the demo handler.
        metrics: Reuses the existing PackingResult dataclass for run metrics.
        timestamp: UTC timestamp at run start.
    """
    source_dataset: Path
    algorithm: str
    html_paths: tuple[Path, ...]
    static_export_paths: tuple[Path, ...]
    placements_csv: Path
    summary_json: Path
    metrics: "PackingResult"  # forward reference
    timestamp: datetime
```

**Validation rules**:

- `algorithm` MUST be a key in `ALGORITHMS` registry (validated at construction time).
- `html_paths` length MUST equal `static_export_paths` length (one of each per bin used).
- `summary_json` file MUST be writable; failure raises `IOError` (no silent skip — Principle V).

**Lifecycle**: Created once per algorithm per `bin-packer demo` invocation. Serialised to `summary.json` immediately after computation; in-memory copy discarded after serialisation.

**Contract reference**: `contracts/demo-command.md`.

---

### `HeadlineDatasetSeed`

**Module**: not a Python module — JSON file on disk at `examples/headline.seed`.
**Kind**: JSON document. Has no Python dataclass counterpart; the generator script reads and validates the JSON directly.
**Public**: Yes — committed to the repository as the deterministic input to the headline dataset generator.
**Introduced by**: spec-02 (FR-034 — Procedural headline dataset).

**Schema** (JSON):

```json
{
  "seed": 42,
  "target_utilisation": 0.65,
  "bin_dimensions": [860.0, 890.0, 1040.0],
  "n_boxes": 50,
  "min_box_volume_mm3": 50000.0
}
```

**Field meanings**:

- `seed` (int): RNG seed for the generator. Stable across runs.
- `target_utilisation` (float in (0, 1)): The volume utilisation the generator aims for in the constructed virtual bin. Set to 0.65 to comfortably exceed the FR-034 ≥60% acceptance floor.
- `bin_dimensions` (3-tuple of float, length / width / height in mm): The bin against which the generator constructs its "feasible packing". Default `[860.0, 890.0, 1040.0]` matches the project's default bin dims.
- `n_boxes` (int): Target number of boxes to emit. Generator continues recursive cuts until this count is reached OR each sub-region falls below `min_box_volume_mm3`.
- `min_box_volume_mm3` (float): Floor on per-box volume (terminates recursion).

**Validation rules** (enforced by `scripts/generate_headline_dataset.py` at read time):

- `seed` is a non-negative integer.
- `target_utilisation` is strictly between 0 and 1.
- `bin_dimensions` is a list of exactly 3 positive floats.
- `n_boxes` is a positive integer.
- `min_box_volume_mm3` is a positive float strictly less than the product of bin_dimensions.

**Lifecycle**: Authored once during Phase A; committed to the repository. Regenerating `examples/headline.csv` from the same seed file MUST produce byte-identical CSV output (Principle IV).

**Contract reference**: `contracts/headline-dataset.md`.

---

## Augmented Existing Entities

### `Placement` — adds derived `colour` attribute

**Module**: `src/bin_packer_3d/models/placement.py` (existing).
**Augmentation**: spec-02 adds a derived (computed, not stored) `colour` attribute.

```python
from dataclasses import dataclass
from functools import cached_property

@dataclass(frozen=True)
class Placement:
    # ... existing fields ...

    @cached_property
    def colour(self) -> str:
        """Deterministic colour assigned to this placement's box.

        Computed via palette.colour_for_box(self.box.identifier).
        Not persisted in placements.csv; consumed by the visualisation layer
        when rendering Plotly figures.
        """
        from bin_packer_3d.visualization.palette import colour_for_box
        return colour_for_box(self.box.identifier)
```

**Notes**:

- `cached_property` keeps the lookup `O(1)` after first call, preserves frozen-dataclass semantics (write-once via descriptor).
- `colour` is NOT serialised to `placements.csv` — that CSV's contract is owned by the existing loader/exporter and remains unchanged. Visualisation reads `colour` from in-memory `Placement` instances.
- The import is local (inside the property) to avoid circular dependencies at module load time.

**Backwards-compatibility**: All existing code that constructs or consumes `Placement` continues to work unchanged. The new attribute is opt-in via attribute access.

---

## Entity Relationship Diagram (additions only)

```text
                   ┌──────────────────────────┐
                   │  HeadlineDatasetSeed     │
                   │  (JSON on disk)          │
                   └────────────┬─────────────┘
                                │ read by
                                ▼
                   ┌──────────────────────────┐
                   │  generate_headline_      │
                   │  dataset.py (script)     │
                   └────────────┬─────────────┘
                                │ emits
                                ▼
                   ┌──────────────────────────┐
                   │  examples/headline.csv   │
                   │  (input to demo + pack)  │
                   └────────────┬─────────────┘
                                │ consumed by
                                ▼
       ┌────────────────────────────────────────┐
       │  bin-packer demo (CLI handler)         │
       │  iterates ALGORITHMS registry          │
       └──┬───────────────┬─────────────────────┘
          │ per algo      │
          ▼               ▼
  ┌───────────────┐  ┌──────────────────────┐
  │  PackingResult│  │  Plotly Figure       │
  │  (existing)   │  │  + apply_theme()     │
  └───────┬───────┘  │  + Kaleido export    │
          │          └──────────┬───────────┘
          └─────────┬───────────┘
                    │
                    ▼
       ┌──────────────────────────┐
       │  DemoArtifact            │
       │  (serialised to          │
       │   summary.json)          │
       └──────────────────────────┘

   ┌─────────────────────────────────────┐
   │  Placement (existing, augmented)    │
   │  + .colour (derived via             │
   │    palette.colour_for_box)          │
   └──────────────────────────┬──────────┘
                              │ read by
                              ▼
                   ┌──────────────────────────┐
                   │  VisualisationStyle      │
                   │  + apply_theme()         │
                   └──────────────────────────┘
```

---

## Public API Surface Additions (`__init__.py`)

The following symbols are added to `src/bin_packer_3d/__init__.py` `__all__`:

```python
__all__ = [
    # ... existing exports ...
    "VisualisationStyle",
    "BIN_PACKER_3D_DARK",
    "BIN_PACKER_3D_LIGHT",
    "apply_theme",
    "colour_for_box",
]
```

`DemoArtifact` and `HeadlineDatasetSeed` are NOT re-exported — they are internal data models. The `bin-packer demo` CLI subcommand is registered via the existing `bin-packer` entry-point and is discoverable via `bin-packer --help`.

---

**End of data-model.**
