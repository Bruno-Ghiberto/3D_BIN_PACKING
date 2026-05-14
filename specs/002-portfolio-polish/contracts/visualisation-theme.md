# Contract: Visualisation Theme

**Owner**: spec-02 (Portfolio Polish) | **Status**: STABLE at `v0.3.0-rc1` | **Source ADRs**: ADR-001, ADR-006, ADR-007, ADR-011

## Purpose

Specify the public surface of the branded Plotly theme system: which symbols are exported, what they guarantee, and what counts as a breaking change.

## Public Surface

All symbols below are re-exported from `bin_packer_3d` top-level.

```python
from bin_packer_3d import (
    VisualisationStyle,
    BIN_PACKER_3D_DARK,
    BIN_PACKER_3D_LIGHT,
    apply_theme,
)
```

### Types

- **`VisualisationStyle`** (frozen dataclass)
  - Fields: `name`, `palette`, `axis_label_format`, `hover_template`, `title_format`, `stats_overlay_layout` (see `data-model.md`).
  - Immutable; constructed only inside the library.

### Constants

- **`BIN_PACKER_3D_DARK: VisualisationStyle`** — dark-variant theme; default for `apply_theme`.
- **`BIN_PACKER_3D_LIGHT: VisualisationStyle`** — light-variant theme.

Both share the same colourblind-safe ColorBrewer Set3 palette (12 entries). Only background, axis, and overlay colours differ.

### Functions

- **`apply_theme(fig: go.Figure, variant: Literal["dark", "light"] = "dark") -> go.Figure`**
  - Mutates `fig.layout.template` to the named variant.
  - Returns the same `fig` object for fluent chaining.
  - Idempotent: calling twice produces the same result.

### Colour assignment

- **`colour_for_box(box_id: str, palette: tuple[str, ...] = SET3) -> str`** (re-exported from `bin_packer_3d.visualization.palette`)
  - Deterministic across runs and Python versions (BLAKE2b hash → modulo palette index).
  - Idempotent: same `box_id` always returns the same hex colour.
  - Returns a 7-character hex string in the form `#RRGGBB`.

## Behavioural Guarantees

1. **Determinism (FR-006, SC-006)**: For identical input (`box_id`, palette), the same colour string is returned across runs, across Python 3.11–3.14, across machines.
2. **Static export (FR-009)**: When the `viz` extra is installed (`pip install 'bin-packer-3d[viz]'`), every interactive HTML can be paired with a static export (PNG by default, SVG opt-in via `--static-format svg`).
3. **Plotly version coupling (ADR-001)**: This contract holds only against Plotly `>=5.18.0,<6.0.0`. A future Plotly 6.x upgrade is a separate spec and a major version bump.
4. **WCAG-AA contrast (FR-038, ADR-012)**: The text-on-background contrast within emitted figures meets WCAG-AA when rendered in either variant on the corresponding light/dark surface.
5. **Colourblind safety (FR-037, ADR-007 + ADR-012)**: The Set3 palette under deuteranopia and protanopia simulation has pairwise CIELAB ΔE ≥ 15. Verification artefact committed at `docs/assets/palette_colourblind_check.png`.

## Failure Modes

| Scenario | Behaviour | Exit / Error |
|---|---|---|
| `apply_theme(fig, "purple")` (unknown variant) | `ValueError: variant must be "dark" or "light"` | Raised at call time |
| `colour_for_box("")` (empty box ID) | Returns the colour at index `0` (modulo of zero digest is zero) | No error |
| Static export requested without `kaleido` installed | Plotter raises `ImportError` with the install instruction: `"Static export requires 'kaleido'. Install with: pip install 'bin-packer-3d[viz]'"` | Raised at export time |

## Test Coverage (CI gates)

- `tests/unit/test_visualization_theme.py` — palette determinism + `apply_theme` template-name assertion.
- `tests/unit/test_palette_colourblind.py` — `colorspacious` ΔE pass under deuteranopia + protanopia.
- `tests/integration/test_visualisation_e2e.py` — two consecutive runs on identical input produce byte-identical HTML; static-export PNG pixel-equal.

## Breaking-Change Policy

Any of the following is a breaking change requiring a major version bump:

- Changing the field types or names of `VisualisationStyle`.
- Removing `BIN_PACKER_3D_DARK` or `BIN_PACKER_3D_LIGHT`.
- Changing `apply_theme`'s signature.
- Changing the palette identity (entries, order, or count).
- Changing the BLAKE2b hash function or digest size used by `colour_for_box`.

Adding new palette variants, new style fields with sensible defaults, or new helper functions are non-breaking minor changes.
