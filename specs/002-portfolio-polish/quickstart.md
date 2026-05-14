# Quickstart: Portfolio Polish of `bin-packer-3d`

**Phase**: 1 (design) | **Spec ref**: User Story 1 (README first-screen credibility) + SC-001 + plan.md § Quickstart Contract | **Date**: 2026-05-13

This document defines the reviewer-facing 5-command flow that every release of spec-02 (v0.3.0-rc1) MUST support verbatim after `pip install`. It is the operational definition of "the repo works on a fresh machine in under 5 minutes" — Constitution Principle VI's 30-second README test, expanded to the running-the-code level.

## Prerequisites

- Python 3.11, 3.12, 3.13, or 3.14 (matches the project's `[project] requires-python` floor and CI matrix).
- `pip` available (or `uv`, `pipx`, etc. — any PEP-517 installer).
- A web browser for the visualisation step (any modern browser).

No git clone required for the basic flow; everything works from the published package.

## The Five Commands

### 1. Install (with `viz` extras for static export)

```bash
pip install 'bin-packer-3d[viz]'
```

The `[viz]` extra pulls in `kaleido` for Plotly static export (PNG / SVG). If you only want interactive HTML, `pip install bin-packer-3d` works without the extra.

**Expected output**: pip's standard install summary; concludes with `Successfully installed bin-packer-3d-0.3.0rc1 ...`.

### 2. Verify CLI + see registered algorithms

```bash
bin-packer info
```

**Expected output** (formatting via Rich):

```text
3D Bin Packer v0.3.0rc1

                       Available Algorithms
┏━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Strategy ┃ Complexity ┃ Description                             ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ bfd      │ O(n log n) │ Best Fit Decreasing — concentrate fills │
│ ffd      │ O(n log n) │ First Fit Decreasing (volume)           │
│ shelf    │ O(n log n) │ Shelf-based                             │
└──────────┴────────────┴─────────────────────────────────────────┘

Default Configuration:
  Bin dimensions: 860.0 x 890.0 x 1040.0 mm
  Allow rotation: True
```

The algorithm list comes from the `ALGORITHMS` registry. After US5 ships, two more algorithms (Extreme Point and Maximal Rectangles) will appear here automatically.

### 3. Pack the bundled headline dataset with BFD

```bash
bin-packer pack examples/headline.csv --strategy bfd --visualize -o /tmp/pack-bfd
```

If `examples/` is not in your current directory (you installed from PyPI rather than cloning), download it from the GitHub repository or substitute your own CSV path.

**Expected output** (truncated; full output includes a Rich-formatted metrics table):

```text
Loaded 50 boxes

Bin dimensions:
  Length: 860.0 mm
  Width:  890.0 mm
  Height: 1040.0 mm

Algorithm: Best-Fit Decreasing (BFD)

Results:
                  Packing Metrics
┃ Metric              ┃ Value                     ┃
│ Algorithm           │ Best-Fit Decreasing (BFD) │
│ Boxes Placed        │ 50/50                     │
│ Success Rate        │ 100.0%                    │
│ Bins Used           │ 1                         │
│ Overall Utilization │ 65.2%                     │
│ Avg Bin Utilization │ 65.2%                     │
│ Time                │ 0.28 ms                   │
└─────────────────────┴───────────────────────────┘
Saved 50 placements to: /tmp/pack-bfd/placements.csv

Generating visualizations...
Saved visualization: /tmp/pack-bfd/bin_1.html
Saved static export: /tmp/pack-bfd/bin_1.png
Created 1 visualization files
```

The 65.2% utilisation is the procedurally generated headline dataset's BFD score on the default bin (FR-034 acceptance: ≥ 60%).

### 4. Run the full demo — every algorithm on the same data

```bash
bin-packer demo -o /tmp/demo
```

**Expected output**:

```text
Running demo against: examples/headline.csv
Output directory: /tmp/demo

[bfd]   50/50 placed, 1 bin, 65.2% util, 0.28 ms
[ffd]   50/50 placed, 1 bin, 63.4% util, 0.31 ms
[shelf] 50/50 placed, 2 bins, 41.0% util, 0.22 ms

Comparison written to: /tmp/demo/comparison.md
Done in 1.4 seconds.
```

The total runtime is well under the 60-second budget (FR-021 / SC-002).

### 5. Open the generated visualisation in your browser

```bash
# Linux
xdg-open /tmp/demo/bfd/bin_1.html
# macOS
open /tmp/demo/bfd/bin_1.html
# Windows (Git Bash / WSL)
start /tmp/demo/bfd/bin_1.html
```

The HTML opens in your default browser and shows the packed bin with:

- Interactive 3D rotation, zoom, pan.
- Deterministic colours per box (same `box_id` → same colour across runs).
- Stats overlay panel: algorithm, boxes placed, utilisation, runtime.
- Branded axis labels in millimetres.
- Colourblind-safe palette (ColorBrewer Set3; verified per FR-037).

## Behavioural Guarantees

Per spec-02 acceptance criteria:

- Every command exits 0 on a clean install (no missing deps).
- The pack and demo commands complete in well under 60 seconds on a commodity laptop.
- Every command's `--help` includes a purpose statement, required flags, and at least one example.
- The visualisation HTML works offline once loaded (no CDN-only assets without an offline fallback).
- Static export (PNG) is reproducible byte-for-byte across runs (FR-006, Kaleido + Plotly pinned).

## Common Variations

### Use a different strategy

```bash
bin-packer pack examples/headline.csv --strategy ffd --visualize -o /tmp/pack-ffd
bin-packer pack examples/headline.csv --strategy shelf --visualize -o /tmp/pack-shelf
```

### Run the demo against your own dataset

```bash
bin-packer demo --dataset path/to/your.csv -o /tmp/your-demo
```

### Skip static export (faster, no `[viz]` extra required)

```bash
pip install bin-packer-3d              # no extras
bin-packer pack examples/headline.csv --strategy bfd --visualize -o /tmp/pack-bfd
# emits bin_1.html only; no PNG
```

### Regenerate the headline dataset

```bash
git clone https://github.com/Bruno-Ghiberto/3D_BIN_PACKING.git
cd 3D_BIN_PACKING
python scripts/generate_headline_dataset.py \
  --seed examples/headline.seed \
  --out examples/headline.csv
# Verify: re-run produces byte-identical CSV
```

## Troubleshooting

| Symptom | Resolution |
|---|---|
| `ImportError: Static export requires 'kaleido'.` | Install with the viz extra: `pip install 'bin-packer-3d[viz]'` |
| `bin-packer: command not found` | Ensure the install scripts directory is on PATH (`~/.local/bin` on user installs); or use `python -m bin_packer_3d`. |
| `examples/headline.csv: No such file or directory` | The `examples/` directory is repo-only (excluded from the wheel). Clone the repo or download the file from GitHub. |
| Visualisation HTML opens but is blank | Open the browser developer console; most issues are CSP-related on `file://` schemes. Try serving the file via `python -m http.server` and opening `http://localhost:8000/bin_1.html`. |

## Reference

Full documentation site: <https://bruno-ghiberto.github.io/3D_BIN_PACKING/>
