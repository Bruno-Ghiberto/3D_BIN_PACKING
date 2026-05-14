# Contract: `bin-packer demo` Command

**Owner**: spec-02 (Portfolio Polish) | **Status**: STABLE at `v0.3.0-rc1` | **Source ADRs**: ADR-001, ADR-008

## Purpose

Specify the public CLI surface, default behaviour, output structure, and timing budget of the `bin-packer demo` subcommand.

## Invocation

```text
Usage: bin-packer demo [OPTIONS]

  Run every registered algorithm against a curated dataset and emit a
  per-algorithm visualisation gallery plus a comparison summary.

Options:
  --dataset PATH                  Input CSV dataset (default: examples/headline.csv)
  -o, --output-dir PATH           Output directory (default: examples/output)
  --static-format [png|svg]       Static export format (default: png; requires [viz] extra)
  --help                          Show this message and exit.
```

## Defaults

- **Dataset**: `examples/headline.csv` (the procedurally generated headline dataset; FR-034).
- **Output directory**: `examples/output/` (gitignored).
- **Static format**: `png` (requires `kaleido` from the `[viz]` extra; ADR-001).

## Behaviour

The command:

1. Loads the dataset via the existing `data.loaders` CSV reader.
2. Iterates the `ALGORITHMS` registry — every key gets a run.
3. For each algorithm:
   a. Packs the dataset using the algorithm's `Packer` class.
   b. Emits one interactive HTML per bin used (`bin_<N>.html`) via the visualisation layer with the dark-variant theme applied.
   c. Emits one static export per bin (`bin_<N>.<png|svg>`) via Kaleido (if available).
   d. Writes the `placements.csv` via the existing loader/exporter.
   e. Writes a `summary.json` (`DemoArtifact` serialisation; see `data-model.md`).
4. Writes a repo-root-relative `comparison.md` at the output directory listing every algorithm's metrics in a Markdown table.

## Output Structure

```text
examples/output/
├── bfd/
│   ├── bin_1.html
│   ├── bin_1.png
│   ├── placements.csv
│   └── summary.json
├── ffd/
│   ├── bin_1.html
│   ├── bin_1.png
│   ├── placements.csv
│   └── summary.json
├── shelf/
│   ├── bin_1.html
│   ├── bin_1.png
│   ├── placements.csv
│   └── summary.json
└── comparison.md
```

Number of `bin_*.html` / `bin_*.png` files per algorithm subdirectory equals the number of bins the algorithm needed.

## `summary.json` Schema

```json
{
  "source_dataset": "examples/headline.csv",
  "algorithm": "bfd",
  "html_paths": ["examples/output/bfd/bin_1.html"],
  "static_export_paths": ["examples/output/bfd/bin_1.png"],
  "placements_csv": "examples/output/bfd/placements.csv",
  "metrics": {
    "algorithm": "Best-Fit Decreasing (BFD)",
    "boxes_placed": 50,
    "boxes_total": 50,
    "bins_used": 1,
    "overall_utilisation": 0.652,
    "elapsed_ms": 0.28
  },
  "timestamp": "2026-05-13T18:42:11.123456Z"
}
```

## `comparison.md` Schema

A Markdown table at the output root summarising every algorithm:

```markdown
# Packing Demo — Comparison

Dataset: `examples/headline.csv` · Generated 2026-05-13T18:42:11Z

| Algorithm | Boxes placed | Bins used | Utilisation | Runtime |
|---|---|---|---|---|
| BFD   | 50/50 | 1 | 65.2% | 0.28 ms |
| FFD   | 50/50 | 1 | 63.4% | 0.31 ms |
| Shelf | 50/50 | 2 | 41.0% | 0.22 ms |
```

The table sort order is by `bins_used` ascending, then `overall_utilisation` descending (best-presenting algorithm first).

## Performance Budget (FR-021, SC-002)

- Total end-to-end runtime ≤ 60 seconds on commodity laptop.
- Verified by `tests/integration/test_demo_command.py` (pytest timeout).

## Failure Modes

| Scenario | Behaviour | Exit code |
|---|---|---|
| Input dataset path does not exist | Click raises `BadParameter`; non-zero exit | 2 (Click default for usage error) |
| `kaleido` not installed and `--static-format` requested | `ImportError` with install instruction; non-zero exit | 1 |
| Output dir exists and is not writable | `PermissionError` propagates | 1 |
| One algorithm fails mid-pack | The error is logged; other algorithms continue; final `comparison.md` notes the failure | 0 (partial success) |
| Total runtime exceeds 60 seconds | Test fails in CI; command itself returns successfully if it completes | — |

## Test Coverage

- `tests/integration/test_demo_command.py`:
  - Asserts exit 0 in ≤ 60 seconds.
  - Asserts output dir has one subdir per registered algorithm.
  - Asserts each subdir contains four expected files.
  - Asserts `comparison.md` exists and references every algorithm.

## Breaking-Change Policy

Changes to the following are breaking and require a major version bump:

- Removing `bin-packer demo` or renaming it.
- Changing the default output directory layout (subdir names, filenames).
- Changing the `summary.json` schema in a way that drops or renames fields.
- Changing the default dataset path.

Adding new options (e.g., `--algorithms bfd,ffd` to filter), additional output files, or new schema fields with default values are non-breaking minor changes.
