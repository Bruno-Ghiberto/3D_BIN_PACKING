# Examples

Curated datasets demonstrating `bin-packer-3d` across difficulty profiles.

This directory will hold:

- `headline.csv` + `headline.seed` — the procedurally generated headline
  dataset consumed by the `bin-packer demo` subcommand (lands in T013/T014
  of Phase A; spec-02 Foundational).
- `small.csv` — a friendly intro dataset (8-15 boxes, single bin, clear
  visualisation; hand-curated, lands in US5 / T070).
- `stress.csv` — an edge-case dataset (oversized / odd-shaped boxes,
  expected partial rejection; hand-curated, lands in US5 / T071).

Generated outputs from `bin-packer demo` default to `examples/output/`,
which is gitignored.

This README is a placeholder created by T007. Final content (full per-dataset
documentation table — dimensions, intent, expected qualitative result) lands
in US5 / T072.
