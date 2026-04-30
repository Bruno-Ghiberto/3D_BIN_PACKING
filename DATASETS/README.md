# DATASETS/

Reference datasets for `bin-packer-3d` examples, integration tests,
and reproducible benchmarks. Every file listed here has passed the
hygiene audit in [`AUDIT.md`](./AUDIT.md) — no PII, no customer-
identifying data, no commercial secrets.

By default the project's wheel and sdist exclude `*.xlsx` from this
directory (`pyproject.toml` → `[tool.hatch.build] exclude`), so
installing `bin-packer-3d` from PyPI will not ship these files.
They live in the repository so examples, notebooks, and
benchmarks can reference them directly.

---

## Inventory

### `sample_boxes.csv`

- **Origin**: Synthetic. Authored for the quickstart demo.
- **Schema** (CSV, 8 rows): `ITEM` (str) · `W` (int, width mm) ·
  `H` (int, height mm) · `L` (int, length mm) · `CANTIDAD` (int,
  quantity) · `CAJA` (str, box-type tag) · `DESCRIPCION` (str,
  free-text description).
- **Intended use**: Primary input for `quickstart.md` Scenarios
  A (CLI) and B (Python API). Loads cleanly via the legacy
  `DataConfig`-derived `ColumnMapping` mapping (`W`/`H`/`L`/
  `ITEM`/`CAJA`/`DESCRIPCION`/`CANTIDAD`). Ships as-is with the
  repository; the loader's new default `ColumnMapping` (lowercase
  `length`/`width`/`height`) would not load this file — pass
  `config=settings.data` or construct an explicit `ColumnMapping`
  that mirrors these column names (see `cli.py::pack` for a
  worked example).

### `PACKING LIST.xlsx`

- **Origin**: Industrial packing-list sample from a CNH
  (Case New Holland) logistics workflow analysis carried out by
  the maintainer. Public-safe; no PII (see `AUDIT.md`).
- **Schema**: sheet `P.L.` — 2 rows × 3 cols:
  `CODIGO` (product code, str) · `DESCRIPCION` (Spanish product
  description, str) · `CANTIDAD` (quantity, int).
- **Intended use**: Minimal smoke-test input for the Spanish-
  header Excel loader path.

### `PACKING LIST-11.xlsx`

- **Origin**: Mid-scale extract from the same CNH logistics
  analysis. Public-safe; no PII (see `AUDIT.md`).
- **Schema**: sheet `P.L.` — 82 rows × 3 cols (same schema as
  `PACKING LIST.xlsx`).
- **Intended use**: Mid-volume integration-test fixture and
  benchmark warm-up input.

### `DIMENSIONES CAJAS-NORMALIZADO.xlsx`

- **Origin**: Normalised box catalogue and product-to-box mapping
  from the CNH analysis; public-safe.
- **Schema**: three sheets —
  - `TAMAÑO-CAJAS` (33 rows × 6 cols): box catalogue with
    `VOLUMEN_m3`, `LARGO_(mm)`, `ALTO_(mm)`, `ANCHO_(mm)`.
  - `CAJA-PRODUCTO` (285 rows × 9 cols): product → box mapping
    with weight and up to three box options per product.
  - `TAMAÑO-CAJONES` (2 rows × 6 cols): outer cardboard container
    dimensions (the default bin used by the CLI:
    860 × 890 × 1040 mm is drawn from this table).
- **Intended use**: Reference bin catalogue for the benchmark
  runner (`benchmark/` package, US5) and source of the canonical
  default bin size.

### `PESO_P.T.xlsx`

- **Origin**: Product-weight lookup table from the same CNH
  analysis; public-safe.
- **Schema**: sheet `PRODUCTO-PESO` — 4768 rows × 3 cols:
  `CODIGO` (str) · `PESO(kg)` (float) · `Columna1` (stray Excel
  column, mostly NaN — do not rely on).
- **Intended use**: Large-scale product-weight dataset for
  stress-testing the loader + weight-aware packing paths, and
  fixture for property-based tests around the `Box.weight` None
  vs 0.0 disambiguation (FR-005).

---

## Adding a new dataset

1. Extend this README with a new `###` subsection covering origin,
   schema, and intended use.
2. Extend `AUDIT.md` with a privacy/hygiene entry (classification:
   KEEP / ANONYMISE / REMOVE).
3. Update the allow-list in `scripts/audit_datasets.py` so the
   release-time gate does not flag the new file.
4. `tests/integration/test_datasets.py` will auto-verify the
   README reference on the next CI run.

If the new file is the result of an external data drop that may
contain PII or commercial secrets, **do not commit it directly** —
anonymise first, or exclude via `.gitignore` and document the
exclusion rationale in `AUDIT.md`.
