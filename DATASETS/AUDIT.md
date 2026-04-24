# DATASETS Privacy & Hygiene Audit

> **Audit date**: 2026-04-24
> **Auditor**: Bruno Ghiberto (maintainer) + spec-01 implementation agent
> **Scope**: FR-032, FR-033, Constitution Principle VII
> **Verdict**: **All four `.xlsx` files classified KEEP** — no PII, no
> customer-identifying data, no commercial secrets. Content is
> industrial reference data (product codes, box dimensions, product
> weights) from a CNH packaging-workflow analysis that has been
> maintainer-cleared for public release.

---

## `DATASETS/PACKING LIST.xlsx`

- **Sheets**: `P.L.` (1 sheet)
- **Shape**: 2 rows × 3 cols
- **Schema**: `CODIGO` (str) · `DESCRIPCION` (str) · `CANTIDAD` (int64)
- **Sample head**: 2 BOMBA APLICACION MF rows keyed to internal part codes.
- **PII/Privacy check**: No personal data, no customer names, no internal
  commercial terms. Product descriptions reference off-the-shelf parts.
- **Disposition**: **KEEP** — documented in `DATASETS/README.md`.
- **Intended use**: Quickstart demo of the Spanish-header CSV/Excel
  loader path (backwards-compat `_mapping_from_config` route).

## `DATASETS/PACKING LIST-11.xlsx`

- **Sheets**: `P.L.` (1 sheet)
- **Shape**: 82 rows × 3 cols
- **Schema**: `CODIGO` (str) · `DESCRIPCION` (str) · `CANTIDAD` (int64)
- **PII/Privacy check**: No personal data; descriptions are product-code
  strings (`ES.BO.16.L`, `EM.BO.14.C11` format). No customers, no
  pricing, no internal commentary.
- **Disposition**: **KEEP** — documented in `DATASETS/README.md`.
- **Intended use**: Mid-scale input sample (82 items) for integration
  tests and benchmark warm-up.

## `DATASETS/DIMENSIONES CAJAS-NORMALIZADO.xlsx`

- **Sheets**: `TAMAÑO-CAJAS` · `CAJA-PRODUCTO` · `TAMAÑO-CAJONES`
- **Shapes**:
  - `TAMAÑO-CAJAS`: 33 rows × 6 cols (box catalogue with volume + L/W/H in mm)
  - `CAJA-PRODUCTO`: 285 rows × 9 cols (product-to-box mapping with weight and 3 box-option columns)
  - `TAMAÑO-CAJONES`: 2 rows × 6 cols (outer-container dimensions)
- **Schema highlights**: dimensional columns (`LARGO_(mm)`, `ALTO_(mm)`,
  `ANCHO_(mm)`, `VOLUMEN_m3`), weight column (`PESO`), option columns
  (`CAJA_OP_1..3`, `CANTIDAD x CAJA_OP_1..3`).
- **PII/Privacy check**: Product codes and cardboard-box descriptions —
  no customers, no orders, no pricing. The "CAJON DE CARTON.CNH"
  strings are public-domain product codes (standard CNH packaging
  nomenclature).
- **Disposition**: **KEEP** — documented in `DATASETS/README.md`.
- **Intended use**: Reference box catalogue (`TAMAÑO-CAJAS`) feeds the
  benchmark's bin-size parameterisation; `CAJA-PRODUCTO` sourced the
  product-to-box lookup that the original research workflow used.

## `DATASETS/PESO_P.T.xlsx`

- **Sheets**: `PRODUCTO-PESO` (1 sheet)
- **Shape**: 4768 rows × 3 cols
- **Schema**: `CODIGO` (str) · `PESO(kg)` (float64) · `Columna1`
  (float64, mostly NaN — stray Excel column)
- **PII/Privacy check**: Product code + mass only — no customers, no
  suppliers, no pricing. Weights range from a few kg to ~100 kg.
- **Disposition**: **KEEP** — documented in `DATASETS/README.md`.
- **Intended use**: Large-scale (~5k rows) weight-lookup dataset useful
  for benchmark warm-up and for stress-testing the
  `weight is None` vs `weight == 0.0` disambiguation from T025.

---

## Summary

| File | Rows | Classification | Git-history scrub? |
|---|---:|---|---|
| `PACKING LIST.xlsx` | 2 | KEEP | No |
| `PACKING LIST-11.xlsx` | 82 | KEEP | No |
| `DIMENSIONES CAJAS-NORMALIZADO.xlsx` | 33 + 285 + 2 | KEEP | No |
| `PESO_P.T.xlsx` | 4768 | KEEP | No |
| `sample_boxes.csv` | (synthetic) | KEEP | No |

**No Stage-B actions required** — no `git filter-repo` scrub, no
anonymisation pass. The hatch build excludes
(`DATASETS/*.xlsx`) keep these out of wheel/sdist by default; they
ship only via the repository itself for reference and benchmarking.

---

## Historical artefacts (git history only, not working tree)

`scripts/audit_datasets.py` (T083) surfaced three additional
file-paths in git history that match deny patterns. All are
derivatives of the same CNH workflow covered above and inherit
the same maintainer clearance. They are present as allow-list
entries in the script with explanatory comments.

| Historical path | Kind | Provenance |
|---|---|---|
| `PACKING LIST.xlsx` (repo root) | Input | Earlier commit location before move to `DATASETS/` — same content |
| `DIMENSIONES CAJAS-NORMALIZADO.xlsx` (repo root) | Input | Same as above |
| `PESO_P.T.xlsx` (repo root) | Input | Same as above |
| `DATASETS/asignacion_cajas_final.csv` | Output | Computed by `CODE/MAIN.py` from the cleared inputs |
| `DATASETS/asignacion_cajas_final-_-.xlsx` | Output | Same workflow, xlsx export |
| `DATASETS/placements_result.csv` | Output | Packing-algorithm placements from the cleared inputs |

These files are NOT in the current working tree — they are
artefacts committed during early research and later removed. A
`git filter-repo --invert-paths` scrub remains an option if the
maintainer later decides repository-history cleanliness is worth
the force-push churn; until then, the release gate treats them
as allow-listed historical data.

**Follow-up**: `scripts/audit_datasets.py` (T083) is configured with
this file set as the allow-list. Any future `DATASETS/*.xlsx`
addition — current or historical — that does not appear in the
allow-list will fail the release-time gate until this `AUDIT.md`
and the script's ``ALLOW`` set are both updated.
