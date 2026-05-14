# Session 2026-04-24 — Invocations 3 and 4

## Scope

Single session on branch `001-public-release-hardening` that executed
two full spec-01 invocations back-to-back.

- **Invocation 3 of 14**: US1 Contract Integrity & Algorithm Honesty
  (T016–T030). 15 commits. T020 deferred to Invocation 6 (US3) — the
  test needs `docs/algorithms/` to exist.
- **Invocation 4 of 14**: US4 Repository Hygiene (T075–T084).
  12 commits. T078 (git-history scrub) and T079 (anonymisation)
  classified NOT REQUIRED after the T077 audit cleared all four
  `DATASETS/*.xlsx` as KEEP per maintainer confirmation.

Total: 27 commits, 75 tests passing, branch HEAD at `49962cb`.

## State at session end

- Branch: `001-public-release-hardening`
- Last commit: `49962cb chore(tasks): mark Phase 6 US4 tasks complete`
- Tag created: `legacy-code-preserved` (annotated, marks pre-`git mv CODE legacy`)
- Working tree: clean
- 75 tests pass, 1 deselected (slow marker)
- Coverage: 80 % overall (below 90 % floor — closes in US5/6/7)
- Mypy: 1 baseline error (`cli.py:177 _display_metrics_table`)

## Key architectural decisions this session

1. **Registry module `bin_packer_3d.algorithms.registry`** holds
   `ALGORITHMS`, `register`, `get_strategies` — concrete packers
   import `register` directly from this submodule to avoid partial-
   init cycles during package load.
2. **Kept `PackerBase` / `FirstFitDecreasingPacker` class names**
   (contracts/api.md says `BasePacker` / `FFDPacker`). Registry is
   keyed by string so class names are implementation detail.
   Contract docs drift flagged for US3 docs pass.
3. **`_resolve_mapping` helper** in `data/loaders.py` with precedence
   `explicit mapping > explicit config > ColumnMapping() defaults`.
   CLI passes `config=settings.data` to preserve pre-0.2 W/H/L/ITEM
   backcompat; new callers use `ColumnMapping()` directly.
4. **`can_fit_weight(weight: float | None)`** treats `None` as
   "unknown weight" and skips the capacity check (consistent with
   the T025 FR-005 semantics).
5. **`legacy/` excluded from linters** via
   `.pre-commit-config.yaml` top-level `exclude: '^legacy/'` and
   pyproject `[tool.ruff] extend-exclude = ["legacy", "DATASETS"]`
   so preserved Alpha-era code stays byte-faithful.

## Audit outcome (US4 T077)

All four `DATASETS/*.xlsx` are CNH (Case New Holland) packaging-
workflow reference data: product codes, box dimensions, product
weights. No PII, no customers, no pricing. Maintainer-cleared
public-safe on 2026-04-24. Recorded in
`DATASETS/AUDIT.md` with per-file schema + disposition.

`scripts/audit_datasets.py` found 6 additional deny-pattern
matches in git history (3 early-history repo-root xlsx paths +
3 generated outputs from legacy `CODE/MAIN.py`). Added to the
ALLOW list with `## Historical artefacts` section in AUDIT.md
explaining the provenance.

## Non-obvious learnings

1. **Pre-commit hooks run on staged rename targets.** `git mv CODE
   legacy` triggered trailing-whitespace / ruff / ruff-format hooks
   on the moved files, auto-modifying preserved content. Fix: top-
   level `exclude: '^legacy/'` in `.pre-commit-config.yaml` applies
   to every hook uniformly.
2. **A regex-scanning test whose docstring cites its pattern will
   flag its own source** once the test file becomes tracked. Keep
   the literal pattern in a compiled `_PATTERN` constant only;
   never spell example matches in prose.
3. **Pre-commit's ruff-format is stricter than manual
   `ruff format --check`** (different reflow decisions on strings).
   Workflow: `git add`, commit attempts, hook fixes, re-`git add`,
   retry commit.
4. **Pre-commit runs only ruff, NOT mypy.** Mypy errors can survive
   past commits — track via the §5.1 post-task gate, not hooks.
5. **Hatchling does NOT read `MANIFEST.in`.** T082's MANIFEST.in
   work is belt-and-suspenders for setuptools fallback. Both files
   were already authored in Setup phase — T082 was a no-op.
6. **`--cov-fail-under=90` surfaces a pre-existing baseline gap**
   at 80 %. Lowest-covered modules: `cli.py` 43 %, `shelf.py` 77 %,
   `loaders.py` 76 %, `base.py` 75 %, `plotter.py` 79 %,
   `bin.py` 79 %. Will close via US5/6/7 test additions.

## Deferred / parking-lot items

- T020 (docs-runtime consistency test) — needs `docs/algorithms/`;
  lands in Invocation 6 alongside T051.
- T049 (CI matrix expansion 3.11/3.12/3.13/3.14 + `--cov-fail-
  under=90`) — lands in Invocation 10 per DR-1 sequence.
- Contract docs drift (`BasePacker` vs `PackerBase`, `FFDPacker`
  vs `FirstFitDecreasingPacker`) — one-line fixes in
  `contracts/api.md` during Invocation 6 docs pass.
- `cli.py:177 _display_metrics_table` missing param annotation —
  baseline mypy error, sweep during US3 docstring phase.
- Hatch-build sdist hygiene check (`tar tzf dist/*.tar.gz | grep
  -E '^legacy/|^CODE/'` returning empty) — maintainer to run when
  convenient; project rule forbids builds from the agent.
- Optional future cleanup: `git filter-repo` scrub of the 3
  historical generated-output files (`asignacion_cajas_*.csv/xlsx`,
  `placements_result.csv`). Only needed if repository-history
  cleanliness matters more than force-push churn.

## Next recommended action

**Invocation 5 — US2 Continuous Integration Pipeline bootstrap.**
Tasks T031–T048 (T049 deferred). Authors `.github/workflows/ci.yml`
(dispatcher) + `_ci-core.yml` (reusable jobs), Dependabot,
CodeQL, PR templates, CODEOWNERS, codecov.yml, PyPI-style README
badges. Heavier on config authoring than code changes. Closes the
v0.2.0 P1 bundle (US1 + US4 + US2 + US3-seed).

## Files of note at session end

New:
- `src/bin_packer_3d/algorithms/registry.py`
- `tests/unit/test_registry.py`
- `tests/unit/test_data_loaders.py`
- `tests/integration/test_cli.py`
- `tests/integration/test_hygiene.py`
- `tests/integration/test_datasets.py`
- `DATASETS/AUDIT.md`
- `DATASETS/README.md`
- `legacy/README.md`
- `scripts/audit_datasets.py`

Modified:
- `src/bin_packer_3d/__init__.py` (re-exports)
- `src/bin_packer_3d/algorithms/__init__.py` (registry import order)
- `src/bin_packer_3d/algorithms/base.py` (ClassVar complexity + description)
- `src/bin_packer_3d/algorithms/ffd.py` (@register + ClassVars)
- `src/bin_packer_3d/algorithms/shelf.py` (@register + ClassVars)
- `src/bin_packer_3d/config.py` (strategy validator)
- `src/bin_packer_3d/models/box.py` (weight Optional)
- `src/bin_packer_3d/models/bin.py` (docstring + total_weight + can_fit_weight)
- `src/bin_packer_3d/cli.py` (info from registry, pack LoadReport)
- `src/bin_packer_3d/data/loaders.py` (ColumnMapping, LoadReport)
- `tests/unit/test_models.py` (weight tests)
- `pyproject.toml` ([tool.ruff] extend-exclude)
- `.pre-commit-config.yaml` (exclude legacy/)
- `CHANGELOG.md` (US1 + US4 bullets)
- `specs/001-public-release-hardening/tasks.md` (checkpoints)

Renamed via `git mv`:
- `CODE/*.py` → `legacy/*.py` (4 files, 100 % similarity)

## How to resume

1. `cd /home/brunoghiberto/Documents/Projects/3D_BIN_PACKING`
2. `git status` should be clean on `001-public-release-hardening`
3. `git log --oneline -30` for the 27-commit session history
4. `python -m pytest tests/ -m "not slow"` → 75 passed
5. `python scripts/audit_datasets.py` → "OK — 10 deny-pattern matches"
6. Next: present Invocation 5 design and await `go`
