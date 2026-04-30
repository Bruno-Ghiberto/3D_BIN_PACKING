# Changelog

All notable changes to `bin-packer-3d` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
as interpreted by the project constitution (see
`.specify/memory/constitution.md` §Release workflow).

## [Unreleased]

### Added

**Setup (T001..T009):**

- Python 3.13 and 3.14 to the supported-version matrix.
- `docs` optional-dependency extra: `mkdocs-material`, `mkdocstrings[python]`.
- `hypothesis` and `pip-audit` added to the `dev` optional-dependency extra.
- `py.typed` marker file (PEP 561) — downstream consumers now get type
  information when importing `bin_packer_3d` (FR-035).
- `.editorconfig` at the repository root — codifies whitespace behaviour
  across editors (FR-036).
- `.pre-commit-config.yaml` — mirrors CI lint/format stages so contributors
  fail fast locally (FR-013).
- Pydocstyle (`ruff` `D` rules, Google convention) enabled on
  `src/bin_packer_3d/` — missing docstrings on public symbols will be
  surfaced during Phase A docstring sweep (FR-026 preparation).
- Hatch build exclusion rules for `legacy/`, `CODE/`, `benchmark/`, `docs/`,
  `tests/`, `DATASETS/*.xlsx`, planning artefacts, and repository-internal
  directories — sdist and wheel now contain only library code and top-level
  metadata files.

**Foundational (T010..T015):**

- New module `bin_packer_3d.observability` with `get_logger(name)` (namespaced
  under `bin_packer_3d` with an idempotent `NullHandler`) and
  `StructuredAdapter` (lifts `extra['fields']` into LogRecord attributes)
  per Constitution §V and ADR-0008 (FR-050).
- New module `bin_packer_3d.models.result` with pydantic `PackingResult`,
  pydantic `LoadReport`, and frozen-dataclass `RejectedRow` per
  `data-model.md` §new entities. Coexists with the legacy `PlacementResult`
  during the Phase A transition.
- `PackerBase._check_constraints(placement, box, bin, existing_placements)`
  hook consulted at every first-time placement-accept seam in `FFDPacker`
  and `ShelfPacker`. Empty-list no-op until US7 T131 adds the
  `constraints` field on `PackerConfig` — forward-compatible via
  `getattr` fallback (FR-063).
- Public re-exports on `bin_packer_3d.__init__`: `get_logger`,
  `StructuredAdapter`, `PackingResult`, `LoadReport`, `RejectedRow`.
- Regression guard `tests/unit/test_library_hygiene.py` — asserts
  `import bin_packer_3d` attaches a `NullHandler` to the package logger
  and does NOT touch the root logger, `logging.basicConfig`, `sys.path`,
  or `cwd` (Constitution §V enforcement, FR-050).
- Seeded `tests/unit/test_observability.py` — `get_logger` + `StructuredAdapter`
  unit tests. Extended in US6 T110..T113.

**US4 Repository Hygiene (T075..T084):**

- `tests/integration/test_hygiene.py` — forward-guard scanning every
  tracked file under `src/`, `tests/`, `benchmark/` for Windows
  drive-letter roots (``C:\``…) and POSIX home directories
  (`/home/<user>/`) (FR-030, T075). Same file also guards author
  alignment between `bin_packer_3d.__author__` and
  `pyproject.toml` authors (FR-034, T084).
- `tests/integration/test_datasets.py` — asserts `DATASETS/README.md`
  exists and references every tracked `.csv`/`.xlsx` file by name
  (FR-032, T076).
- `DATASETS/AUDIT.md` — per-file privacy + hygiene audit of the
  four CNH business `.xlsx` files plus historical artefacts
  surfaced from git history. All cleared KEEP; no scrub required
  (FR-033, Constitution §VII, T077).
- `DATASETS/README.md` — documents `sample_boxes.csv` and every
  retained `.xlsx` file (origin + schema + intended use) per
  FR-032 (T080).
- `legacy/README.md` — explains the preserved Alpha-era scripts
  are historical reference only, not supported, and excluded
  from wheel/sdist. The pre-move layout is recoverable via the
  annotated git tag `legacy-code-preserved` (T081).
- `scripts/audit_datasets.py` — release pre-publish gate that
  scans working tree and git history for deny-pattern filenames,
  exits non-zero on any match not in the maintained allow-list.
  Called from `release.yml` in US8 (T083).

**US1 Contract Integrity (T016..T030):**

- `ALGORITHMS` dict registry, `@register(name)` decorator, and
  `get_strategies()` helper in a new
  `bin_packer_3d.algorithms.registry` module — the single source of
  truth for valid strategy keys (ADR-0001, FR-002).
- `ColumnMapping` dataclass in `bin_packer_3d.data.loaders`
  (`length`/`width`/`height` required; `identifier`/`description`/
  `weight` optional) — the authoritative mapping from on-disk column
  names to `Box` fields (FR-004).
- `PackerConfig.get_strategies()` classmethod mirroring
  `ALGORITHMS.keys()` (FR-002).
- `complexity` and `description` `ClassVar[str]` attributes on
  `PackerBase` (defaulting to `"unknown"`), overridden on each
  concrete packer — sourced by `bin-packer info` (FR-003).
- Public re-exports on `bin_packer_3d.__init__`: `ALGORITHMS`,
  `register`, `get_strategies`, `ColumnMapping`.
- New tests: `tests/unit/test_registry.py` (4 cases),
  `tests/unit/test_data_loaders.py` (3 cases),
  `tests/integration/test_cli.py::test_info_lists_registry`, and
  `TestBox.test_weight_*` additions in `tests/unit/test_models.py`.

**US2 Continuous Integration Pipeline — bootstrap (T031..T048, T049 deferred):**

- `.github/workflows/ci.yml` — thin dispatcher with
  concurrency-cancellation per ADR-0003. Triggers on push to `main`
  and PR open/sync/reopen/ready_for_review (FR-010).
- `.github/workflows/_ci-core.yml` — reusable quality-gate core
  exposing `outputs.all_passed` for downstream callers. Six required
  jobs (`lint`, `format-check`, `type-check`, `test`, `pip-audit`,
  `pre-commit-parity`) plus an `aggregate` OR-bar (FR-010, FR-011,
  FR-014, FR-017).
- `.github/workflows/pr-title.yml` — Conventional Commits enforcement
  on PR titles via `amannn/action-semantic-pull-request`; allowed
  type list mirrors Constitution §Commit style verbatim.
- `.github/workflows/security.yml` — CodeQL (Python) on push, PR, and
  Monday cron with `security-and-quality` queries (FR-014, advisory
  at v0.2.0).
- `codecov.yml` — `library` flag scoped to `src/bin_packer_3d/`,
  90 % project + patch targets, `range: 70...100`. Hard
  `--cov-fail-under=90` gate stays OFF until T049 (FR-012).
- `.github/dependabot.yml` — weekly pip + github-actions updates
  targeting `main` (Constitution §Dependency policy).
- `.github/ISSUE_TEMPLATE/bug_report.yml` and
  `.github/ISSUE_TEMPLATE/feature_request.yml` — structured GitHub
  issue forms (FR-016).
- `.github/PULL_REQUEST_TEMPLATE.md` — Summary / linked spec /
  changes / testing checklist / breaking-change checklist /
  eight-principle Constitution impact walk (Constitution §PR workflow).
- `.github/CODEOWNERS` — `* @Bruno-Ghiberto`.
- `docs/maintainers.md` (T040 hard stop) — operational notes for
  branch-protection rules, required-status-check table, Codecov
  activation, and reserved sections for T140 OIDC publisher / T139
  GHCR visibility / T072 Pages source / T157 repo metadata.
- `pyyaml >= 6.0` added to dev extras (explicit; previously transitive
  via pre-commit) for the new YAML-parsing tests.
- New tests: `tests/integration/test_ci_workflows.py` (5 cases —
  parametrised over the four workflow files plus a parity check),
  `tests/integration/test_precommit_parity.py` (2 cases —
  `pre-commit-parity` job design + sanity).

### Changed

**Setup:**

- **Python floor raised from `>=3.10` to `>=3.11`** (public) — matches
  Constitution v1.0.1 §Technology Baseline and Spec FR-011. Maintainer
  develops on Python 3.14.3; CI matrix covers 3.11 / 3.12 / 3.13 / 3.14.
- `__author__` in `bin_packer_3d.__init__` aligned with `pyproject.toml`
  `authors` — both now read `Bruno Ghiberto` (FR-034).
- Project version bumped to `0.2.0.dev0` — Phase A development cycle.
- `ruff` floor raised to `>=0.4.0` to support the `lint.` sub-table and
  pydocstyle rules.

**Foundational (drive-by baseline cleanup forced by pre-commit hooks):**

- Renamed ambiguous loop variable `l` → `length` in `algorithms/base.py`
  and `algorithms/ffd.py` (ruff E741).
- Renamed unused loop variables `w, depth` → `_w, _depth` in
  `algorithms/shelf.py` (ruff B007).
- Added one-line docstrings to `__init__`, `__post_init__`, `__repr__`,
  and `name` properties across `algorithms/base.py`, `ffd.py`, `shelf.py`
  (ruff D102/D105/D107).
- Fixed D212 docstring-summary-on-line-2 issue in `__init__.py`.
- Normalized whitespace and import ordering across `algorithms/`.

**US1 BREAKING (v0.2.0):**

- `Box.weight: float` default `0.0` changed to `float | None` default
  `None` so "unknown" is distinguishable from "known zero" (FR-005).
  **Migration:** `Box(weight=0.0)` semantics unchanged; implicit `Box()`
  now yields `weight=None`. Callers relying on an implicit zero must
  pass `weight=0.0` explicitly.
- `load_boxes_from_csv` / `load_boxes_from_excel` now return
  `LoadReport` rather than `list[Box]`. **Migration:** replace
  `boxes = load_boxes_from_csv(path)` with
  `report = load_boxes_from_csv(path); boxes = report.boxes`.
  `report.rejected_rows` (structured) and `report.warnings` expose
  per-row failures (FR-053).

**US4 (v0.2.0):**

- `CODE/` directory relocated to `legacy/` via `git mv` (rename
  tracking preserved). The 7 hardcoded Windows paths (FR-030
  offenders) now live under `legacy/` — out of the T075
  public-tree scan scope. Pre-move HEAD preserved as the
  annotated git tag `legacy-code-preserved` (T081, FR-031,
  SC-009).
- `.pre-commit-config.yaml` gains a top-level `exclude: '^legacy/'`
  so hooks (trailing-whitespace, ruff, ruff-format) no longer
  touch preserved reference scripts.
- `pyproject.toml` `[tool.ruff]` gains
  `extend-exclude = ["legacy", "DATASETS"]` so local `ruff check`
  runs match CI behaviour.

**US1 (v0.2.0):**

- `PackerConfig.strategy` type narrowed from
  `Literal["ffd", "bfd", "shelf", "extreme_points"]` to `str` +
  `@field_validator` checking membership in `ALGORITHMS`. Phantom
  `bfd` and `extreme_points` strategies are no longer advertised
  (FR-001, Principle I Contract Honesty). Unknown strategy error now
  lists every registered name (FR-047).
- `bin-packer info` sources its algorithm table from `ALGORITHMS` at
  runtime with Strategy / Complexity / Description columns read from
  each packer's `ClassVar`s (FR-003). Adding a new packer via
  `@register(...)` auto-extends the output.
- `bin-packer pack --strategy` `click.Choice` is generated from
  `sorted(ALGORITHMS)`; the CLI strategy list stays in lockstep with
  the registry.
- `bin_packer_3d.data.loaders` logs per-row failures via
  `get_logger("data.loaders").warning(...)` in place of
  `print("Warning: ...")` — no silent `continue` (Constitution §V).
- `Bin.total_weight` now skips boxes with `weight is None` rather
  than summing `None` into a float (spec §Edge Cases).

**US2 (v0.2.0):**

- `README.md` badge block swapped to live signals: GitHub Actions CI,
  Codecov coverage (library flag), Python `3.11 | 3.12 | 3.13 | 3.14`,
  MIT licence, and Ruff. Stale badges removed (Python 3.10+, vanity
  "39 tests passing", Black). PyPI badge lands in US8 T143; README
  body overhaul lands in US3 T074. (FR-015)
- Every `uses:` clause in `.github/workflows/*.yml` is pinned to a
  full 40-char commit SHA with friendly version comment. Dependabot
  rotates on a weekly cadence (T035).

### Fixed

**US4 Repository Hygiene:**

- `CODE/*.py` with hardcoded `C:\Users\bghiberto\source\repos\…`
  paths no longer sits at the repo root; moved to `legacy/` with
  a clearly-marked README so visitors understand the directory
  is preserved historical reference, not supported code (FR-031).
- `scripts/audit_datasets.py` self-test on current repo finds 10
  deny-pattern matches across working tree + history; all 10 are
  in ALLOW → exit 0. Any new dataset landing in the repo without
  a matching `AUDIT.md` entry + ALLOW update will fail this gate
  at release time (FR-033).

**US1 Contract Integrity:**

- Phantom strategies `bfd` and `extreme_points` exposed via
  `PackerConfig.strategy` but backed by no implementation —
  an Alpha-era credibility bug flagged during audit. Construction
  now fails loudly with an error naming every registered strategy.
- `data/loaders.py` previously hardcoded `CAJA` / `DESCRIPCION` /
  `PESO` column names outside the configurable mapping — broke for
  any CSV that didn't originate from the maintainer's legacy
  dataset (FR-004).
- Loader previously swallowed row-level parse errors with a bare
  `print(...)` statement — the caller had no programmatic way to
  inspect which rows were dropped or why.

### Removed

- `[tool.black]` section from `pyproject.toml`. Formatting is now handled
  by `ruff format` (black-compatible) per Constitution v1.0.1 amendment.
  The standalone `black` tool is no longer a project dependency.
- Hardcoded algorithm listing in `cli.py::info` — the table is now
  computed from `ALGORITHMS` (US1 T029).
- Redundant `### Changed` subsection duplicating Setup entries.

## [0.1.0] — 2026-04-22

### Added

- Initial alpha release: FFD and Shelf heuristics, Plotly 3D
  visualisation, Pydantic configuration, Click CLI, pandas data loaders.

[Unreleased]: https://github.com/Bruno-Ghiberto/3D_BIN_PACKING/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Bruno-Ghiberto/3D_BIN_PACKING/releases/tag/v0.1.0
