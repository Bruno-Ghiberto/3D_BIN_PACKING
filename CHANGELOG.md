# Changelog

All notable changes to `bin-packer-3d` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
as interpreted by the project constitution (see
`.specify/memory/constitution.md` §Release workflow).

## [Unreleased]

### Added

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

### Changed

- **Python floor raised from `>=3.10` to `>=3.11`** (public) — matches
  Constitution v1.0.1 §Technology Baseline and Spec FR-011. Maintainer
  develops on Python 3.14.3; CI matrix covers 3.11 / 3.12 / 3.13 / 3.14.
- `__author__` in `bin_packer_3d.__init__` aligned with `pyproject.toml`
  `authors` — both now read `Bruno Ghiberto` (FR-034).
- Project version bumped to `0.2.0.dev0` — Phase A development cycle.
- `ruff` floor raised to `>=0.4.0` to support the `lint.` sub-table and
  pydocstyle rules.

### Removed

- `[tool.black]` section from `pyproject.toml`. Formatting is now handled
  by `ruff format` (black-compatible) per Constitution v1.0.1 amendment.
  The standalone `black` tool is no longer a project dependency.

## [0.1.0] — 2026-04-22

### Added

- Initial alpha release: FFD and Shelf heuristics, Plotly 3D
  visualisation, Pydantic configuration, Click CLI, pandas data loaders.

[Unreleased]: https://github.com/Bruno-Ghiberto/3D_BIN_PACKING/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Bruno-Ghiberto/3D_BIN_PACKING/releases/tag/v0.1.0
