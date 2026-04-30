# Implementation Plan: Public-Release Hardening of `bin-packer-3d`

**Branch**: `001-public-release-hardening` | **Date**: 2026-04-23 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-public-release-hardening/spec.md`
**Constitution**: [v1.0.0](../../.specify/memory/constitution.md)

## Summary

Take the existing `bin-packer-3d` Python package (currently `0.1.0` / Alpha) and
harden it into a credible public open-source release through three phased
milestones — `v0.2.0` (Foundations), `v0.3.0` (Capability Expansion), and
`v1.0.0` (Public Release). The primary requirement is a library whose public
API, CLI, configuration schema, and documentation tell the same story (Contract
Honesty), enforced by a continuous integration pipeline that is the ground
truth for "green" (Automated Quality Gates). The approach layers on the
existing `src/bin_packer_3d/` tree — extending it with a registry-driven
algorithm portfolio (FFD, Shelf existing; BFD, Extreme Point, Maximal Rectangles
new), a constraint framework (visitor pattern over pydantic models), a
benchmark runner with reproducible outputs against Bischoff & Ratcliff
1995 reference instances, a structured-logging observability layer on the
stdlib `logging` module, MkDocs Material documentation published to GitHub
Pages, and a PyPI release pipeline using OIDC trusted publishing. No
greenfield rewrite; every change is additive or a targeted repair of an
existing contract defect.

## Technical Context

**Language/Version**: Python 3.11 (public floor per constitution §Technology
Baseline — bumped from current `>=3.10`); maintainer develops on Python 3.14.3;
CI matrix covers 3.11 / 3.12 / 3.13 / 3.14 on Linux.

**Primary Dependencies** (runtime unless noted):
- `numpy >= 1.24` — geometric calculations, array operations (existing)
- `plotly >= 5.18` — 3D interactive visualisation (existing)
- `pydantic >= 2` + `pydantic-settings >= 2` — config and constraint models (existing)
- `click >= 8.1` — CLI framework (existing)
- `pandas >= 2.0` + `openpyxl >= 3.1` — CSV / Excel loaders (existing)
- `rich >= 13.0` — CLI user-facing output (existing)
- `pytest >= 7.4` + `pytest-cov >= 4.1` + `pytest-mock >= 3.12` — test runner (dev)
- `hypothesis >= 6.100` — property-based testing (dev, NEW)
- `ruff >= 0.4` — linting and formatting (dev, existing)
- `mypy >= 1.8` — strict static type checking (dev, existing)
- `pre-commit >= 3.6` — local hook runner (dev, existing)
- `pip-audit >= 2.9` — dependency vulnerability scanner (dev, NEW)
- `mkdocs-material >= 9.5` + `mkdocstrings[python] >= 0.25` — documentation
  (docs extra, NEW)

**Storage**: no database. Files only. CSV / XLSX datasets read at runtime via
`bin_packer_3d.data.loaders`; benchmark results written to JSON at
`docs/benchmarks/results/` via `bin_packer_3d.benchmark.runner`. Reference
benchmark instances (Bischoff & Ratcliff 1995 BR1–BR8) bundled at
`benchmark/instances/` when the license permits, otherwise fetched via a
downloader script.

**Testing**: `pytest` with `pytest-cov` (`--cov=src/bin_packer_3d
--cov-fail-under=90` — 90 % floor per Constitution III and Spec FR-012).
Markers: `unit`, `integration`, `property`, `slow`. Hypothesis profile `ci`
uses a fixed seed for deterministic CI runs (Principle IV).

**Target Platform**: Linux x86_64 (CI and container image). macOS and Windows
are best-effort — the pure-Python package installs cleanly on both but is not
CI-verified at `v1.0.0`.

**Project Type**: pure-Python library with a CLI entry point (`bin-packer`),
published to PyPI as `bin-packer-3d` (with a fallback list if taken). Single
package, single import root, `py.typed` marker shipped per PEP 561.

**Performance Goals** (measure-first per Constitution VIII):
- Baseline benchmark captured in Phase B against Bischoff & Ratcliff BR1 on
  every registered algorithm — utilisation %, bins used, success rate, runtime,
  with a fixed `--seed`.
- No numeric performance target committed before the baseline is measured.
- Regression guard: ≥ 25 % utilisation drop or ≥ 25 % runtime increase on the
  BR1 reference instance surfaces a CI warning that must be acknowledged in
  the PR description (Constitution §Benchmark workflow).
- **Reference hardware** for all "commodity laptop" claims in this spec:
  16 GB RAM, mid-range x86_64 CPU (4–8 physical cores, base clock ≥ 2.0 GHz,
  released 2022 or later), no discrete GPU. The per-push CI benchmark
  (`benchmark-br1.yml`) runs on `ubuntu-latest` GitHub-hosted runner
  (currently 4 vCPU, 16 GB RAM) — this is the authoritative baseline
  environment for all published benchmark numbers.

**Constraints**:
- Zero new mandatory runtime dependencies beyond the list above — any addition
  requires ≥ 2 modules of use, ≥ 50 lines of hand-written code removed, or a
  one-line ADR (Constitution V — Library Citizenship).
- No `print()` calls in `src/bin_packer_3d/` except inside the CLI module's
  Rich-backed output — enforced by CI grep (Constitution V).
- No `logging.basicConfig()` or root-logger configuration at library level —
  module loggers with `NullHandler` only (Constitution V, FR-050).
- No long-lived API tokens in repository secrets — PyPI publishing uses OIDC
  trusted publisher (Constitution §Release workflow, FR-070).
- Coordinate convention locked: `X = length`, `Y = width`, `Z = height` —
  changing requires a MAJOR constitution amendment.
- Determinism: every algorithm and benchmark with randomness accepts a seed
  and produces bit-identical output for the same `(seed, input, config)`
  (Constitution IV, FR-043).

**Scale/Scope**: single maintainer (Bruno Ghiberto). Expected input size:
10–200 boxes per packing run; reference benchmarks go to ~1 000 boxes on the
BR suite. Repository is a CV artefact targeting senior engineers, hiring
managers, and operations-research practitioners.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Each principle from `.specify/memory/constitution.md` v1.0.0 is instantiated as
a pass / fail gate. Gates marked **NON-NEG.** cannot be waived — a violation
requires a constitution amendment, not a plan exception.

### G1 — Contract Honesty (Principle I, NON-NEG.)

**Pass criterion**: A single algorithm registry is the source of truth. The
accepted `PackerConfig.strategy` values equal, exactly, the registry's keys,
asserted by an automated test (FR-006). The CLI `info` command lists algorithms
from the registry — never hardcoded (FR-003). The CSV / Excel loader consumes
a declared column mapping; optional columns never raise `KeyError` (FR-004).
The `Box` model distinguishes "weight unknown" from "weight = 0" (FR-005).

**How the plan satisfies it**:
- ADR-010 (research.md) ratifies the registry pattern using a module-level
  `ALGORITHMS: dict[str, type[BasePacker]]` in `algorithms/__init__.py` with a
  `@register` decorator.
- `PackerConfig.strategy` becomes a `Literal` derived from registry keys at
  import time via a runtime-generated `Annotated[str, ...]`.
- `tests/unit/test_registry.py` asserts `set(PackerConfig.get_strategies()) ==
  set(ALGORITHMS.keys())`.
- Phase A deliverable — blocks Phase A sign-off.

**Status**: PASS (design addresses every FR).

### G2 — Test-First Discipline (Principle II, NON-NEG.)

**Pass criterion**: Tests for every new or materially-changed public symbol are
committed BEFORE the implementation commit. Property-based tests via
`hypothesis` cover algorithm invariants (FR-046). Integration tests cover
end-to-end CLI flows.

**How the plan satisfies it**:
- Every task in the Phase B implementation task list (generated by
  `/speckit.tasks` later) is paired with its test task preceding it in the
  dependency graph.
- `tests/property/test_invariants.py` is authored first, parametrised over
  every algorithm registered via `pytest.mark.parametrize` seeded from
  `ALGORITHMS`. Adding a new algorithm automatically enrols it in the
  property suite.
- Commit messages on public-surface-touching commits include the line
  `tests authored first`.

**Status**: PASS (design enforces ordering).

### G3 — Automated Quality Gates (Principle III, NON-NEG.)

**Pass criterion**: CI runs on every push to `main` and every pull request
executing lint (ruff), format-check (ruff format --check — black is deprecated
by the constitution update), strict type check (`mypy --strict`), unit tests,
integration tests, coverage (≥ 90 % on `src/bin_packer_3d/`), and dependency
vulnerability scan. A PR that fails any gate — or drops below 90 % coverage —
is blocked from merging (FR-010, FR-011, FR-012, FR-017).

**How the plan satisfies it**:
- ADR-003 (research.md) locks the workflow architecture: thin dispatcher
  (`ci.yml`) + reusable core (`_ci-core.yml`) + publish workflows that gate on
  `_ci-core.yml`'s `all_passed` output.
- CI matrix covers Python 3.11 / 3.12 / 3.13 / 3.14 on `ubuntu-latest`.
- `pytest --cov-fail-under=90` hard-enforces the coverage floor.
- `pip-audit` runs in the matrix as a required job.
- `codecov/codecov-action` uploads coverage — public badge on README.
- Branch protection on `main` requires every required status check to pass.

**Status**: PASS (design delivers every CI gate the constitution mandates).

### G4 — Reproducibility & Determinism (Principle IV)

**Pass criterion**: Every randomised algorithm, test, and benchmark accepts a
seed and produces bit-identical output for the same `(seed, input, config)`
(FR-043). Benchmarks emit dated JSON artefacts stored with the build (FR-045).

**How the plan satisfies it**:
- `bin-packer benchmark --seed <N>` wired through the runner; benchmark runs
  without a seed generate a timestamp-based seed and print it.
- CI benchmark job (`benchmark-br1`) attaches the BR1 JSON artefact to every
  build; the release workflow additionally runs the full BR1–BR8 suite.
- `hypothesis` CI profile uses a deterministic seed (`HYPOTHESIS_DATABASE =
  ':memory:'`, `Phase.explicit`); failures print the reproducing example.

**Status**: PASS.

### G5 — Library Citizenship (Principle V)

**Pass criterion**: No `print()` in `src/bin_packer_3d/` outside `cli.py`
(FR-052). No `logging.basicConfig()` (FR-050). No `os.chdir`, `sys.path`
mutation, env writes at import time. No network or blocking I/O at import.
Errors via typed exceptions; never `sys.exit()` from library code.

**How the plan satisfies it**:
- Pre-commit hook + CI grep check fails on `print(` matches outside
  `src/bin_packer_3d/cli.py` (Phase A deliverable).
- `tests/unit/test_library_hygiene.py` asserts that `import bin_packer_3d`
  leaves root-logger handlers empty and does not raise on a clean interpreter
  with stdin redirected and `HOME` unset.
- ADR-008 (observability) commits to stdlib logging only — no third-party
  logging framework at runtime.

**Status**: PASS.

### G6 — Documentation as Artefact (Principle VI, NON-NEG.)

**Pass criterion**: Every public class and function has a docstring (FR-026);
missing docstrings fail CI. Documentation site builds in CI and publishes on
every push to `main` (FR-020). `docs/adr/` contains ≥ 3 accepted ADRs (FR-024).
README passes the 30-second credibility test (FR-022, SC-001). `CONTRIBUTING.md`,
`CHANGELOG.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md` present and linked
(FR-023).

**How the plan satisfies it**:
- ADR-002 locks MkDocs Material + mkdocstrings.
- `ruff` `D` ruleset (pydocstyle) enabled on `src/bin_packer_3d/` — missing
  docstrings break the lint job.
- `mkdocs build --strict` added as a required CI job.
- research.md includes ≥ 8 ADRs (exceeds FR-024's minimum of 3).
- README overhaul is a Phase C deliverable.
- Community files (`CONTRIBUTING`, `CODE_OF_CONDUCT`, `SECURITY`) are Phase C
  deliverables; `CHANGELOG.md` starts in Phase A with `[0.2.0]` and Unreleased
  sections.

**Status**: PASS.

### G7 — Privacy by Default (Principle VII)

**Pass criterion**: No identifiable real-world business, personal, or customer
data in the working tree OR git history (FR-033). `DATASETS/README.md`
enumerates every file (FR-032). A pre-release audit script scans tree + history
for business-artefact patterns.

**How the plan satisfies it**:
- Phase A includes a mandatory per-file audit of `DATASETS/*.xlsx` — the
  Spanish packing lists (`PACKING LIST.xlsx`, `PACKING LIST-11.xlsx`,
  `DIMENSIONES CAJAS-NORMALIZADO.xlsx`, `PESO_P.T.xlsx`) identified in the
  spec clarification session.
- Audit outcome documented in `DATASETS/AUDIT.md` with per-file disposition
  (keep / anonymise / remove + scrub from history).
- Files classified confidential are scrubbed via `git filter-repo` — the
  remediation happens BEFORE any public push.
- A pre-release audit script at `scripts/audit_datasets.py` is added in Phase
  C and runs in the release workflow as a blocking gate.

**Status**: PASS (Phase A blocks on audit completion).

### G8 — Performance Discipline (Principle VIII)

**Pass criterion**: Optimisations are justified by measurement. Skipping rungs
on the optimisation ladder requires an ADR. Native extensions (Rust, C,
Cython-compiled units) are OUT OF SCOPE for this phase (Spec §Out of Scope).

**How the plan satisfies it**:
- ADR-005 (research.md) ratifies measure-first. No performance optimisation
  work is scheduled until Phase B captures the BR1 baseline.
- Spec §Out of Scope explicitly rules out native extensions for this phase;
  a future ADR is required to introduce one.
- Gate G11 (in the full gate list) blocks Phase C perf work without a baseline.

**Status**: PASS.

### Gate summary

| Gate | Principle | Status |
|---|---|---|
| G1 | I. Contract Honesty (NON-NEG.) | PASS |
| G2 | II. Test-First Discipline (NON-NEG.) | PASS |
| G3 | III. Automated Quality Gates (NON-NEG.) | PASS |
| G4 | IV. Reproducibility & Determinism | PASS |
| G5 | V. Library Citizenship | PASS |
| G6 | VI. Documentation as Artefact (NON-NEG.) | PASS |
| G7 | VII. Privacy by Default | PASS |
| G8 | VIII. Performance Discipline | PASS |

**Overall**: All 8 principles PASS at plan authoring time. No amendments
required. No complexity-tracking entries needed.

## Project Structure

### Documentation (this feature)

```text
specs/001-public-release-hardening/
├── spec.md                      # ratified feature specification (input)
├── plan.md                      # this file (/speckit.plan output)
├── research.md                  # Phase 0 output — 10 ADRs
├── data-model.md                # Phase 1 output — entities + relationships
├── quickstart.md                # Phase 1 output — 5-command install+run flow
├── contracts/                   # Phase 1 output
│   ├── api.md                   # public Python API surface
│   ├── cli.md                   # CLI commands, flags, exit codes
│   ├── config-schema.md         # PackerConfig fields and versioning
│   └── benchmark-format.md      # benchmark JSON schema
├── checklists/                  # existing (requirements.md, usability.md, etc.)
└── tasks.md                     # Phase 2 output (/speckit.tasks — NOT created here)
```

### Source Code (repository root)

```text
src/bin_packer_3d/
├── algorithms/
│   ├── __init__.py              # registry: ALGORITHMS dict + @register decorator
│   ├── base.py                  # BasePacker ABC (existing)
│   ├── ffd.py                   # First Fit Decreasing (existing — register)
│   ├── shelf.py                 # Shelf heuristic (existing — register)
│   ├── bfd.py                   # Best Fit Decreasing (NEW — Phase B, FR-040)
│   ├── extreme_point.py         # Extreme Point — Crainic/Perboli/Tadei 2008 (NEW — Phase B)
│   └── maximal_rectangles.py    # Maximal Rectangles family (NEW — Phase B, FR-040)
├── benchmark/                   # NEW — US5, Phase B
│   ├── __init__.py
│   ├── runner.py                # BenchmarkRunner; seed-accepting
│   ├── results.py               # BenchmarkResult + serialisers
│   ├── instances.py             # BR1–BR8 loader (bundled or downloader)
│   └── formats.py               # text / JSON / Markdown output
├── constraints/                 # NEW — US7, Phase B
│   ├── __init__.py
│   ├── base.py                  # Constraint ABC + ConstraintRegistry
│   ├── orientation.py           # AllowedOrientations constraint (FR-061)
│   └── weight.py                # SupportedWeight constraint (FR-062)
├── models/
│   ├── __init__.py
│   ├── box.py                   # existing + weight Optional + allowed_orientations (Phase A/B)
│   ├── bin.py                   # existing + max_weight Optional (Phase A)
│   ├── placement.py             # existing
│   ├── result.py                # NEW — PackingResult, LoadReport (Phase A)
│   └── metadata.py              # NEW — AlgorithmMetadata (Phase B)
├── visualization/
│   ├── __init__.py
│   └── plotter.py               # existing
├── data/
│   ├── __init__.py
│   └── loaders.py               # existing — column mapping fix Phase A (FR-004)
├── utils/
│   ├── __init__.py
│   └── metrics.py               # existing
├── observability.py             # NEW — US6, Phase B (stdlib logging adapter)
├── cli.py                       # existing + info/benchmark/pack --explain (Phase A & B)
├── config.py                    # existing + dynamic Literal from registry (Phase A)
├── __init__.py                  # public API surface (updated each phase)
├── __main__.py                  # existing
└── py.typed                     # NEW — PEP 561 marker (Phase A, FR-035)

tests/
├── conftest.py                  # existing fixtures + new registry/benchmark fixtures
├── unit/
│   ├── test_models.py           # existing + weight-unknown, metadata
│   ├── test_algorithms.py       # existing + bfd, extreme_point, maximal_rectangles
│   ├── test_registry.py         # NEW — FR-006
│   ├── test_constraints.py      # NEW — US7
│   ├── test_observability.py    # NEW — US6
│   ├── test_library_hygiene.py  # NEW — Principle V enforcement
│   └── test_data_loaders.py     # NEW + existing — FR-004
├── integration/
│   ├── test_packing.py          # existing
│   ├── test_cli.py              # NEW — end-to-end CLI flows
│   ├── test_benchmark.py        # NEW — FR-041..FR-045
│   └── test_docs_consistency.py # NEW — docs-vs-runtime sync (US1 AC6)
└── property/                    # NEW — hypothesis
    └── test_invariants.py       # FR-046: overlap, bounds, conservation

benchmark/                       # NEW at repo root — DEV-ONLY, excluded from sdist
├── instances/                   # BR1–BR8 bundled if license allows (cache dir)
├── run_baseline.py              # standalone baseline capture (calls BenchmarkRunner)
└── download_instances.py        # thin CLI wrapper around the installed package's
                                 # `bin_packer_3d.benchmark.download.fetch_br`
                                 # — convenient for maintainers, not shipped to users
#
# NOTE: the INSTALLED package also has `src/bin_packer_3d/benchmark/` containing
# runner.py, results.py, instances.py, formats.py, download.py. The package-
# internal `download.py` is the runtime-reachable implementation; `benchmark/
# download_instances.py` at repo root is purely a dev-convenience CLI that
# pre-populates `benchmark/instances/` for local iteration. End users never see
# repo-root `benchmark/` — it is excluded by `MANIFEST.in` and the hatch build
# config.

docs/                            # NEW — MkDocs Material
├── index.md                     # landing page + hero visualisation
├── problem.md                   # 3D-BPP intro + coordinate-convention diagram
├── quickstart/
│   ├── cli.md
│   └── python.md
├── algorithms/                  # one page per algorithm (FR-021)
│   ├── ffd.md
│   ├── shelf.md
│   ├── bfd.md
│   ├── extreme_point.md
│   └── maximal_rectangles.md
├── constraints.md
├── benchmarks.md                # regenerated from CI artefact
├── api/
│   └── index.md                 # mkdocstrings auto-generated
├── adr/                         # FR-024 — architecture decision records
│   ├── 0001-algorithm-registry.md
│   ├── 0002-documentation-tooling.md
│   ├── 0003-ci-workflow.md
│   ├── 0004-docs-host.md
│   ├── 0005-performance-discipline.md
│   ├── 0006-pypi-name-and-distribution.md
│   ├── 0007-constraint-framework.md
│   ├── 0008-observability.md
│   ├── 0009-coordinate-convention.md
│   └── 0010-algorithm-portfolio.md
├── contributing.md              # links to CONTRIBUTING.md
└── changelog.md                 # include of CHANGELOG.md

examples/                        # NEW — FR-025
└── compare_algorithms.ipynb     # runnable notebook: load, run 3 algos, plot

.github/
├── workflows/
│   ├── ci.yml                   # thin dispatcher
│   ├── _ci-core.yml             # reusable quality gate
│   ├── release.yml              # tag v*.*.* → PyPI (OIDC) → docs-deploy → Docker
│   ├── security.yml             # CodeQL (Python), weekly + on push
│   ├── pr-title.yml             # Conventional Commits enforcement
│   └── benchmark-br1.yml        # per-push BR1 benchmark + artefact
├── ISSUE_TEMPLATE/
│   ├── bug_report.yml
│   └── feature_request.yml
├── PULL_REQUEST_TEMPLATE.md
├── CODEOWNERS
└── dependabot.yml

# repo-root files
mkdocs.yml                       # MkDocs Material config
Dockerfile                       # multi-stage, non-root (FR-072)
.dockerignore
.pre-commit-config.yaml          # mirrors CI (FR-013)
.editorconfig                    # FR-036
CHANGELOG.md                     # Keep-a-Changelog format (FR-023)
CONTRIBUTING.md                  # 5-command setup path (FR-027, SC-007)
CODE_OF_CONDUCT.md               # Contributor Covenant
SECURITY.md                      # disclosure channel (Constitution §Security)
codecov.yml                      # 90 % threshold, library flag
pyproject.toml                   # requires-python >=3.11; author alignment
MANIFEST.in                      # include py.typed; exclude CODE/
README.md                        # 30-second credibility overhaul (Phase C)

# preserved but out of public scope
CODE/                            # legacy scripts — moved to legacy/ or removed (Phase A/C)
DATASETS/                        # post-audit content + README.md (Phase A)
```

**Structure Decision**: Option 1 (single-project) adapted for a library-with-CLI.
The existing `src/bin_packer_3d/` tree is extended, not reorganised. Three new
top-level sibling modules (`benchmark/`, `constraints/`, `observability.py`)
are added inside the package. Tests gain a `property/` tier for Hypothesis.
Documentation gets a full `docs/` tree with MkDocs Material. Legacy `CODE/`
is excluded from the sdist via `MANIFEST.in` and relocated to `legacy/`
(with a README explaining it is historical reference only) or removed in
Phase A. Community files (`CONTRIBUTING`, `CHANGELOG`, `CODE_OF_CONDUCT`,
`SECURITY`), CI workflows, and tooling files are added at the repository
root.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

Not applicable — all 8 constitutional gates pass at plan authoring time. No
principle is violated; no complexity entries required.

## Phased Milestones

Three phases map to three semantic version milestones. Each phase ships a
coherent, shippable increment. Phase boundaries are hard — the next phase
cannot begin until its predecessor's gates are green.

### Phase A — Foundations (`v0.2.0`)

**User stories covered**: US1 (Contract Integrity), US3 (partial — setup path),
US4 (Repository Hygiene), US2 (partial — CI bootstrap).

**Deliverables**:
1. `pyproject.toml`: `requires-python = ">=3.11"`; `[tool.mypy]
   python_version = "3.11"`; `[tool.ruff] target-version = "py311"`;
   author alignment with `__init__.py`; add `D` ruff rules (pydocstyle).
2. `src/bin_packer_3d/py.typed` marker file (PEP 561, FR-035).
3. Algorithm registry in `algorithms/__init__.py`: `ALGORITHMS` dict,
   `@register` decorator, `get_strategies()` helper on `PackerConfig`.
   Existing FFD and Shelf implementations register themselves.
4. `PackerConfig.strategy` dynamically derives its `Literal` from registry
   keys; `tests/unit/test_registry.py` asserts the invariant (FR-006).
5. `Box.weight` becomes `Optional[float] = None` (FR-005); `Bin.max_weight`
   becomes `Optional[float] = None` with explicit "unlimited" semantics
   documented.
6. CSV / Excel loader refactored around a declared column mapping
   (`ColumnMapping` dataclass); optional columns absent do not raise
   `KeyError` (FR-004). `LoadReport` returned with warnings and rejected rows
   (FR-053).
7. `.editorconfig` at repo root (FR-036).
8. `.pre-commit-config.yaml`: ruff (lint + format), trailing-whitespace,
   end-of-file-fixer, check-yaml, check-toml, check-added-large-files,
   pydocstyle — mirrors CI.
9. CI bootstrap: `ci.yml` (dispatcher) + `_ci-core.yml` (single Python
   3.11 at Phase A; matrix expands in Phase B) + `pr-title.yml` +
   `dependabot.yml`. Jobs: lint, format-check, type-check, test, pip-audit,
   pre-commit-parity, aggregate.
10. Branch protection on `main` configured (via repo settings, documented in
    `docs/maintainers.md`): every required status check must pass; no force
    pushes.
11. `CHANGELOG.md` created (Keep-a-Changelog format) with `[0.2.0]` and
    `Unreleased` sections.
12. `DATASETS/AUDIT.md` authored after a per-file audit of every `.xlsx`
    (FR-033). Confidential files scrubbed via `git filter-repo`; safe files
    documented in `DATASETS/README.md`.
13. Legacy `CODE/` excluded from sdist via `MANIFEST.in`; either relocated to
    `legacy/` or removed on `main` (the history tag `legacy-code` preserves
    the old state).
14. PyPI name `bin-packer-3d` reserved by publishing a placeholder `0.2.0`
    wheel using the completed release workflow stub.
15. `tests/unit/test_library_hygiene.py` asserts Constitution V invariants on
    `import bin_packer_3d`.

**Gate checks before Phase A sign-off**: G1, G2, G3, G5, G6 (docstring
baseline), G7.

### Phase B — Capability Expansion (`v0.3.0`)

**User stories covered**: US5 (Algorithm Portfolio & Benchmarks), US6
(Observability), US7 (Constraints), US2 (completion — full CI matrix + per-push
benchmark).

**Deliverables**:
1. `algorithms/bfd.py` — Best-Fit Decreasing (FR-040). Tests authored first.
2. `algorithms/extreme_point.py` — Extreme Point per Crainic/Perboli/Tadei
   2008 (FR-040). Tests authored first.
3. `algorithms/maximal_rectangles.py` — Maximal Rectangles family (FR-040,
   fulfils "one additional family" clause). Tests authored first.
4. `constraints/` package: `Constraint` ABC + `ConstraintRegistry` +
   `AllowedOrientations` + `SupportedWeight` (FR-060..FR-064).
5. `observability.py` — stdlib logging with `NullHandler` + optional
   `StructuredAdapter` (FR-050..FR-054).
6. `benchmark/` package + Bischoff & Ratcliff BR1–BR8 instance loader. If
   license permits bundling, instances land at `benchmark/instances/`;
   otherwise `benchmark/download_instances.py` fetches on demand.
7. `bin-packer benchmark` CLI command (text / JSON / Markdown output,
   `--seed` support, `--algorithm` filter) (FR-041..FR-047).
8. `bin-packer pack --explain` mode — DEBUG per-box placement trace (FR-054).
9. CI matrix expanded to Python 3.11 / 3.12 / 3.13 / 3.14 in `_ci-core.yml`.
10. `benchmark-br1.yml` per-push workflow: runs BR1 on every registered
    algorithm, attaches JSON artefact (FR-045).
11. `security.yml` — CodeQL (Python-only), weekly cron + on push (ADVISORY).
12. `codecov.yml` — 90 % library-flag threshold; wired into `_ci-core.yml`.
13. `tests/property/test_invariants.py` — FR-046 parametrised over every
    registered algorithm.
14. `ruff` coverage gate activated: `--cov-fail-under=90`.
15. `CHANGELOG.md` `[0.3.0]` section authored.

**Gate checks before Phase B sign-off**: G1 (still), G2, G3, G4, G5, G6,
G8 (baseline captured, no perf work without it).

### Phase C — Public Release (`v1.0.0`)

**User stories covered**: US3 (completion), US8 (Distribution & Release),
US9 (Interactive Demo), US2 (final README badges).

**Deliverables**:
1. `docs/` tree fully authored: problem intro + coordinate-convention diagram
   (FR-021), per-algorithm reference pages with complexity / pseudo-code /
   citation / guidance (FR-021), constraints reference, benchmarks page
   regenerated from CI artefact, contributing guide, 10 ADRs at `docs/adr/`.
2. `mkdocs.yml` with MkDocs Material + mkdocstrings + search.
3. `mkdocs build --strict` job added as required in `_ci-core.yml`.
4. `CONTRIBUTING.md` — 5-command setup path (FR-027, SC-007).
5. `CODE_OF_CONDUCT.md` — Contributor Covenant v2.1.
6. `SECURITY.md` — disclosure channel.
7. `examples/compare_algorithms.ipynb` — runnable notebook (FR-025).
8. `Dockerfile` — multi-stage, non-root user (FR-072). Image published to
   GHCR (`ghcr.io/bruno-ghiberto/bin-packer-3d`) via `release.yml`.
9. `release.yml` completed: `ci` gate → `publish` (PyPI via OIDC) →
   `docs-deploy` (GitHub Pages) → `docker-publish` (GHCR) → BR1–BR8 full
   benchmark gate (FR-045).
10. OIDC trusted publisher configured on PyPI dashboard (`environment: pypi`).
11. `README.md` overhaul: hero visualisation, two-sentence problem statement,
    headline benchmark numbers (from BR1 CI artefact), ≤ 5 canonical use
    cases, single docs link, live badges (CI status, PyPI version, coverage,
    Python versions, license) (FR-015, FR-022, SC-001).
12. US9 interactive demo — static GitHub Pages page with a pre-rendered
    `plotly` HTML visualisation of a BR1 packing run, linked from the README
    (FR-080, FR-081).
13. GitHub repo settings: description, topics, homepage URL (GH Pages),
    issue templates linked, PR template present, CODEOWNERS present.
14. Pre-release audit script `scripts/audit_datasets.py` added; run as a
    blocking job in `release.yml`.
15. `CHANGELOG.md` `[1.0.0]` section authored; migration notes for any
    breaking changes (e.g. `Box.weight` type change).
16. Tag `v1.0.0` pushed → `release.yml` runs end-to-end → PyPI publish +
    GH Pages deploy + Docker image + GitHub Release with BR1–BR8 artefact.

**Gate checks before Phase C sign-off**: G3, G4, G6, G7, G8. Full constitution
compliance audit per §Compliance review.

## Risks, Non-Goals & Open Questions

### Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `bin-packer-3d` PyPI name taken | Low | Medium | Reserve as first Phase A action; fallback list in ADR-006 |
| `mypy --strict` reveals deep type errors in existing code | Medium | Medium | Fix incrementally in Phase A; narrow `type: ignore` with comment tracked in CHANGELOG |
| BR instance license forbids redistribution | Medium | Medium | `benchmark/download_instances.py` fetches on demand; document license in `DATASETS/README.md` |
| `hypothesis` finds a latent bug in existing FFD/Shelf | Medium | Low | Fix in Phase B as a bonus correctness improvement; logged in CHANGELOG |
| Docker image publishing to GHCR requires extra tokens | Low | Low | GHCR supports `GITHUB_TOKEN` with `packages: write`; no new secrets |
| Python 3.14 compatibility (PEP 695 syntax, ExceptionGroup) | Low | Low | CI matrix catches it; fix before tagging |
| Legacy `CODE/` referenced from docs or issues by users | Low | Low | Move to `legacy/` with README rather than delete on main |

### Non-Goals (Spec §Out of Scope, reaffirmed)

- Web UI or SaaS deployment.
- Distributed / parallel packing.
- Exact ILP solvers (Gurobi, CPLEX, OR-Tools CP-SAT).
- Reinforcement-learning packers.
- Multi-bin-type heterogeneous fleet optimisation.
- REST API.
- Internationalisation beyond English.
- Backwards compatibility with `CODE/` scripts.
- Native extensions (Rust, C, Cython-compiled units) — deferred to a future
  spec per Constitution VIII; requires an ADR with flamegraph + measured
  speedup justification.

### Open Questions

All spec-level ambiguities resolved in `/speckit.clarify` (Session 2026-04-22,
spec §Clarifications). No open questions remain. Plan-level refinements
(concrete test file layout, exact benchmark JSON schema) are resolved in the
Phase 1 artefacts (`data-model.md`, `contracts/`, `quickstart.md`).

## Artefact Inventory

Phase 0 / 1 artefacts generated by this `/speckit.plan` invocation:

- `research.md` — 10 ADRs resolving every technology / architecture decision
- `data-model.md` — 9 entities with fields, invariants, state transitions
- `contracts/api.md` — public Python API surface (14 symbols)
- `contracts/cli.md` — CLI command reference
- `contracts/config-schema.md` — `PackerConfig` versioned schema
- `contracts/benchmark-format.md` — benchmark JSON schema
- `quickstart.md` — 5-command install + run flow

Next phase (`/speckit.tasks`) consumes these artefacts to produce
`tasks.md` — a dependency-ordered task list grouped by user story.
