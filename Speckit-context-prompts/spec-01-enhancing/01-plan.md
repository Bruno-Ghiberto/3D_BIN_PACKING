# Plan Context — Phase 01: Public-Release Hardening of `bin-packer-3d`

> Hand this entire document to `speckit.plan` as the planning description.
> SpecKit will turn it into a rigorous plan.md with phased implementation,
> data-model.md, quickstart.md, and contracts/. The goal of THIS document is
> to give SpecKit enough signal that it never has to emit `[NEEDS CLARIFICATION]`.
>
> **Input spec**: `specs/001-public-release-hardening/spec.md` (365 lines,
> 9 user stories, 54 functional requirements, 14 success criteria).
> **Constitution**: `.specify/memory/constitution.md` (v1.0.0, 481 lines,
> 8 principles, 4 NON-NEGOTIABLE).

---

## 0. Brief

Take the existing `bin-packer-3d` project — a Python implementation of the 3D
Bin Packing Problem currently at `0.1.0 / Alpha` — and harden it into a
credible public open-source release that can be cited on a CV and stand up to
inspection by a senior engineer or hiring manager. This is a **level-up pass on
a working codebase**, not a greenfield rewrite.

The sole maintainer is Bruno Ghiberto. After this change ships, a recruiter or
peer reviewer landing on the repo should within 30 seconds conclude: *"this
person knows how to ship serious software."* The plan must produce a clean
public release at version `0.2.0` through `1.0.0` across three phases.

Key non-negotiables inherited from the constitution (v1.0.0):
- Contract Honesty (I): public API is exhaustively typed; no silent breaking changes.
- Test-First Discipline (II): tests precede implementation; coverage ≥ 85 % on `src/`.
- Automated Quality Gates (III): CI is the only authoritative pass/fail signal.
- Documentation (VI): every public symbol has a docstring; quickstart runs in < 5 min.

---

## 1. Primary Input References

The `/speckit.plan` invocation MUST read all three inputs before emitting any
artefact. Do not paraphrase or summarise — use the source documents directly.

| Artefact | Path | Role |
|---|---|---|
| Feature spec | `specs/001-public-release-hardening/spec.md` | 9 USs, 54 FRs, 14 SCs — authoritative scope |
| Constitution | `.specify/memory/constitution.md` | Non-negotiable principles and quality thresholds |
| Plan template | `.specify/templates/plan-template.md` | Output structure; every placeholder must be resolved |

**Environment baseline (maintainer machine)**:
- Python: 3.14.3 (CPython)
- OS: Linux x86_64 (Fedora)
- Package manager: pip / uv
- VCS: Git, remote at `github.com/Bruno-Ghiberto/3D_BIN_PACKING`
- CI platform: GitHub Actions (ubuntu-latest runners)

---

## 2. Technical Context

Fills the nine `[NEEDS CLARIFICATION]` placeholders in plan-template.md
§ Technical Context verbatim.

**Language/Version**: Python 3.11 (public floor); maintainer develops on 3.14.3;
CI matrix covers 3.11 / 3.12 / 3.13 / 3.14 on Linux.

**Primary Dependencies** (locked at major versions, pinned in CI):
- `numpy` — geometric calculations and array operations
- `plotly` — 3D interactive visualisation (existing)
- `pydantic >= 2` — config and constraint models (existing `PackerConfig`)
- `click` — CLI framework (existing `cli.py`)
- `pandas` / `openpyxl` — data loaders (existing `loaders.py`)
- `hypothesis` — property-based testing (NEW, dev-only)
- `pytest` / `pytest-cov` — test runner with coverage (existing)
- `ruff` — linting + formatting (existing in pyproject.toml)
- `mypy` — static type checking (existing)
- `mkdocs-material` + `mkdocstrings[python]` — documentation (NEW — ADR-002)

**Storage**: no database; files only — CSV and XLSX datasets read at runtime via
`loaders.py`; benchmark results optionally written to JSON via `benchmark/runner.py`.

**Testing**: `pytest` with `pytest-cov` (`--cov=src/bin_packer_3d
--cov-fail-under=85`); `hypothesis` for property-based invariant tests;
test markers: `unit`, `integration`, `property`, `slow`.

**Target Platform**: Linux x86_64 (CI); macOS and Windows are best-effort at the
user's discretion — no CI matrix for them at v1.0.0 (ADR-003 decision).

**Project Type**: pure-Python library with an optional CLI entry point, published
to PyPI as `bin-packer-3d` (ADR-006).

**Performance Goals** (Principle VIII, measure-first — ADR-005):
- Establish a baseline benchmark in Phase B (`v0.3.0`) before setting thresholds.
- Provisional target derived from domain convention: pack 50 boxes into a single
  bin in < 500 ms on a commodity laptop (to be confirmed by benchmark output).

**Constraints**:
- Zero new mandatory runtime dependencies beyond the list above unless a user
  story explicitly requires one (Principle V — Library Citizenship).
- `observability.py` MUST use stdlib `logging` only — no third-party logging
  frameworks at runtime (ADR-008).
- Constraint framework (ADR-007) depends on `pydantic` which is already a
  runtime dep — no additional dep introduced.

**Scale/Scope**: single maintainer; typical input: 10–200 boxes per packing run;
repository is a CV artefact targeting senior engineer / hiring manager reviewers.

---

## 3. Project Structure Decision

**Selected option**: Option 1 (single-project), adapted for a library with CLI,
plus three new top-level modules (`constraints/`, `benchmark/`, `observability.py`)
required by the spec.

The legacy `CODE/` directory (hardcoded Windows paths) is excluded from the
public surface and flagged for removal or `.gitignore` exclusion in Phase A.

### Source layout (concrete — no option labels)

```text
src/bin_packer_3d/
├── algorithms/
│   ├── __init__.py
│   ├── base.py                  # ABCs (existing)
│   ├── ffd.py                   # First Fit Decreasing (existing)
│   ├── shelf.py                 # Shelf heuristic (existing)
│   └── extreme_point.py         # Extreme Point (NEW — ADR-001, Phase B)
├── benchmark/                   # NEW — US5
│   ├── __init__.py
│   ├── runner.py                # BenchmarkRunner; emits BenchmarkResult
│   └── results.py               # BenchmarkResult dataclass (→ data-model.md)
├── constraints/                 # NEW — US7, Phase B
│   ├── __init__.py
│   ├── base.py                  # ConstraintVisitor ABC (ADR-007)
│   └── standard.py              # built-in constraints (weight, rotation lock)
├── models/
│   ├── __init__.py
│   ├── box.py                   # existing
│   ├── bin.py                   # existing
│   ├── placement.py             # existing
│   └── metadata.py              # NEW — AlgorithmMetadata (Phase B, US5)
├── visualization/
│   ├── __init__.py
│   └── plotter.py               # existing
├── data/
│   ├── __init__.py
│   └── loaders.py               # existing
├── utils/
│   ├── __init__.py
│   └── metrics.py               # existing
├── observability.py             # NEW — US6, Phase B (stdlib logging adapter)
├── cli.py                       # existing; strategy Literal fix in Phase A
├── config.py                    # existing; Python floor bump side-effects here
├── __init__.py                  # public API surface; py.typed sibling
├── __main__.py                  # existing
└── py.typed                     # NEW — PEP 561 marker (Phase A, US3)

tests/
├── conftest.py                  # existing fixtures
├── unit/
│   ├── test_models.py           # existing + new metadata tests
│   ├── test_algorithms.py       # existing + extreme_point tests
│   ├── test_constraints.py      # NEW — US7
│   └── test_observability.py    # NEW — US6
├── integration/
│   ├── test_packing.py          # existing
│   └── test_benchmark.py        # NEW — US5
└── property/                    # NEW — hypothesis invariants
    └── test_invariants.py       # packing overlap, bounds, volume conservation

benchmark/                       # NEW at repo root (excluded from sdist)
└── run_baseline.py              # standalone script for Phase B baseline capture
```

### Documentation layout

```text
docs/
├── index.md                     # project overview + badges
├── quickstart.md                # 5-command flow (§8 below)
├── api/
│   └── index.md                 # mkdocstrings auto-generated from docstrings
├── algorithms.md                # algorithm comparison + selection guide
├── cli.md                       # CLI reference (click --help mirror)
└── changelog.md                 # symlink or include of CHANGELOG.md

mkdocs.yml                       # MkDocs Material config (ADR-002, ADR-004)
```

### CI/CD layout

```text
.github/
├── workflows/
│   ├── ci.yml                   # thin dispatcher — calls _ci-core.yml
│   ├── _ci-core.yml             # reusable quality gate (all job logic here)
│   ├── release.yml              # PyPI publish via OIDC trusted publisher
│   ├── security.yml             # CodeQL scan — Python only, ADVISORY
│   └── pr-title.yml             # Conventional Commits enforcement
├── ISSUE_TEMPLATE/
│   ├── bug_report.yml
│   └── feature_request.yml
├── PULL_REQUEST_TEMPLATE.md
├── CODEOWNERS
└── dependabot.yml               # pip + github-actions ecosystems
```

### Tooling files

```text
.pre-commit-config.yaml          # ruff + ruff-format + trailing-whitespace + end-of-file-fixer + check-yaml + check-toml
.editorconfig                    # NEW — encoding, line endings, indent
CHANGELOG.md                     # NEW — conventional changelog (Phase A)
pyproject.toml                   # Python floor >=3.11 (was >=3.10); hatchling build
MANIFEST.in                      # ensure py.typed is included in sdist
```

**Structure Decision**: Option 1 (single-project) adapted for a library with CLI.
New modules (`constraints/`, `benchmark/`, `observability.py`) extend the existing
`src/bin_packer_3d/` tree without restructuring the working layout. Legacy `CODE/`
directory is removed from the published sdist via `MANIFEST.in` exclusion and
flagged with a deprecation notice; removal is deferred to Phase C to avoid
breaking any user who cloned the repo directly.

---

## 4. Constitutional Gates

Fills the `[Gates determined based on constitution file]` placeholder at
plan-template.md line 34. Each gate maps to a principle from
`.specify/memory/constitution.md` v1.0.0.

Gates MUST be checked before Phase A begins (pre-implementation) and re-checked
after Phase B (pre-v1.0.0 release candidate).

| Gate | Principle | Pass criterion | Fail action |
|---|---|---|---|
| **G1 — API contract** | I. Contract Honesty (NON-NEG.) | All exported symbols in `__init__.py` have complete type annotations; `mypy --strict` passes with zero errors on `src/`; `py.typed` marker present | Block Phase A sign-off |
| **G2 — Test-first** | II. Test-First (NON-NEG.) | Tests for each FR exist and are committed BEFORE the implementation commit; `pytest` green on Python 3.11 | Block implementation of that FR |
| **G3 — Coverage** | II. Test-First (NON-NEG.) | `pytest --cov=src/bin_packer_3d --cov-fail-under=85` passes; no module below 70 % | Block Phase B sign-off |
| **G4 — CI gate** | III. Quality Gates (NON-NEG.) | `_ci-core.yml` aggregate job reports `all_passed == 'true'` on the release commit | Block PyPI publish |
| **G5 — Ruff clean** | III. Quality Gates (NON-NEG.) | `ruff check src/ tests/` exits 0; `ruff format --check src/ tests/` exits 0 | Block merge to main |
| **G6 — Mypy strict** | III. Quality Gates (NON-NEG.) | `mypy --strict src/bin_packer_3d/` exits 0 with zero errors | Block Phase A sign-off |
| **G7 — Reproducibility** | IV. Reproducibility (NON-NEG.) | `uv pip sync requirements-dev.txt` installs without conflict on Python 3.11 and 3.14; `pytest` green on both | Block Phase C sign-off |
| **G8 — Dep minimalism** | V. Library Citizenship | `pip-audit` exits 0 (no known CVEs in runtime deps); no new runtime dep added without a spec FR citing it | Block PyPI publish |
| **G9 — Docstrings** | VI. Documentation (NON-NEG.) | Every public symbol in `__init__.py` surface has a docstring; `mkdocs build --strict` exits 0 | Block Phase C sign-off |
| **G10 — Privacy** | VII. Privacy | DATASETS per-file audit completed; no file containing PII or confidential business data committed; decision documented in `DATASETS/AUDIT.md` | Block Phase A sign-off |
| **G11 — Perf baseline** | VIII. Performance Discipline | Benchmark baseline captured and committed before any perf optimisation is attempted; provisional target validated by measurement | Block Phase C perf work |

---

## 5. Pending Architectural Decisions (→ research.md ADRs)

These decisions are LOCKED. The recommendations below are ratified. SpecKit
MUST record each as a resolved ADR in `specs/001-public-release-hardening/research.md`
with the status `ACCEPTED`. No further user input is required.

---

### ADR-001 — Third algorithm family

**Decision**: Implement **Extreme Point** heuristic as the third algorithm family.

**Rationale**: The spec (US2) explicitly names "at least one additional algorithm
family" and Extreme Point is the most widely cited in 3D-BPP literature (Crainic,
Perboli & Tadei 2008). It has a direct connection to the First Fit Decreasing
heuristic already implemented, making incremental implementation feasible. The
alternative (Guillotine cuts) would require a fundamentally different data model.

**Implementation scope**:
- New file: `src/bin_packer_3d/algorithms/extreme_point.py`
- Class: `ExtremePointPacker(BasePacker)`
- New `PackerConfig.strategy` value: `"extreme_point"` (Literal update — US1)
- Phase B deliverable.

---

### ADR-002 — Documentation tool

**Decision**: **MkDocs Material** with the `mkdocstrings[python]` plugin.

**Rationale**: Markdown-first authoring matches the existing README style.
`mkdocstrings` generates API reference directly from docstrings without a
separate build step. MkDocs Material has built-in search, dark/light mode, and
native GitHub Pages deployment. Alternative (Sphinx) requires RST or MyST
conversion overhead for a single-maintainer project.

**Implementation scope**:
- `mkdocs.yml` at repo root.
- `docs/` directory (§3 layout above).
- `mkdocs-material` and `mkdocstrings[python]` added to `[project.optional-dependencies]`
  section `docs` in `pyproject.toml` (not runtime deps — Principle V).
- `mkdocs build --strict` added as a CI job in `_ci-core.yml` (Gate G9).

---

### ADR-003 — CI platforms and workflow architecture

**Decision**: **Linux-only CI** for v1.0.0 (ubuntu-latest). macOS and Windows
are best-effort; the CI matrix does not include them for this release.

**Workflow architecture** (modelled on The-Embedinator's battle-tested pattern):

**Thin dispatcher + reusable core**:
- `ci.yml` is a minimal dispatcher. It triggers on `push` to `main` and
  `pull_request` (opened, synchronize, reopened, ready_for_review). It calls
  `_ci-core.yml` via `workflow_call`. All job logic lives in `_ci-core.yml`.
- Concurrency group: `ci-${{ github.ref }}` with `cancel-in-progress: true`
  to avoid wasted runner minutes on stale pushes.
- All action references pinned to full SHA with version comment
  (e.g., `actions/checkout@<sha> # v4.3.1`).

**`_ci-core.yml` jobs** (Linux matrix: 3.11 / 3.12 / 3.13 / 3.14):

| Job | Command | Blocking |
|---|---|---|
| `lint` | `ruff check src/ tests/ --output-format=github` | YES |
| `format-check` | `ruff format --check src/ tests/` | YES |
| `type-check` | `mypy --strict src/bin_packer_3d/` | YES |
| `test` (matrix 3.11–3.14) | `pytest tests/ --cov=src/bin_packer_3d --cov-report=xml --cov-fail-under=85` | YES |
| `pip-audit` | `pip-audit` against runtime deps | YES |
| `docs-build` | `mkdocs build --strict` | YES |
| `pre-commit-parity` | `pre-commit run --all-files` | YES |
| `aggregate` | collects `all_passed` output | — |

The `aggregate` job (runs with `if: always()`, depends on all above) outputs
`all_passed: true/false`. Release and publish workflows consume this output and
refuse to proceed unless `all_passed == 'true'`.

**Coverage reporting**: `codecov/codecov-action` uploads `coverage.xml` with
flag `library` after the `test` job completes. `codecov.yml` at repo root
configures the `library` flag with a 85 % threshold.

**PR title enforcement** (`pr-title.yml`):
- Uses `amannn/action-semantic-pull-request` (pinned SHA).
- Allowed types: `feat`, `fix`, `chore`, `docs`, `refactor`, `test`, `ci`,
  `build`, `perf`, `style`, `revert`.
- `requireScope: false`; `validateSingleCommit: false`.
- REQUIRED status check on `main`.

**Dependabot** (`dependabot.yml`):
- `pip` ecosystem: weekly, target `main`.
- `github-actions` ecosystem: weekly, target `main`.

---

### ADR-004 — Documentation host

**Decision**: **GitHub Pages** via MkDocs Material's built-in deploy action
(`mkdocs gh-deploy`).

**Rationale**: Zero additional infrastructure. The repo is already on GitHub.
Deployment is a single `mkdocs gh-deploy --force` step added to `release.yml`
after a successful PyPI publish. Alternative (Read the Docs) requires an external
account and webhook configuration — unnecessary overhead for a sole-maintainer
project.

**Implementation scope**:
- `release.yml` gains a `docs-deploy` job that runs `mkdocs gh-deploy --force`
  after the `publish` job succeeds.
- `gh-pages` branch auto-created by `mkdocs gh-deploy` on first run.

---

### ADR-005 — Performance budget methodology

**Decision**: **Measure-first** (Principle VIII mandate).

**Rationale**: Principle VIII of the constitution ("Performance Discipline")
explicitly prohibits setting perf budgets before measuring. The Phase B
deliverable is a benchmark baseline, not an optimisation. Targets are set
after baseline data is available.

**Implementation scope**:
- `benchmark/run_baseline.py` captures pack-time for a reference input
  (50 boxes, single bin) across all three algorithms.
- Results committed to `benchmark/baseline.json` in Phase B.
- Phase C perf work (if any) uses `baseline.json` as the lower bound.
- Gate G11 enforces this ordering — no perf optimisation without a baseline.

---

### ADR-006 — PyPI package name

**Decision**: **`bin-packer-3d`**. Reserve immediately.

**Rationale**: The name is descriptive, hyphenated (PyPI convention), and matches
the existing import name `bin_packer_3d`. Reservation is a zero-cost action —
publish a `0.1.0` placeholder to PyPI before any other implementation work
begins to claim the name.

**Fallback list** (in priority order, if `bin-packer-3d` is taken):
1. `bin-packer3d`
2. `binpacker3d`
3. `bin-packing-3d`

**Implementation scope**:
- `pyproject.toml` `[project]` name set to `bin-packer-3d`.
- OIDC trusted publisher configured on PyPI: owner `Bruno-Ghiberto`,
  repo `3D_BIN_PACKING`, workflow `release.yml`, environment `pypi`.
- `release.yml` uses `pypa/gh-action-pypi-publish` (pinned SHA) with no
  `password:` — authentication via OIDC token (`id-token: write` permission).

---

### ADR-007 — Constraint framework shape (US7)

**Decision**: **Pydantic model + visitor pattern**.

**Rationale**: `pydantic` is already a runtime dependency (`PackerConfig`). Adding
a `ConstraintVisitor` ABC lets users register custom constraints without
subclassing the packer, keeping the core algorithm implementations closed to
modification (Open/Closed). Alternative (plain dataclass + protocol) lacks
validation; alternative (full plugin system) is overengineered for v1.0.0.

**Implementation scope**:
- `src/bin_packer_3d/constraints/base.py`: `ConstraintVisitor(ABC)` with
  `visit_placement(placement: Placement, box: Box, bin: Bin) -> bool`.
- `src/bin_packer_3d/constraints/standard.py`: `WeightConstraint`,
  `RotationLockConstraint` as concrete visitors.
- `PackerConfig` gains a `constraints: list[ConstraintVisitor] = []` field.
- Phase B deliverable.

---

### ADR-008 — Observability stack (US6)

**Decision**: **stdlib `logging` with a structured adapter** — zero new runtime
dependencies.

**Rationale**: Principle V (Library Citizenship) prohibits introducing heavyweight
logging frameworks as runtime dependencies. `structlog` and `loguru` are popular
but would add mandatory deps for downstream users who may not want them.
A thin adapter on top of stdlib `logging` gives structured output via
`logging.LogRecord` extra fields without lock-in.

**Implementation scope**:
- `src/bin_packer_3d/observability.py`:
  - `get_logger(name: str) -> logging.Logger` — returns a logger namespaced
    under `bin_packer_3d.<name>`.
  - `StructuredAdapter(logging.LoggerAdapter)` — adds `extra` dict fields to
    log records for machine-readable output.
  - Emits `packing.start`, `packing.end`, `packing.algorithm_selected` events
    at `DEBUG` level; emits `packing.warning` at `WARNING` level for
    constraint violations.
- Phase B deliverable.

---

## 6. Phased Implementation Strategy

Three phases map to three semantic version milestones. Each phase ships a
coherent, shippable increment.

### Phase A — Foundations (`v0.2.0`)

**User stories covered**: US1 (API contract), US3 (developer experience),
US4 (privacy), US8 (release infrastructure — partial), US9 (documentation — partial).

**Deliverables**:
1. `pyproject.toml`: Python floor bumped `>=3.10 → >=3.11`; `[project]` name
   set to `bin-packer-3d`; `py.typed` added to `[tool.hatch.build.targets.sdist]`
   includes.
2. `src/bin_packer_3d/py.typed` marker created (PEP 561).
3. `PackerConfig.strategy` Literal updated: `"ffd" | "shelf" | "extreme_point"`
   (stub only — `extreme_point` raises `NotImplementedError` until Phase B).
4. `mypy --strict` passes with zero errors on `src/`.
5. `DATASETS/AUDIT.md` created with per-file privacy decision (see §11).
6. Legacy `CODE/` excluded from sdist via `MANIFEST.in`; deprecation comment
   added to `CODE/MAIN.py`.
7. `.pre-commit-config.yaml` updated: ruff, ruff-format, trailing-whitespace,
   end-of-file-fixer, check-yaml, check-toml.
8. `.editorconfig` created.
9. `CHANGELOG.md` created with `[0.2.0]` section.
10. GitHub Actions workflows: `ci.yml`, `_ci-core.yml` (without matrix — Phase A
    uses single Python version 3.11 to bootstrap), `pr-title.yml`, `dependabot.yml`.
    Matrix expansion (3.11–3.14) added in Phase B.
11. PyPI name `bin-packer-3d` reserved (placeholder publish of `0.1.0` or `0.2.0`).

**Gate checks before Phase A sign-off**: G1, G2, G5, G6, G10.

---

### Phase B — Capability Expansion (`v0.3.0`)

**User stories covered**: US2 (Extreme Point algorithm), US5 (benchmarking),
US6 (observability), US7 (constraints).

**Deliverables**:
1. `extreme_point.py` implemented and tested (property tests included).
2. `constraints/` package implemented (ADR-007).
3. `observability.py` implemented (ADR-008).
4. `benchmark/` package + `benchmark/run_baseline.py`; baseline captured and
   committed to `benchmark/baseline.json`.
5. CI matrix expanded to Python 3.11 / 3.12 / 3.13 / 3.14 in `_ci-core.yml`.
6. `security.yml` (CodeQL, Python-only, ADVISORY) added.
7. `codecov.yml` added; Codecov integration wired in `_ci-core.yml`.
8. Property-based tests (`tests/property/test_invariants.py`) added.
9. Coverage gate enforced: `--cov-fail-under=85`.
10. `CHANGELOG.md` updated with `[0.3.0]` section.

**Gate checks before Phase B sign-off**: G2, G3, G4, G5, G6, G7, G8, G11.

---

### Phase C — Public Release (`v1.0.0`)

**User stories covered**: US3 (developer experience — full), US8 (release),
US9 (documentation — full).

**Deliverables**:
1. `docs/` directory fully authored (index, quickstart, api, algorithms, cli,
   changelog).
2. `mkdocs.yml` configured with MkDocs Material theme + mkdocstrings plugin.
3. `mkdocs build --strict` passes (Gate G9).
4. `release.yml` completed: CI gate → PyPI publish (OIDC) → GitHub Pages deploy.
5. OIDC trusted publisher configured on PyPI dashboard.
6. GitHub repository configured: description, topics, homepage URL (GH Pages),
   issue templates, PR template, CODEOWNERS.
7. `README.md` overhauled: badges (CI status, PyPI version, coverage, Python
   versions), quickstart, algorithm comparison table, contribution guide pointer.
8. `CHANGELOG.md` `[1.0.0]` section completed.
9. Tag `v1.0.0` pushed → `release.yml` runs → PyPI publish + GH Pages deploy.

**Gate checks before Phase C sign-off**: G1, G3, G4, G7, G8, G9.

---

## 7. Data Model Outline

Fills the `data-model.md` artefact. Describes new and modified entities.
Existing entities (`Box`, `Bin`, `Placement`, `PackerConfig`) are carried
forward; only changes and new entities are described here.

### Modified entities

**`PackerConfig`** (`src/bin_packer_3d/config.py`):
- `strategy: Literal["ffd", "shelf", "extreme_point"]` — adds `"extreme_point"`.
- `constraints: list[ConstraintVisitor] = []` — NEW field (Phase B).

### New entities

**`AlgorithmMetadata`** (`src/bin_packer_3d/models/metadata.py`):
```python
@dataclass(frozen=True)
class AlgorithmMetadata:
    name: str                          # e.g. "extreme_point"
    version: str                       # semver of the package
    parameters: dict[str, Any]         # captured from PackerConfig at run time
    timestamp: datetime                # UTC, set at packing start
```

**`BenchmarkResult`** (`src/bin_packer_3d/benchmark/results.py`):
```python
@dataclass
class BenchmarkResult:
    algorithm: str                     # strategy name
    n_boxes: int
    n_bins_used: int
    volume_utilisation: float          # [0.0, 1.0]
    elapsed_seconds: float
    metadata: AlgorithmMetadata
    notes: str = ""
```
Serialised to JSON via `dataclasses.asdict`. No ORM, no schema registry.

**`ConstraintVisitor`** (`src/bin_packer_3d/constraints/base.py`):
```python
class ConstraintVisitor(ABC):
    @abstractmethod
    def visit_placement(
        self,
        placement: Placement,
        box: Box,
        bin: Bin,
    ) -> bool:
        """Return True if the placement satisfies this constraint."""
```

**`WeightConstraint`** (`src/bin_packer_3d/constraints/standard.py`):
```python
class WeightConstraint(ConstraintVisitor):
    max_weight: float                  # validated by Pydantic if wrapped

    def visit_placement(self, placement, box, bin) -> bool:
        current = sum(p.box.weight for p in bin.placements)
        return current + box.weight <= self.max_weight
```

**`RotationLockConstraint`** (`src/bin_packer_3d/constraints/standard.py`):
Rejects placements that apply a rotation not permitted by the box's
`allowed_rotations` attribute.

---

## 8. Quickstart Contract

Fills the `quickstart.md` artefact. The 5-command flow MUST work verbatim
after `pip install bin-packer-3d`. This is the primary success criterion for
US3 AC4.

```bash
# 1. Install
pip install bin-packer-3d

# 2. Verify CLI
bin-packer info

# 3. Initialise a sample config
bin-packer init --output packing_config.json

# 4. Pack a sample dataset
bin-packer pack --config packing_config.json --input sample_boxes.csv

# 5. Explore the 3D visualisation (opens browser)
bin-packer pack --config packing_config.json --input sample_boxes.csv --visualise
```

Each command must:
- Exit 0 on a clean install (no missing deps, no import errors).
- Produce human-readable output or a browser visualisation within < 5 seconds
  on a commodity laptop (no GPU required).
- Print a `--help` message that includes the command's purpose, required flags,
  and at least one example.

The `quickstart.md` doc must include the five commands above, annotated with
expected output snippets (not screenshot images — text only for reproducibility).

---

## 9. Public Contracts Surface

Fills the `contracts/` directory in `specs/001-public-release-hardening/`.
Enumerates everything that constitutes a public, versioned contract that users
may depend on and that a breaking change policy applies to.

### Python API (`contracts/api.md`)

All names exported from `src/bin_packer_3d/__init__.py`:

| Symbol | Kind | Status |
|---|---|---|
| `BinPacker` | class | existing — stabilise |
| `Box` | dataclass | existing — stabilise |
| `Bin` | dataclass | existing — stabilise |
| `Placement` | dataclass | existing — stabilise |
| `PackerConfig` | pydantic model | existing — stabilise; add `constraints` field |
| `pack()` | function | existing — stabilise |
| `load_boxes_from_csv()` | function | existing — stabilise |
| `load_boxes_from_excel()` | function | existing — stabilise |
| `plot_packing()` | function | existing — stabilise |
| `ConstraintVisitor` | ABC | NEW Phase B |
| `WeightConstraint` | class | NEW Phase B |
| `RotationLockConstraint` | class | NEW Phase B |
| `AlgorithmMetadata` | dataclass | NEW Phase B |
| `BenchmarkResult` | dataclass | NEW Phase B |
| `get_logger()` | function | NEW Phase B |

Any symbol NOT listed here is considered internal and may change without a
version bump.

### CLI (`contracts/cli.md`)

Entry point: `bin-packer` (installed by `[project.scripts]` in pyproject.toml).

| Subcommand | Flags | Status |
|---|---|---|
| `bin-packer pack` | `--config`, `--input`, `--visualise`, `--algorithm` | existing — stabilise |
| `bin-packer info` | none | existing — stabilise |
| `bin-packer init` | `--output` | existing — stabilise |

Exit codes: 0 success, 1 user error (bad input), 2 internal error.

### Configuration schema (`contracts/config-schema.md`)

`PackerConfig` fields constitute a versioned schema. Any field rename or type
change is a breaking change requiring a major version bump.

| Field | Type | Default | Since |
|---|---|---|---|
| `strategy` | `Literal["ffd","shelf","extreme_point"]` | `"ffd"` | 0.2.0 (Literal fix) |
| `bin_dimensions` | `tuple[float, float, float]` | required | 0.1.0 |
| `allow_rotation` | `bool` | `True` | 0.1.0 |
| `constraints` | `list[ConstraintVisitor]` | `[]` | 0.3.0 |

---

## 10. Testing Strategy

Fills the testing section of `plan.md`. Principle II (Test-First) and Gate G2
require tests to exist before implementation commits.

### Test pyramid

```
                ┌──────────────────┐
                │   Property       │  hypothesis — invariants
                │   (10 %)         │  (new — Phase B)
                ├──────────────────┤
                │   Integration    │  pytest — workflow + overlap + bounds
                │   (20 %)         │  (existing + benchmark tests)
                ├──────────────────┤
                │   Unit           │  pytest — models, algorithms, constraints,
                │   (70 %)         │  observability (existing + new)
                └──────────────────┘
```

### Coverage thresholds (Gate G3)

- Overall: ≥ 85 % line coverage on `src/bin_packer_3d/`
- No single module below 70 % (enforced via `--cov-fail-under=85` and
  `coverage.xml` post-processing in CI)
- New modules (Phase B): must reach ≥ 85 % before Phase B sign-off

### Property-based tests (`tests/property/test_invariants.py`)

Using `hypothesis`. Invariants to verify on every generated packing:
1. **Non-overlap**: no two placed boxes share any interior volume.
2. **Bounds containment**: every placed box fits strictly within bin dimensions.
3. **Volume conservation**: sum of placed box volumes ≤ bin volume.
4. **Determinism**: same input + same `PackerConfig` always produces the same
   placement list (no non-deterministic side effects).

### Test markers

```
pytest -m unit          # fast, no I/O
pytest -m integration   # may write files
pytest -m property      # hypothesis — may be slow
pytest -m slow          # explicit opt-in (benchmarks, large datasets)
```

CI runs `pytest -m "not slow"` by default. The `slow` suite runs locally before
Phase C tagging.

### Pre-commit parity

`_ci-core.yml` includes a `pre-commit-parity` job that runs
`pre-commit run --all-files` against the exact commit SHA. This ensures the
local hook configuration matches CI — no "works on my machine" escapes.

---

## 11. Operational Plans

### 11.1 Privacy Plan (US4 — DATASETS audit)

**Context**: The clarification session resolved that DATASETS requires a per-file
audit. Each file is inspected; any file containing PII or confidential business
data is scrubbed (anonymised) or excluded from the public repository.

**Decision tree** (apply to each file in `DATASETS/`):

```
For each file F in DATASETS/:
  1. Open F and inspect column headers and sample rows.
  2. Does F contain any of:
       - Personal names, email addresses, phone numbers, or national IDs?
       - Internal company names, product codes, or pricing data not
         intended for public disclosure?
     → YES: either anonymise (replace with synthetic data) and keep,
             or remove from the repo and add to .gitignore.
     → NO: keep as-is; document decision in DATASETS/AUDIT.md.
  3. Record in DATASETS/AUDIT.md:
       - File name
       - Decision (keep / anonymise / remove)
       - Reason (public domain / synthetic / confidential)
       - Reviewer: Bruno Ghiberto
       - Date of review
```

**Files known at spec time** (from §2 of the spec):
- `sample_boxes.csv` — likely synthetic; expected decision: keep.
- Three `.xlsx` files — content unknown; must be inspected individually.

**Gate G10** blocks Phase A sign-off until `DATASETS/AUDIT.md` is committed
and every file decision is documented.

---

### 11.2 Release Plan (US8)

**Version milestones**:

| Version | Phase | Trigger | Contents |
|---|---|---|---|
| `0.2.0` | A | manual `git tag v0.2.0` | Foundations (API contract, py.typed, CI bootstrap) |
| `0.3.0` | B | manual `git tag v0.3.0` | Capability expansion (EP algorithm, constraints, observability, benchmarks) |
| `1.0.0` | C | manual `git tag v1.0.0` | Public release (docs, full CI matrix, PyPI + GH Pages) |

**PyPI publish workflow** (`release.yml`):

```
Trigger: push of tag v*
  │
  ├─ Job: ci (calls _ci-core.yml)
  │     └─ Output: all_passed
  │
  ├─ Job: publish (needs: ci, if: all_passed == 'true')
  │     ├─ actions/checkout (pinned SHA)
  │     ├─ actions/setup-python python-version: "3.11"
  │     ├─ pip install build
  │     ├─ python -m build
  │     └─ pypa/gh-action-pypi-publish (pinned SHA)
  │           permissions: id-token: write   # OIDC trusted publisher
  │
  └─ Job: docs-deploy (needs: publish)
        ├─ pip install mkdocs-material mkdocstrings[python]
        └─ mkdocs gh-deploy --force
```

**OIDC trusted publisher configuration** (done once on PyPI dashboard before
first publish):
- Publisher: GitHub Actions
- Owner: `Bruno-Ghiberto`
- Repository: `3D_BIN_PACKING`
- Workflow: `release.yml`
- Environment: `pypi`

No `PYPI_API_TOKEN` secret is needed. The `id-token: write` permission on the
`publish` job is sufficient.

**Fallback**: if `bin-packer-3d` is taken on PyPI at reservation time, use the
first available name from the fallback list in ADR-006. Update `pyproject.toml`
`[project]` name and all documentation references before Phase A sign-off.

---

## 12. Risks, Non-Goals & Open Questions

### Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `bin-packer-3d` PyPI name already taken | Low | Medium | Reserve immediately (Phase A, first action); fallback list in ADR-006 |
| `mypy --strict` reveals deep type errors in existing code | Medium | Medium | Fix incrementally in Phase A; `# type: ignore` with TODO comment as temporary escape hatch, tracked in CHANGELOG |
| Extreme Point algorithm correctness harder than expected | Low | High | Property tests (invariants) catch placement errors; fall back to published pseudocode from Crainic 2008 |
| DATASETS files contain confidential data | Unknown until audit | High | Gate G10 blocks Phase A until audit is complete |
| Python 3.14 compatibility issues (new stdlib changes) | Low | Low | CI matrix catches it; fix before tagging |
| Codecov token required despite OIDC | Low | Low | `codecov/codecov-action` supports tokenless uploads for public repos |

### Non-Goals (explicitly out of scope for this change)

- Windows or macOS CI matrix entries.
- GPU-accelerated packing.
- A web UI or REST API.
- Solving the 3D-BPP to optimality (this is a heuristic library).
- Support for non-rectangular bin or box shapes.
- Multi-bin heterogeneous packing (different bin sizes in one run).
- A plugin ecosystem or dynamic algorithm registration beyond Phase B constraints.
- Rust extension modules (deferred to a future spec per Principle VIII footnote).

### Open Questions (resolved — documented for audit trail)

All spec-level ambiguities were resolved in the `/speckit.clarify` session:

1. **Privacy disposition of DATASETS files** → per-file audit (§11.1 above).
2. **Python floor** → `>=3.11` public; maintainer on 3.14.3; CI matrix 3.11–3.14.
3. **Rust extension modules** → deferred; Principle VIII footnote in constitution.

No open questions remain at plan authoring time.
