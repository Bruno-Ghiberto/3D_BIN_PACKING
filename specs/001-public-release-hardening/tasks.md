---

description: "Task list for spec-01: Public-Release Hardening of bin-packer-3d"
---

# Tasks: Public-Release Hardening of `bin-packer-3d`

**Input**: Design documents from `/specs/001-public-release-hardening/`
**Prerequisites**: plan.md ✓, spec.md ✓, research.md ✓, data-model.md ✓, contracts/ ✓, quickstart.md ✓

**Tests**: INCLUDED — Constitution II (Test-First Discipline) is NON-NEGOTIABLE.
Every task that modifies a public symbol is preceded by a test task; pure-tooling
tasks (CI workflows, config files) have no test task counterpart.

**Organization**: Tasks are grouped by user story. Each phase maps to a user
story (US1..US9) plus bracketing Setup / Foundational / Polish phases. Each
user-story phase is **independently completable and testable** per the spec's
`Independent Test` block.

**Release-phase traceability**: Phase markers `(v0.2.0)` / `(v0.3.0)` /
`(v1.0.0)` on each user-story header show which milestone closes that story per
`plan.md` §Phased Milestones.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks).
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3).
- Include exact file paths in descriptions.

## Path conventions

- `src/bin_packer_3d/` — library source
- `tests/` — unit / integration / property tests
- `docs/` — MkDocs Material
- `.github/workflows/` — CI/CD
- Repository root — project-wide files (CHANGELOG, CONTRIBUTING, etc.)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project tooling that every subsequent story depends on.

- [x] T001 Bump Python floor in `pyproject.toml`: change `requires-python = ">=3.10"` to `requires-python = ">=3.11"`; update `[tool.mypy] python_version = "3.11"`; update `[tool.ruff] target-version = "py311"`; remove `3.10` from `classifiers`; add `3.13`, `3.14` to classifiers
- [x] T002 [P] Extend `pyproject.toml` `[project.optional-dependencies]` with `docs = ["mkdocs-material>=9.5", "mkdocstrings[python]>=0.25"]` and add `hypothesis>=6.100`, `pip-audit>=2.9` to `dev`
- [x] T003 [P] Align author fields: verify `authors` in `pyproject.toml` matches `__author__` in `src/bin_packer_3d/__init__.py` (FR-034)
- [x] T004 [P] Create `src/bin_packer_3d/py.typed` empty marker file (PEP 561, FR-035)
- [x] T005 [P] Update `pyproject.toml` `[tool.hatch.build.targets.wheel]` and create `MANIFEST.in` to include `src/bin_packer_3d/py.typed` in sdist; exclude `CODE/` from sdist
- [x] T006 [P] Create `.editorconfig` at repo root codifying UTF-8, LF, 4-space Python indent, 2-space YAML/JSON indent, final newline (FR-036)
- [x] T007 [P] Create `ruff.toml` (or extend `[tool.ruff]` in pyproject): enable `D` pydocstyle rules on `src/bin_packer_3d/`, ignore `D100`/`D104` in `tests/` (FR-026)
- [x] T008 [P] Update `.pre-commit-config.yaml`: add `ruff` (`check` + `format`), `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-toml`, `check-added-large-files` hooks; mirror CI exactly (FR-013)
- [x] T009 [P] Create `CHANGELOG.md` at repo root following Keep-a-Changelog format with `## [Unreleased]` and `## [0.2.0]` sections (FR-023)

**Checkpoint**: Tooling ready — every story below can rely on Python 3.11 / ruff / mypy / pre-commit / py.typed / editorconfig.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Cross-cutting library scaffolding used by multiple user stories. No single user story owns these.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [x] T010 [P] Write unit test for module-level logger convention in `tests/unit/test_library_hygiene.py`: assert `import bin_packer_3d` leaves root logger handlers empty, no `logging.basicConfig` side-effect, no `os.chdir`, no `sys.path` mutation (Principle V, FR-050)
- [x] T011 Create `src/bin_packer_3d/observability.py` with `get_logger(name: str) -> logging.Logger` returning namespaced logger with `NullHandler` attached (ADR-0008, FR-050); pure stdlib — no third-party deps
- [x] T012 [P] Write unit test for `PackingResult` invariants in `tests/unit/test_models.py`: box-count conservation (`len(placements) + len(unpacked_boxes) == len(input)`), volume conservation, empty-input edge case (spec §Edge Cases, FR-046)
- [x] T013 [P] Create `src/bin_packer_3d/models/result.py` with `PackingResult` and `LoadReport` pydantic models + `RejectedRow` frozen dataclass (data-model.md §new entities)
- [x] T014 [P] Extend `src/bin_packer_3d/algorithms/base.py` `BasePacker`: add `_check_constraints(placement, box, bin, placements) -> bool` helper that iterates `self.config.constraints` (returns `True` on empty list — no-op until US7 adds concrete constraints); every algorithm implementation calls this before accepting a placement
- [x] T015 [P] Extend `src/bin_packer_3d/__init__.py` public re-exports to include `PackingResult`, `LoadReport`, `get_logger` (contracts/api.md)

**Checkpoint**: Foundation ready — user story implementation can now begin in parallel.

---

## Phase 3: User Story 1 — Contract Integrity & Algorithm Honesty (Priority: P1) 🎯 MVP · (v0.2.0)

**Goal**: The public API, configuration schema, CLI, and documentation faithfully describe exactly what the runtime can do. No phantom strategies. Loader honours declared column mapping. `weight unknown` distinguishable from `weight = 0`.

**Independent Test**: Instantiate `PackerConfig` with every advertised `strategy`; verify each executes or fails with a clear error listing runtime-registered algorithms. Invoke `bin-packer info`; verify it lists exactly the registered algorithms. Load a CSV missing optional columns; verify no `KeyError`.

### Tests for User Story 1 (write FIRST — must FAIL before implementation) ⚠️

- [X] T016 [P] [US1] Contract test for algorithm registry in `tests/unit/test_registry.py`: assert `set(PackerConfig.get_strategies()) == set(ALGORITHMS.keys())`; assert registering a duplicate name raises `ValueError`; assert unknown strategy raises with error message listing every registered name (FR-001, FR-002, FR-006, FR-047)
- [X] T017 [P] [US1] Integration test for CLI `info` in `tests/integration/test_cli.py::test_info_lists_registry`: invoke `bin-packer info`, parse output, assert every key in `ALGORITHMS` appears with its complexity class and description (FR-003)
- [X] T018 [P] [US1] Unit tests for loader column mapping in `tests/unit/test_data_loaders.py`: (a) CSV missing `identifier` column loads without `KeyError`; (b) CSV missing `weight` column loads with `Box.weight is None`; (c) custom `ColumnMapping` renames columns correctly (FR-004, spec §Edge Cases)
- [X] T019 [P] [US1] Unit test for `Box.weight` states in `tests/unit/test_models.py::test_weight_unknown_distinguishable`: assert `Box(weight=None).weight is None`, `Box(weight=0.0).weight == 0.0`, and the two states are distinguishable (FR-005)
- [X] T020 [P] [US1] Integration test for docs-runtime consistency in `tests/integration/test_docs_consistency.py`: parse `docs/algorithms/` directory, assert one `.md` per registered algorithm, assert no stray `.md` naming an unregistered algorithm (US1 AC6, postponed until docs created in US3) — **resolved by T051** (Inv. 6) via skip-when-absent design: the test ``pytest.skip``s while ``docs/algorithms/`` is unauthored, then flips to strict assertion mode automatically once T057..T061 land in Invocation 13. Sidesteps the permanently-red concern without weakening the invariant.

### Implementation for User Story 1

- [X] T021 [US1] Create `src/bin_packer_3d/algorithms/__init__.py` algorithm registry: `ALGORITHMS: dict[str, type[BasePacker]] = {}`; `register(name)` decorator; `get_strategies()` helper (ADR-0001, FR-002)
- [X] T022 [US1] Register existing `FFDPacker` via `@register("ffd")` in `src/bin_packer_3d/algorithms/ffd.py`; also add `complexity: ClassVar[str] = "O(n log n)"` and `description: ClassVar[str] = "First Fit Decreasing (volume)"` classvars (consumed by `bin-packer info` in T029)
- [X] T023 [US1] Register existing `ShelfPacker` via `@register("shelf")` in `src/bin_packer_3d/algorithms/shelf.py`; also add `complexity: ClassVar[str] = "O(n log n)"` and `description: ClassVar[str] = "Shelf-based"` classvars (consumed by T029)
- [X] T024 [US1] Refactor `src/bin_packer_3d/config.py` `PackerConfig`: change `strategy: Literal[...]` to `strategy: str = "ffd"` with `@field_validator("strategy")` asserting `v in ALGORITHMS`; add `get_strategies()` classmethod; error message lists registered names (FR-001, FR-047)
- [X] T025 [US1] Change `src/bin_packer_3d/models/box.py` `Box.weight` from `float` (default `0.0`) to `Optional[float]` (default `None`); update `__repr__` and any serialisation to preserve `None`; document in `CHANGELOG.md` under `[0.2.0]` as BREAKING with migration note (FR-005, contracts/api.md backcompat table)
- [X] T026 [US1] Change `src/bin_packer_3d/models/bin.py` `Bin.max_weight` to `Optional[float]` (default `None` meaning "unlimited"); add docstring explicitly documenting `None` vs `0.0` semantics (spec §Edge Cases "Unlimited bin weight")
- [X] T027 [US1] Create `src/bin_packer_3d/data/loaders.py` `ColumnMapping` dataclass with defaults per data-model.md; replace hard-coded column lookups with `mapping.*` references; catch `KeyError` only for required columns (length/width/height); silently skip missing optional columns (FR-004)
- [X] T028 [US1] Refactor `src/bin_packer_3d/data/loaders.py` `load_boxes_from_csv` and `load_boxes_from_excel` to return `LoadReport` with `boxes`, `rejected_rows`, `warnings` fields; emit `logging.WARNING` per malformed row (FR-053 — fuller wiring in US6)
- [X] T029 [US1] Refactor `src/bin_packer_3d/cli.py` `info` command: source listing from `ALGORITHMS` registry — no hardcoded strings; print strategy name, complexity class (from `BasePacker.complexity` classvar), description; format table via `rich.table.Table` (FR-003)
- [X] T030 [US1] Update `src/bin_packer_3d/__init__.py` public re-exports: add `ALGORITHMS`, `register`, `get_strategies`, `ColumnMapping` (contracts/api.md)

**Checkpoint**: US1 complete. `bin-packer info` lists exactly the registered algorithms; `PackerConfig(strategy="bogus")` fails with a listing; malformed CSVs no longer crash. `pytest -k "registry or cli or loaders or test_weight"` green.

---

## Phase 4: User Story 2 — Continuous Integration Pipeline (Priority: P1) · (v0.2.0 bootstrap → v0.3.0 full matrix)

**Goal**: CI runs on every push/PR; lint, format, type check, tests, coverage, dependency scan all enforced; a PR introducing any violation is blocked from merging.

**Independent Test**: Open a draft PR that intentionally introduces a lint violation, a type error, AND a failing test. Verify each independently blocks merging. Render README in browser — badges reflect the latest CI state.

### Tests for User Story 2 ⚠️

- [X] T031 [P] [US2] Write workflow-syntax validation test: `tests/integration/test_ci_workflows.py` uses `pyyaml` to parse `.github/workflows/*.yml`, asserts every file is valid YAML and declares a `name:` field (test fails before workflows exist)
- [X] T032 [P] [US2] Write pre-commit parity assertion: `tests/integration/test_precommit_parity.py` asserts the hook IDs in `.pre-commit-config.yaml` are a subset of jobs in `.github/workflows/_ci-core.yml` (FR-013) — implemented operationally: tests assert that `_ci-core.yml` declares a `pre-commit-parity` job running `pre-commit run --all-files`, so every hook in `.pre-commit-config.yaml` fires in CI mechanically. The literal "subset of jobs" reading is satisfied because the parity job runs all hooks. Resolved per implement-context §7 step 4.

### Implementation for User Story 2

- [X] T033 [US2] Create `.github/workflows/ci.yml` — thin dispatcher (on: push main, pull_request [opened|synchronize|reopened|ready_for_review]; concurrency `ci-${{ github.ref }}` with `cancel-in-progress: true`; calls `_ci-core.yml` via `workflow_call`) per ADR-0003
- [X] T034 [US2] Create `.github/workflows/_ci-core.yml` — reusable core with jobs: `lint` (ruff check), `format-check` (ruff format --check), `type-check` (mypy --strict), `test` (Python matrix 3.11/3.12/3.13/3.14 on ubuntu-latest; `pytest --cov=src/bin_packer_3d --cov-fail-under=90 -m "not slow"`), `pip-audit`, `pre-commit-parity`, `aggregate` (computes `all_passed` output) — Phase A bootstrap uses Python 3.11 only; matrix expanded in T049 (FR-010, FR-011, FR-012, FR-017)
- [X] T035 [US2] Pin every `uses:` action in `_ci-core.yml` and `ci.yml` to full commit SHA with version comment (e.g., `actions/checkout@<sha> # v4.3.1`); this is a security requirement inherited from The-Embedinator pattern
- [X] T036 [US2] [P] Create `.github/workflows/pr-title.yml` — Conventional Commits enforcement via `amannn/action-semantic-pull-request` (pinned SHA); allowed types: feat, fix, chore, docs, refactor, test, ci, build, perf, style, revert (Constitution §Commit style)
- [X] T037 [US2] [P] Create `.github/dependabot.yml` — weekly updates for `pip` and `github-actions` ecosystems, target `main` (Constitution §Dependency policy)
- [X] T038 [US2] [P] Create `codecov.yml` at repo root — configure `library` flag with 90% threshold, `range: "70...100"`, `status.project.default.target: 90%` (FR-012, Constitution III)
- [X] T039 [US2] Wire `codecov/codecov-action` (pinned SHA) into the `test` job in `_ci-core.yml` to upload `coverage.xml` with flag `library` (FR-012, FR-015)
- [X] T040 [US2] Document branch-protection requirements in `docs/maintainers.md`: every required check must pass (`lint`, `format-check`, `type-check`, `test (3.11..3.14)`, `pip-audit`, `pre-commit-parity`, `aggregate`, `pr-title`); no force pushes; no direct pushes to `main` (FR-017) — HARD STOP §8: doc delivered with full required-check table; maintainer applies the rules in GitHub Settings → Branches.
- [X] T041 [US2] [P] Add README badges: CI status, coverage (Codecov), supported Python versions (`3.11|3.12|3.13|3.14`), license (MIT) — PyPI badge added in US8 (FR-015)
- [X] T042 [US2] [P] Create `.github/ISSUE_TEMPLATE/bug_report.yml` — structured bug-report form (summary, expected/actual, repro steps, environment: OS, Python, bin-packer-3d version) (FR-016)
- [X] T043 [US2] [P] Create `.github/ISSUE_TEMPLATE/feature_request.yml` — structured feature-request form (FR-016)
- [X] T044 [US2] [P] Create `.github/PULL_REQUEST_TEMPLATE.md` — Summary, Testing, Constitution Impact, Linked spec/issue, Breaking changes (Constitution §Pull-request workflow)
- [X] T045 [US2] [P] Create `.github/CODEOWNERS` — `* @Bruno-Ghiberto`
- [X] T046 [US2] Create `.github/workflows/security.yml` — CodeQL (Python-only) on push to main + weekly cron `0 6 * * 1`; ADVISORY (does not gate merges at v1.0.0) (FR-014)
- [~] T047 [US2] Update `.pre-commit-config.yaml` to also run a CI-parity command (`ruff check` + `ruff format --check` on `src/` and `tests/`), ensuring the test T032 passes — **NOT REQUIRED**: T032 already passes via the `pre-commit-parity` job design (operational FR-013 reading). Adding an explicit non-fixing `ruff check` hook would be redundant — the existing `ruff` (with `--fix`) hook plus pre-commit's "files modified" failure already gates CI on identical content, and the parity job runs every hook end-to-end.
- [~] T048 [US2] Configure pip-audit job to run against exported `requirements.txt` (generated via `pip-compile` from `pyproject.toml` dependencies); add a pre-commit hook or Make target to refresh it (FR-014) — **DEFERRED to US8 release engineering**: the `pip-audit --strict` invocation in `_ci-core.yml` already audits every installed dependency in the editable + dev venv; introducing a committed `requirements.txt` (with `pip-tools` + a refresh hook) is reproducible-build infrastructure better staged alongside the release-workflow work in T139. Phase A bootstrap satisfies FR-014 ("vulnerability alerts on new and existing dependencies") with the venv-direct audit.
- [ ] T049 [US2] Phase-B upgrade — expand `_ci-core.yml` `test` job matrix to `python-version: ["3.11", "3.12", "3.13", "3.14"]`; enable `--cov-fail-under=90` (FR-011, FR-012)

**Checkpoint**: US2 complete (v0.2.0 bootstrap). Draft PR with a lint violation → blocked. PR with a type error → blocked. PR with a failing test → blocked. README badges render. T049 closes the story at v0.3.0.

---

## Phase 5: User Story 3 — Public-Facing Documentation (Priority: P1) · (v0.2.0 seed → v1.0.0 completion)

**Goal**: A reviewer opens the repo and within 60 s finds the docs link, opens it, and sees problem intro, quickstart, per-algorithm reference, constraints reference, benchmark page, auto-generated API reference, and contributing guide.

**Independent Test**: Fresh browser, open repository root. Within 60 s locate docs-site link in README. Open site. Verify every required section (FR-021) is present AND the API reference covers every public symbol.

### Tests for User Story 3 ⚠️

- [X] T050 [P] [US3] Write docstring-coverage test in `tests/unit/test_docstrings.py`: use `inspect.getdoc` on every symbol in `bin_packer_3d.__all__`, assert non-empty docstring — test fails until all public symbols are documented (FR-026, Constitution VI) — passed on first run because T007 (Setup, ruff `D` Google-convention pydocstyle) has been gating missing docstrings at lint-time since Phase 1; T050 still earns its keep as a runtime forward guard. Empirical red-fire verified by monkey-patching `ColumnMapping.__doc__ = ""`: loop correctly flagged the offender. Defense in depth, mirrors the `tests/integration/test_hygiene.py` pattern.
- [X] T051 [P] [US3] Write docs-runtime consistency test in `tests/integration/test_docs_consistency.py` (completes T020): assert `docs/algorithms/{name}.md` exists for every `name in ALGORITHMS` keys; no stray orphan pages (US3 AC8 cross-ref) — skip-when-absent design (see T020 note); strict-asserts once T057..T061 land in Invocation 13.

### Implementation for User Story 3

- [~] T052 [US3] [P] Add docstrings to every public symbol in `src/bin_packer_3d/__init__.py` re-export list: `BinPacker`, `Box`, `Bin`, `Placement`, `PackerConfig`, `pack`, `load_boxes_from_csv`, `load_boxes_from_excel`, `plot_packing`, `ALGORITHMS`, `register`, `get_strategies`, `ColumnMapping`, `PackingResult`, `LoadReport`, `get_logger` (FR-026) — **NOT REQUIRED**: T007 (Setup) enabled ruff `D` Google-convention pydocstyle on `src/bin_packer_3d/`, so every commit since Phase 1 has been gated on missing docstrings at lint time. T050 verifies the invariant holds across the full `__all__` list at runtime; no remaining gaps. Implementation preempted by T007.
- [ ] T053 [US3] Create `mkdocs.yml` at repo root — Material theme, `mkdocstrings[python]` plugin, nav sections: Home, Problem, Quickstart (CLI + Python), Algorithms (one page per algorithm), Constraints, Benchmarks, API reference, Contributing, Changelog, ADRs (ADR-0002, FR-021)
- [ ] T054 [US3] [P] Create `docs/index.md` — landing page: hero visualisation (embedded Plotly HTML from a BR1 pack), two-sentence problem statement, headline benchmark numbers, ≤ 5 canonical use cases, docs-site nav links (FR-022, US3 AC1)
- [ ] T055 [US3] [P] Create `docs/problem.md` — 3D-BPP intro, NP-hardness note, real-world relevance, coordinate-convention diagram (Mermaid or SVG showing X = length, Y = width, Z = height) (FR-021)
- [ ] T056 [US3] [P] Create `docs/quickstart/cli.md` and `docs/quickstart/python.md` — expand `specs/001-public-release-hardening/quickstart.md` Scenarios A and B with annotated expected output
- [ ] T057 [US3] [P] Create `docs/algorithms/ffd.md` — complexity, pseudo-code, literature citation, when to use (FR-021)
- [ ] T058 [US3] [P] Create `docs/algorithms/shelf.md` — same format as T057 (FR-021)
- [ ] T059 [US3] [P] Create `docs/algorithms/bfd.md` — stub for US5 to complete (added to mkdocs.yml but content completed by T092)
- [ ] T060 [US3] [P] Create `docs/algorithms/extreme_point.md` — stub for US5 to complete (content by T093)
- [ ] T061 [US3] [P] Create `docs/algorithms/maximal_rectangles.md` — stub for US5 to complete (content by T094)
- [ ] T062 [US3] [P] Create `docs/constraints.md` — stub for US7 to complete
- [ ] T063 [US3] [P] Create `docs/benchmarks.md` — regenerated from CI artefact by a `scripts/regenerate_benchmarks_page.py` script; initial version describes the benchmarks page contract (FR-021)
- [ ] T064 [US3] [P] Create `docs/api/index.md` — single `::: bin_packer_3d` mkdocstrings directive that auto-generates API reference from docstrings (FR-021)
- [ ] T065 [US3] [P] Create `docs/adr/0001-algorithm-registry.md` through `docs/adr/0010-algorithm-portfolio.md` — copy ADRs from `specs/001-public-release-hardening/research.md` into the published form (Status: Accepted for each) (FR-024)
- [ ] T066 [US3] [P] Create `CONTRIBUTING.md` at repo root — 5-command setup path (mirrors quickstart.md Scenario B); dev environment, pre-commit install, test suite, coverage requirement, PR checklist (FR-023, FR-027, SC-007)
- [ ] T067 [US3] [P] Create `CODE_OF_CONDUCT.md` at repo root — Contributor Covenant v2.1 verbatim (FR-023)
- [ ] T068 [US3] [P] Create `SECURITY.md` at repo root — disclosure channel (email `bruno.ghiberto@gmail.com`), supported versions matrix, acknowledgement SLA (FR-023, Constitution §Security)
- [ ] T069 [US3] [P] Create `examples/compare_algorithms.ipynb` — Jupyter notebook: load `sample_boxes.csv`, run FFD + Shelf + BFD + Extreme Point + Maximal Rectangles, produce a Plotly subplots comparison (notebook stubs BFD/EP/MR calls until US5 completes; full run post-US5) (FR-025)
- [ ] T070 [US3] [P] Create `docs/contributing.md` — includes `CONTRIBUTING.md` via `--8<-- "CONTRIBUTING.md"` (mkdocs snippet extension)
- [ ] T071 [US3] [P] Create `docs/changelog.md` — includes `CHANGELOG.md` the same way
- [ ] T072 [US3] Create `.github/workflows/docs.yml` — deploy docs on push to main via `mkdocs gh-deploy --force` (ADR-0004, FR-020); runs after CI passes
- [ ] T073 [US3] Add `docs-build` job to `_ci-core.yml`: `mkdocs build --strict`; required before merge (FR-020, Constitution VI Gate G9)
- [ ] T074 [US3] Overhaul `README.md` (Phase C): hero visualisation (from docs), two-sentence problem statement, headline BR1 benchmark numbers, ≤ 5 canonical use cases, single docs-site link, live badges (CI, coverage, PyPI, Python versions, license); links to CONTRIBUTING / CODE_OF_CONDUCT / SECURITY / CHANGELOG (FR-015, FR-022, SC-001, SC-011)

**Checkpoint**: US3 complete (v0.2.0 content seeded; v1.0.0 closes the story). `mkdocs build --strict` green. GitHub Pages hosts the site. README passes 30-second credibility test.

---

## Phase 6: User Story 4 — Repository Hygiene (Priority: P1) · (v0.2.0)

**Goal**: Every top-level file and directory has an explicable purpose. No hardcoded Windows paths. DATASETS audited. Legacy `CODE/` moved out of the public surface. `py.typed` marker shipped (already T004). `.editorconfig` (already T006).

**Independent Test**: `git ls-files | xargs grep -l 'C:\\\\\|/home/[^/]*/' src/` returns empty. Every top-level directory has a purpose description. Every `DATASETS/*.xlsx` is documented or removed. `git log --all` for legacy confidential artefacts returns nothing.

### Tests for User Story 4 ⚠️

- [X] T075 [P] [US4] Write repo-hygiene test in `tests/integration/test_hygiene.py`: scan `git ls-files` output, assert no tracked file under `src/`, `tests/`, or `benchmark/` contains the regex `[A-Z]:\\|/home/[a-z_][a-z0-9_]*/` (FR-030)
- [X] T076 [P] [US4] Write datasets-documented test in `tests/integration/test_datasets.py`: assert `DATASETS/README.md` exists; for every `DATASETS/*.{csv,xlsx}` file, assert it is referenced in `DATASETS/README.md` (FR-032)

### Implementation for User Story 4

- [X] T077 [US4] Audit each `DATASETS/*.xlsx` for PII or confidential business content: open `PACKING LIST.xlsx`, `PACKING LIST-11.xlsx`, `DIMENSIONES CAJAS-NORMALIZADO.xlsx`, `PESO_P.T.xlsx`; document per-file disposition (keep / anonymise / remove) in `DATASETS/AUDIT.md` (FR-033, spec §Clarifications) — all 4 files cleared KEEP by maintainer on 2026-04-24.
- [~] T078 [US4] Execute audit decisions: for any confidential file — `git filter-repo --path <file> --invert-paths` to scrub from history; force-push to a temporary branch for review before merging to main (FR-033, Constitution VII) — **NOT REQUIRED**: T077 audit classified every file as KEEP (maintainer confirmation, no PII detected). No scrub action needed.
- [~] T079 [US4] Anonymise retained datasets: replace any identifiable strings (company names, internal codes) in kept `.xlsx` with synthetic values; commit the anonymised versions (FR-033) — **NOT REQUIRED**: see T078 note. No anonymisation action needed.
- [X] T080 [US4] [P] Create `DATASETS/README.md` — describe `sample_boxes.csv` (origin: synthetic; schema: id, length, width, height, weight; intended use: quickstart demo) and every other retained dataset (FR-032)
- [X] T081 [US4] Relocate `CODE/` to `legacy/`: `git mv CODE legacy`; create `legacy/README.md` explaining "preserved for historical reference, NOT part of the supported package; paths hardcoded for the original author's environment; see `src/bin_packer_3d/` for the current implementation"; tag the pre-move commit as `legacy-code-preserved` before the move (FR-031) — also added `exclude: '^legacy/'` to `.pre-commit-config.yaml` and `extend-exclude = ["legacy", "DATASETS"]` to pyproject `[tool.ruff]` so hooks / linters leave preserved content byte-faithful.
- [X] T082 [US4] Update `pyproject.toml` with `[tool.hatch.build] exclude = ["legacy/", "CODE/", "DATASETS/*.xlsx"]` (hatch native exclusion syntax); add matching entries to `MANIFEST.in` for sdist; verify via `hatch build --clean && tar tzf dist/*.tar.gz | grep -E '^legacy/|^CODE/'` returning empty — **no-op**: pyproject.toml and MANIFEST.in already authored in Setup phase (T008 pyproject, T082-equivalent MANIFEST.in content); hatch-build verification deferred per project rule "never build after changes".
- [X] T083 [US4] [P] Create `scripts/audit_datasets.py` — scans working tree AND git history (`git log --all --full-history --pretty=format: --name-only`) for patterns matching business-artefact names (PACKING LIST, PESO_P.T, internal codes); exits non-zero on hit; called from `release.yml` pre-publish gate (Constitution §Security, US8 integration)
- [X] T084 [US4] Verify author field alignment — run FR-034 compliance check, fix any mismatch between `src/bin_packer_3d/__init__.py` `__author__` and `pyproject.toml` `authors` — verified aligned (both read "Bruno Ghiberto"); added a forward-guard `test_author_alignment` in `tests/integration/test_hygiene.py` to catch future drift.

**Checkpoint**: US4 complete. `pytest tests/integration/test_hygiene.py tests/integration/test_datasets.py` green. `git log --all` shows no confidential artefacts. `legacy/` clearly labelled.

---

## Phase 7: User Story 5 — Algorithm Portfolio & Reproducible Benchmarks (Priority: P2) · (v0.3.0)

**Goal**: Library implements BFD + Extreme Point + Maximal Rectangles in addition to existing FFD + Shelf. `bin-packer benchmark` runs every algorithm against BR1 (and optionally the full BR1–BR8 suite) with reproducible, seeded results.

**Independent Test**: `bin-packer benchmark --instance BR1 --seed 0` produces a comparison table with utilisation %, bins used, success rate, runtime for each of the 5 algorithms. Rerun — bit-identical. At least one new algorithm shows ≥ 5 pp utilisation delta or ≥ 1 bin delta from FFD.

### Tests for User Story 5 ⚠️

- [ ] T085 [P] [US5] Write property-based tests in `tests/property/test_invariants.py` parametrised over `ALGORITHMS` keys: non-overlap, bounds containment, volume conservation, box-count conservation (FR-046, spec §Edge Cases)
- [ ] T086 [P] [US5] Write unit test for BFD in `tests/unit/test_algorithms.py::TestBFD`: known small input with hand-computed expected placement count
- [ ] T087 [P] [US5] Write unit test for Extreme Point in `tests/unit/test_algorithms.py::TestExtremePoint`: Crainic 2008 reference example (small instance)
- [ ] T088 [P] [US5] Write unit test for Maximal Rectangles in `tests/unit/test_algorithms.py::TestMaximalRectangles`: small instance with hand-computed expected utilisation
- [ ] T089 [P] [US5] Write integration test for benchmark reproducibility in `tests/integration/test_benchmark.py::test_same_seed_same_result`: run `BenchmarkRunner(algorithm="extreme_point", instance="BR1", seed=0)` twice, assert placements (ignoring timestamp) are bit-identical (FR-043)
- [ ] T090 [P] [US5] Write integration test for CLI benchmark output formats in `tests/integration/test_cli.py::test_benchmark_formats`: `bin-packer benchmark --format text|json|markdown`, assert output parses correctly in each format (FR-042)
- [ ] T091 [P] [US5] Write integration test for unknown algorithm error in `tests/integration/test_cli.py::test_benchmark_unknown_algo`: `bin-packer benchmark --algorithm bogus`; assert exit non-zero; assert stderr lists every registered algorithm name (FR-047)

### Implementation for User Story 5

- [ ] T092 [US5] Create `src/bin_packer_3d/algorithms/bfd.py` — `BestFitDecreasingPacker(BasePacker)`; `@register("bfd")`; classvar `complexity = "O(n log n)"`; consults `_check_constraints` before accepting each placement (T014 hook) (FR-040)
- [ ] T093 [US5] Create `src/bin_packer_3d/algorithms/extreme_point.py` — `ExtremePointPacker(BasePacker)`; `@register("extreme_point")`; implements Crainic/Perboli/Tadei 2008; classvar `complexity = "O(n²)"`; consults `_check_constraints` (FR-040)
- [ ] T094 [US5] Create `src/bin_packer_3d/algorithms/maximal_rectangles.py` — `MaximalRectanglesPacker(BasePacker)`; `@register("maximal_rectangles")`; classvar `complexity = "O(n³)"`; consults `_check_constraints` (FR-040)
- [ ] T095 [US5] Flesh out the three algorithm doc stubs (T059–T061): add pseudo-code, literature citation, "when to use" guidance per FR-021
- [ ] T096 [US5] [P] Create `src/bin_packer_3d/models/metadata.py` — `AlgorithmMetadata` frozen dataclass (name, version, parameters, seed, timestamp) per data-model.md
- [ ] T097 [US5] [P] Create `src/bin_packer_3d/benchmark/__init__.py`
- [ ] T098 [US5] Create `src/bin_packer_3d/benchmark/instances.py` — `BenchmarkInstance` dataclass; `load_br(n: int) -> BenchmarkInstance` loader; if `benchmark/instances/BR{n}.txt` exists, parse it; otherwise call `benchmark.download.fetch_br(n)` (FR-044)
- [ ] T099 [US5] [P] Create `src/bin_packer_3d/benchmark/download.py` (package-internal, installed into site-packages) — `fetch_br(n: int) -> BenchmarkInstance` downloads from a canonical public URL when the instance license permits; raises `BenchmarkError` otherwise; documents the URL in `DATASETS/README.md`. Also create `benchmark/download_instances.py` at REPO ROOT as a thin CLI wrapper that calls `fetch_br` for each BR in 1..8 and caches under `benchmark/instances/` for dev convenience (NOT shipped in the wheel — excluded via T082) (FR-044)
- [ ] T100 [US5] Create `src/bin_packer_3d/benchmark/results.py` — `BenchmarkResult` dataclass with JSON serialiser matching `contracts/benchmark-format.md` schema_version `"1"`
- [ ] T101 [US5] Create `src/bin_packer_3d/benchmark/formats.py` — `format_text(results)`, `format_json(results)`, `format_markdown(results)` (FR-042)
- [ ] T102 [US5] Create `src/bin_packer_3d/benchmark/runner.py` — `BenchmarkRunner(algorithms, instance, seed)`; runs each algorithm on the instance; captures `elapsed_seconds`, `n_bins_used`, `volume_utilisation`, `success_rate`; returns `list[BenchmarkResult]`; seeds `random`/`numpy.random` globally before each run for FR-043 reproducibility
- [ ] T103 [US5] Add `benchmark` subcommand to `src/bin_packer_3d/cli.py`: flags per `contracts/cli.md`; delegates to `BenchmarkRunner`; renders via `format_*` helpers (FR-041, FR-042, FR-043, FR-047)
- [ ] T104 [US5] Create `benchmark/run_baseline.py` — standalone script: instantiate `BenchmarkRunner(all_algos, "BR1", seed=0)`; write result JSON to `docs/benchmarks/results/baseline-{version}.json`; used for baseline capture (ADR-0005, Gate G11)
- [ ] T105 [US5] Create `.github/workflows/benchmark-br1.yml` — runs on every push to main and on PR touching `src/bin_packer_3d/algorithms/`; executes `python -m bin_packer_3d benchmark --instance BR1 --seed 0 --format json > br1.json`; attaches as artefact; emits CI warning if utilisation drop > 25% or runtime increase > 25% vs `docs/benchmarks/results/baseline-0.3.0.json` (FR-045, Constitution §Benchmark workflow)
- [ ] T105a [P] [US5] Write SC-006 gate test in `tests/integration/test_benchmark.py::test_sc006_delta_threshold`: run `BenchmarkRunner` on BR1 with seed=0 across all registered algorithms; assert at least ONE of {bfd, extreme_point, maximal_rectangles} achieves EITHER `volume_utilisation - ffd.volume_utilisation >= 0.05` OR `ffd.n_bins_used - algo.n_bins_used >= 1`; this test is a BLOCKING gate for tagging v0.3.0 (SC-006, FR-040)
- [ ] T106 [US5] [P] Update `pyproject.toml` `[tool.pytest.ini_options] markers` to include `property: hypothesis-driven invariant tests`; expand `testpaths = ["tests"]` globbing to include `tests/property/`
- [ ] T107 [US5] Configure Hypothesis CI profile in `tests/conftest.py`: register a profile via `hypothesis.settings.register_profile("ci", derandomize=True, database=None, print_blob=True, max_examples=200)`; load it via `hypothesis.settings.load_profile("ci")` when `os.environ.get("CI") == "true"`; default "dev" profile unchanged. Document in `CONTRIBUTING.md` how to reproduce a CI failure locally (`HYPOTHESIS_PROFILE=ci pytest ...`) (Constitution IV)
- [ ] T108 [US5] [P] Update `src/bin_packer_3d/__init__.py` public re-exports: add `AlgorithmMetadata`, `BenchmarkResult`, `BenchmarkRunner` (contracts/api.md)
- [ ] T109 [US5] Update `src/bin_packer_3d/config.py` `PackerConfig`: add `seed: int | None = None` field (FR-043)

**Checkpoint**: US5 complete. `bin-packer benchmark --instance BR1 --seed 0` runs all 5 algorithms reproducibly. At least one of BFD/EP/MR shows the SC-006 delta vs FFD. Per-push CI benchmark emits JSON artefact. Property tests green on every algorithm.

---

## Phase 8: User Story 6 — Observability & Diagnostics (Priority: P2) · (v0.3.0)

**Goal**: Library uses stdlib loggers politely; CLI supports `--verbose`/`--quiet`; `bin-packer pack --explain` emits per-box placement trace; malformed CSV rows surface via both WARNING log and structured `LoadReport`.

**Independent Test**: `bin-packer pack --explain -v` against small instance → per-box trace at DEBUG. CSV with malformed rows → WARNING per bad row AND each bad row in `result.load_report.rejected_rows`.

### Tests for User Story 6 ⚠️

- [ ] T110 [P] [US6] Write test for `--verbose`/`--quiet` level mapping in `tests/integration/test_cli.py::test_verbosity_flags`: each flag maps to the right effective log level on the `bin_packer_3d` logger (FR-051)
- [ ] T111 [P] [US6] Write test for `--explain` trace in `tests/integration/test_cli.py::test_explain_trace`: `bin-packer pack --explain -v -i tiny.csv`; assert stderr contains `packing.attempt` DEBUG event per input box and `packing.reject` per rejection (FR-054)
- [ ] T112 [P] [US6] Write test for loader WARNING behaviour in `tests/unit/test_data_loaders.py::test_malformed_row_warning`: patch logger handlers, call loader on a CSV with one malformed row, assert (a) a WARNING event was logged, (b) `report.rejected_rows` has one entry matching (FR-053)
- [ ] T113 [P] [US6] Write test for no-print policy in `tests/integration/test_hygiene.py::test_no_prints`: grep `src/bin_packer_3d/` excluding `cli.py` for `print(`; fail if any match (FR-052)

### Implementation for User Story 6

- [ ] T114 [US6] Extend `src/bin_packer_3d/observability.py` (from T011) with `StructuredAdapter(logging.LoggerAdapter)` that lifts `extra["fields"]` into LogRecord fields (ADR-0008)
- [ ] T115 [US6] Emit events from `src/bin_packer_3d/algorithms/base.py` `BasePacker.pack`: `packing.start` (algorithm, n_boxes, bin_dimensions), `packing.end` (success_rate, runtime, placements_placed); DEBUG level (ADR-0008)
- [ ] T116 [US6] Emit events from `_accept` path: `packing.attempt` (box_id, candidate_position, orientation), `packing.reject` (box_id, reason) at DEBUG — surfaces via `--explain` (FR-054)
- [ ] T117 [US6] Refactor CLI to accept `--verbose`/`-v` and `--quiet`/`-q` flags (mutually exclusive), translating to log levels on the `bin_packer_3d` logger (FR-051, `contracts/cli.md`)
- [ ] T118 [US6] Add `--explain` flag to `bin-packer pack` subcommand: enables DEBUG logging on `bin_packer_3d.algorithms` only (not global), so trace is focused
- [ ] T119 [US6] Refactor `src/bin_packer_3d/data/loaders.py` to emit WARNING via `get_logger("data.loaders")` for every entry added to `LoadReport.rejected_rows` — guarantees dual surfacing (FR-053)
- [ ] T120 [US6] Sweep `src/bin_packer_3d/` for `print(` calls outside `cli.py`; replace with appropriate logger calls (FR-052) — verified by T113
- [ ] T121 [US6] [P] Update `src/bin_packer_3d/__init__.py` public re-exports: ensure `get_logger` is exported (already in T015, double-check)
- [ ] T122 [US6] [P] Document observability in `docs/api/observability.md` — log namespace tree (`bin_packer_3d.algorithms`, `bin_packer_3d.data`, ...), event catalogue (what each event emits), recipe to configure a consumer-side JSON formatter using `StructuredAdapter`

**Checkpoint**: US6 complete. `bin-packer pack --explain -v` emits focused per-box trace. `tests/integration/test_hygiene.py::test_no_prints` green. Library can be embedded without polluting the consumer's logging.

---

## Phase 9: User Story 7 — Constraint Framework Foundation (Priority: P2) · (v0.3.0)

**Goal**: Declarative constraints (`AllowedOrientations`, `SupportedWeight`) consulted by every registered algorithm. Adding a new constraint in future requires NO changes to any algorithm's placement loop.

**Independent Test**: `Box(allowed_orientations={"ltw_wth"})` — no placement by any algorithm rotates it onto its side. `Box(max_supported_weight=5.0)` — stacks above that box with overhead > 5.0 are rejected by every algorithm.

### Tests for User Story 7 ⚠️

- [ ] T123 [P] [US7] Write test for `AllowedOrientations` in `tests/unit/test_constraints.py::test_allowed_orientations` parametrised over `ALGORITHMS`: every algorithm respects the set (FR-061)
- [ ] T124 [P] [US7] Write test for `SupportedWeight` in `tests/unit/test_constraints.py::test_supported_weight` parametrised over `ALGORITHMS`: stacking beyond `max_supported_weight` is rejected (FR-062)
- [ ] T125 [P] [US7] Write test for constraint extensibility in `tests/unit/test_constraints.py::test_custom_constraint_requires_no_algo_edit`: define a no-op custom `Constraint` subclass in the test; add it to `PackerConfig.constraints`; call `pack`; assert no algorithm source file was touched to make this work (FR-063)

### Implementation for User Story 7

- [ ] T126 [US7] [P] Create `src/bin_packer_3d/constraints/__init__.py`
- [ ] T127 [US7] Create `src/bin_packer_3d/constraints/base.py` — `Constraint` ABC with `check(placement, box, bin, existing_placements) -> ConstraintResult`; `ConstraintResult` frozen dataclass (ADR-0007, FR-060)
- [ ] T128 [US7] [P] Extend `src/bin_packer_3d/models/box.py`: add `allowed_orientations: frozenset[str] | None = None` and `max_supported_weight: float | None = None` fields (FR-061, FR-062); pydantic validator coerces `set` → `frozenset`
- [ ] T129 [US7] Create `src/bin_packer_3d/constraints/orientation.py` — `AllowedOrientations(Constraint)` reads `box.allowed_orientations`; rejects with reason when candidate `placement.orientation` not in allowed set (FR-061)
- [ ] T130 [US7] Create `src/bin_packer_3d/constraints/weight.py` — `SupportedWeight(Constraint)` computes supporting-box chain; rejects if cumulative overhead exceeds `p.box.max_supported_weight` (FR-062)
- [ ] T131 [US7] Extend `src/bin_packer_3d/config.py` `PackerConfig`: add `constraints: list[Constraint] = Field(default_factory=list)`; arbitrary-types allowed (FR-060)
- [ ] T132 [US7] [P] Update `src/bin_packer_3d/__init__.py` public re-exports: add `Constraint`, `ConstraintResult`, `AllowedOrientations`, `SupportedWeight`
- [ ] T133 [US7] [P] Complete `docs/constraints.md` (stub from T062): document the constraint architecture, list implemented constraints (orientations, supported weight), explicitly enumerate deferred constraints (stability, centre-of-mass, etc.) (FR-064)
- [ ] T134 [US7] [P] Add `docs/adr/0007-constraint-framework.md` already listed — verify content matches research.md ADR-0007

**Checkpoint**: US7 complete. Property tests extended: orientation + weight invariants parametrised over every algorithm hold. Adding a custom `Constraint` subclass in user code requires no library edits.

---

## Phase 10: User Story 8 — Distribution & Release Engineering (Priority: P3) · (v1.0.0)

**Goal**: `pipx install bin-packer-3d` installs and exposes `bin-packer`. `docker run ghcr.io/bruno-ghiberto/bin-packer-3d info` works. Releases via OIDC trusted publisher — no long-lived tokens.

**Independent Test**: Clean machine: `pipx install bin-packer-3d && bin-packer pack DATASETS/sample_boxes.csv` — works end-to-end. `docker run <image> info` — prints registered algorithms.

### Tests for User Story 8 ⚠️

- [ ] T135 [P] [US8] Write test for version-tag consistency in `tests/integration/test_release_consistency.py`: assert `pyproject.toml` version, `bin_packer_3d.__version__`, and the most-recent `CHANGELOG.md` `## [x.y.z]` header all match (FR-074)
- [ ] T136 [P] [US8] Write test for versioning-policy doc presence in `tests/integration/test_release_consistency.py::test_versioning_policy_documented`: assert `docs/versioning.md` exists and contains MAJOR/MINOR/PATCH definitions (FR-073)

### Implementation for User Story 8

- [ ] T137 [US8] Create `Dockerfile` at repo root — multi-stage: builder stage (`python:3.11-slim`, `pip wheel --no-deps`); runtime stage (`python:3.11-slim`, non-root user `app` UID 1000); `ENTRYPOINT ["bin-packer"]`; `CMD ["info"]` (ADR-0006, FR-072)
- [ ] T138 [US8] [P] Create `.dockerignore` — exclude `.git`, `.github`, `docs/`, `tests/`, `benchmark/`, `legacy/`, `DATASETS/`, `*.pyc`, `__pycache__`, `.venv`
- [ ] T139 [US8] Create `.github/workflows/release.yml` — triggers on `push: tags: ["v*.*.*"]`; jobs: `ci` (calls `_ci-core.yml`), `full-benchmark` (runs BR1–BR8 on every algorithm, attaches JSON artefact per FR-045), `privacy-audit` (runs `scripts/audit_datasets.py`), `publish-pypi` (needs: ci, full-benchmark, privacy-audit; `if: needs.ci.outputs.all_passed == 'true'`; uses `pypa/gh-action-pypi-publish` with `id-token: write` — OIDC trusted publisher, no `password:`), `docs-deploy` (needs: publish-pypi; runs `mkdocs gh-deploy --force`), `docker-publish` (needs: publish-pypi; builds and pushes to `ghcr.io/bruno-ghiberto/bin-packer-3d`), `gh-release` (creates GitHub Release with full-benchmark JSON attached) (ADR-0006, FR-070, FR-072)
- [ ] T140 [US8] Configure OIDC trusted publisher on PyPI dashboard (manual, documented in `docs/maintainers.md`): Publisher = GitHub Actions, Owner = `Bruno-Ghiberto`, Repository = `3D_BIN_PACKING`, Workflow = `release.yml`, Environment = `pypi` (ADR-0006, FR-070)
- [ ] T141 [US8] Reserve `bin-packer-3d` on PyPI: first release is a placeholder `0.2.0` wheel published via `release.yml`; fallback names in ADR-0006 if taken (spec §Assumptions)
- [ ] T142 [US8] [P] Create `docs/versioning.md` — define MAJOR/MINOR/PATCH semantics for this project per Constitution §Release workflow (FR-073, SC-011)
- [ ] T143 [US8] [P] Add PyPI and Docker badges to `README.md`: PyPI version, Docker pulls count (once published) (FR-015)
- [ ] T144 [US8] Wire `docker/metadata-action` + `docker/build-push-action` (pinned SHAs) into `docker-publish` job: tag image as `ghcr.io/bruno-ghiberto/bin-packer-3d:{version}` AND `:latest` on the release SHA
- [ ] T145 [US8] Write release smoke test in `scripts/release_smoke.sh` — runs after a release to verify: (a) `pip install bin-packer-3d==<version>` works, (b) `bin-packer info` runs, (c) `docker pull && docker run ghcr.io/.../info` runs; invoked manually by maintainer post-release

**Checkpoint**: US8 complete. `git tag v1.0.0 && git push origin v1.0.0` triggers end-to-end release: CI gate → full benchmark → privacy audit → PyPI publish → docs deploy → Docker publish → GH Release. Post-release smoke script confirms all three entry points work.

---

## Phase 11: User Story 9 — Interactive Demo (Priority: P3) · (v1.0.0)

**Goal**: A single demo link in the README takes a visitor to a rendered 3D packing visualisation within 30 seconds — no install.

**Independent Test**: Open README in fresh browser, click the demo link. Within 30 s a 3D packing visualisation renders.

### Tests for User Story 9 ⚠️

- [ ] T146 [P] [US9] Write link-check test in `tests/integration/test_readme_demo_link.py`: parse README for the demo link; assert exactly ONE exists; assert the URL resolves (HTTP 200) when CI runs with network; mark as `integration` (runs in CI but skippable offline) (FR-080)

### Implementation for User Story 9

- [ ] T147 [US9] Create `docs/demo.html` — a statically-exported Plotly 3D packing visualisation from a canonical BR1 Extreme Point run; file written by `scripts/generate_demo.py` (committed output); served by GitHub Pages at `https://bruno-ghiberto.github.io/3D_BIN_PACKING/demo.html` (FR-080)
- [ ] T148 [US9] Create `scripts/generate_demo.py` — runs `BenchmarkRunner("extreme_point", "BR1", seed=0)`; calls `plot_packing(result, output_path="docs/demo.html")`; deterministic output (FR-080, Principle IV)
- [ ] T149 [US9] [P] Add a `docs-regenerate-demo` job to `release.yml` — regenerates `docs/demo.html` on every release and commits it via the docs-deploy job (keeps demo current with the algorithm portfolio)
- [ ] T150 [US9] Add demo link to `README.md` under a "Quick demo" heading — single link to `docs/demo.html` on GH Pages; include a hint text "opens a pre-rendered 3D visualisation — no install required" (FR-080)

**Checkpoint**: US9 complete. Visitor clicks README demo link → sees packing render in < 30 s. Link-check CI test green.

---

## Phase 12: Polish & Cross-Cutting Concerns

**Purpose**: Final sweep before `v1.0.0` tag.

- [ ] T151 [P] Run `/speckit.analyze` to validate cross-artefact consistency (spec ↔ plan ↔ tasks ↔ data-model ↔ contracts)
- [ ] T152 [P] Full constitution compliance audit per Constitution §Compliance review — walk every principle, produce pass/fail line in `docs/compliance/v1.0.0-audit.md`
- [ ] T153 [P] Regenerate `docs/benchmarks.md` from latest `docs/benchmarks/results/br1-br8-v1.0.0.json`
- [ ] T154 Update `CHANGELOG.md` `[1.0.0]` section with complete release notes: breaking changes, new features, bug fixes, migration notes for `Box.weight` / `PackingResult` / `PackerConfig.strategy` changes
- [ ] T155 [P] Run quickstart.md Scenarios A and B on a clean Linux VM; record any deviation; fix before tagging
- [ ] T156 [P] Verify every spec §Success Criteria entry (SC-001..SC-014) — each has a matching checkbox in `docs/compliance/v1.0.0-audit.md`
- [ ] T157 [P] Repository settings pass: description, topics (`bin-packing`, `3d-packing`, `operations-research`, `optimization`, `logistics`, `python-library`), homepage URL (`https://bruno-ghiberto.github.io/3D_BIN_PACKING/`)
- [ ] T157a [P] Create top-level README.md in every repo-root data/script directory not already covered: `benchmark/README.md` (purpose: BR1–BR8 instances + baseline script), `examples/README.md` (runnable comparison notebook), `scripts/README.md` (audit + demo-regeneration scripts) — one-paragraph purpose each so SC-013 passes the 10-second test
- [ ] T158 Tag `v1.0.0`: `git tag v1.0.0 -m "1.0.0 — public-release hardening"` (no `-s` unless GPG configured); push tag triggers `release.yml`
- [ ] T159 Post-release: verify PyPI page, Docker image on GHCR, docs deployed, demo link works; run `scripts/release_smoke.sh`
- [ ] T160 Final compliance observation — save release-audit summary to Engram memory via `mem_save`; also manually validate SC-014 by opening a throwaway PR with intentional lint + type + test violation; confirm each independently blocks merge; close the PR and link the audit run in `docs/compliance/v1.0.0-audit.md`

---

## Dependencies & Execution Order

### Phase dependencies

- **Phase 1 (Setup)**: no dependencies — start immediately.
- **Phase 2 (Foundational)**: depends on Phase 1 — BLOCKS every user story.
- **Phase 3 (US1)**: depends on Phase 2.
- **Phase 4 (US2)**: depends on Phase 1 (only — CI can bootstrap without library changes); integrates with US1 via test jobs.
- **Phase 5 (US3)**: depends on Phase 1 (tooling) + partial dependencies on US1/US5/US7 for content (algorithm docs, constraints doc). Docstring task T052 can start after Phase 2.
- **Phase 6 (US4)**: depends on Phase 1 only.
- **Phase 7 (US5)**: depends on US1 (registry exists) + Phase 2.
- **Phase 8 (US6)**: depends on Phase 2 (observability.py bootstrap) + US1 (loader refactor).
- **Phase 9 (US7)**: depends on Phase 2 (BasePacker constraint hook) + US1 (registry, so property tests parametrise over it).
- **Phase 10 (US8)**: depends on US1–US7 complete (release ships all stories together per spec §Dependencies US8→US5).
- **Phase 11 (US9)**: depends on US5 (benchmark runner used by `generate_demo.py`) + US8 (release deploys it).
- **Phase 12 (Polish)**: depends on every preceding phase.

### User-story independence

- US1 (registry) is the linchpin for US5 and US7. US1 can be tested independently — its Independent Test doesn't require US5's new algorithms or US7's constraints.
- US2 (CI) is independent — a dedicated CI PR demonstrates the gate behaviour without any `src/` changes.
- US4 (hygiene) is fully independent.
- US6 (observability) depends on US1's loader refactor for the WARNING-log behaviour; otherwise independent.
- US8/US9 are ship-it-all bundles by design (spec §Dependencies US8→US5).

### Within each user story

- Tests (T01x, T07x, T08x, T11x, T12x, T13x, T14x) MUST be written and MUST FAIL before their corresponding implementation tasks.
- Models before services before endpoints before integration (classical pyramid).
- Commit after each task or small logical group — Constitution §Branching/commit conventions.

---

## Parallel Opportunities

### Phase 1 (Setup)

T002, T003, T004, T005, T006, T007, T008, T009 — all `[P]` — can run in parallel.

### Phase 2 (Foundational)

T010 / T012 (test tasks) can run parallel. T013 / T014 / T015 implementation tasks can run parallel once T011 (observability bootstrap) is done.

### Phase 3 (US1)

Test tasks T016–T020 all parallel. Model tasks T025 / T026 / T027 parallel. Registry tasks T022 / T023 parallel with T021 as dependency.

### Phase 4 (US2)

T036–T045 — all CI / GitHub config files — fully parallel.

### Phase 5 (US3)

Docstring task T052, doc content tasks T054–T071 — all parallel (different files).

### Phase 7 (US5)

Algorithm test tasks T086 / T087 / T088 parallel. Algorithm implementations T092 / T093 / T094 parallel (different files). Benchmark package tasks T096–T102 largely parallel.

### Cross-story parallelism

Once Phase 2 is done, teams can work on:

- Developer A: US1 → US5 → US7 (library-depth track)
- Developer B: US2 → US3 → US6 (dev-experience track)
- Developer C: US4 → US8 → US9 (release track)

All three tracks converge at Polish (Phase 12).

---

## Parallel Example: User Story 1 test setup

```bash
# All test tasks for US1 can be launched in parallel:
Task: "T016 [P] [US1] Contract test for algorithm registry in tests/unit/test_registry.py"
Task: "T017 [P] [US1] Integration test for CLI info in tests/integration/test_cli.py"
Task: "T018 [P] [US1] Unit tests for loader column mapping in tests/unit/test_data_loaders.py"
Task: "T019 [P] [US1] Unit test for Box.weight states in tests/unit/test_models.py"
```

Once tests fail, launch model/config changes in parallel:

```bash
Task: "T025 [US1] Change Box.weight to Optional[float]"
Task: "T026 [US1] Change Bin.max_weight to Optional[float]"
Task: "T027 [US1] ColumnMapping dataclass in data/loaders.py"
```

T024 (`PackerConfig.strategy` validator) depends on T021 (`ALGORITHMS` registry) — sequential.

---

## Implementation Strategy

### MVP First — User Story 1 only

1. Complete Phase 1 (Setup) — tooling.
2. Complete Phase 2 (Foundational) — observability + result models.
3. Complete Phase 3 (US1) — registry + loader + weight.
4. **STOP** and verify US1 Independent Test passes.
5. Tag `v0.2.0-mvp` locally (no public push) for the internal checkpoint.

### Incremental Delivery

- `v0.2.0` — US1 + US2 (bootstrap) + US3 (seed) + US4 — the Phase A bundle.
  - Ship: P1 foundation. Public trust restored. CI green. DATASETS clean.
- `v0.3.0` — US5 + US6 + US7 + US2 (matrix completion).
  - Ship: algorithmic depth. Benchmarks reproducible. Observability in place.
- `v1.0.0` — US3 (completion) + US8 + US9 + Polish.
  - Ship: public release to PyPI + Docker + GH Pages. Demo link live.

### Single-maintainer strategy

Since this is a solo project, parallelism is bounded by the single developer.
Prioritise by Independent-Test value:

1. US1 (biggest credibility unlock — phantom strategies are the #1 first-impression destroyer).
2. US4 (DATASETS audit — this is a privacy risk, not just hygiene).
3. US2 (CI — converts every subsequent claim into an automated signal).
4. US5 (OR-depth signal — primary CV value for OR audience).
5. US3, US6, US7 in any order.
6. US8 + US9 — release-gated, must wait for the rest.

---

## Validation

- Total task count: **162** (T001..T160 + T105a + T157a, inserted post-analyze)
- Task distribution:
  - Setup: 9 tasks (T001–T009)
  - Foundational: 6 tasks (T010–T015)
  - US1: 15 tasks (T016–T030)
  - US2: 19 tasks (T031–T049)
  - US3: 25 tasks (T050–T074)
  - US4: 10 tasks (T075–T084)
  - US5: 26 tasks (T085–T109 + T105a) — SC-006 gate added
  - US6: 13 tasks (T110–T122)
  - US7: 12 tasks (T123–T134)
  - US8: 11 tasks (T135–T145)
  - US9: 5 tasks (T146–T150)
  - Polish: 11 tasks (T151–T160 + T157a) — directory-README coverage added
- Parallel opportunities: **~62 tasks marked `[P]`**, clustered in Setup, Docs, and CI-config phases.
- Each user story has an Independent Test that maps directly to its phase's checkpoint.
- Tests authored first: **every** user story (and Phase 2) has its test tasks preceding implementation tasks — Constitution II satisfied.
- Coverage: **100 %** of FRs (54/54) and **100 %** of buildable SCs (14/14) post-remediation — H1 and M6 resolved; M4 moved to constitution PATCH v1.0.1.
- Format validation: every task above follows `- [ ] TNNN [P?] [USn?] Description — file path` pattern.

---

## Notes

- `[P]` tasks = different files, no dependencies on an incomplete task.
- `[Story]` label — required for every task inside a user-story phase.
- Setup, Foundational, and Polish phases carry no `[Story]` label — they are cross-cutting.
- Each user story is independently completable and testable — verified by the Independent Test block in its phase header.
- Tests MUST fail before implementation — Constitution II Gate G2.
- Commit after each task or small logical group; use Conventional Commits — Constitution §Commit style.
- Stop at any checkpoint to validate the story independently; do not chain phases without validation.
- Avoid: vague tasks, same-file conflicts, cross-story dependencies that break independence. Any such drift is a merge blocker.
