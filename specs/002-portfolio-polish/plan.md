# Implementation Plan: Portfolio Polish of `bin-packer-3d`

**Branch**: `002-portfolio-polish` | **Date**: 2026-05-13 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/002-portfolio-polish/spec.md`
**Constitution**: [v1.0.1](../../.specify/memory/constitution.md)

## Summary

Take `bin-packer-3d` at `v0.2.0.dev0` — a working library with three registered algorithms (`bfd`, `ffd`, `shelf`), 103 passing tests, a 10-check CI pipeline, `mypy --strict` clean, and MIT license — and produce a presentation-layer polish pass that ships under tag `v0.3.0-rc1`. This is **not** a feature-completion phase; US5 Parts 2-3 (Extreme Point + Maximal Rectangles + benchmark CI gate) continues on its parallel branch (`feature/us5-extreme-point-benchmark`). The polish pass MUST NOT block, depend on, or anticipate that work.

The approach is additive: extend the existing `src/bin_packer_3d/` tree with a branded Plotly theme module (`visualization/theme.py`) and deterministic colour-assignment module (`visualization/palette.py`); add a `bin-packer demo` Click subcommand; commit a procedurally generated headline dataset (`examples/headline.csv` + seed); publish a `mkdocs-material` documentation site to GitHub Pages (`actions/deploy-pages`); rewrite the README as a generated-section composition (algorithm comparison table + Highlights section + Project Structure section all sourced from the runtime via `scripts/regenerate_readme.py`); install a baseline accessibility floor (alt text on every README image, colourblind-safe palette verified empirically via `colorspacious`, mkdocs-material WCAG-AA defaults preserved); and tag `v0.3.0-rc1` at PR merge time. The five clarifications recorded in `spec.md § Clarifications` (2026-05-13) are LOCKED and MUST NOT be reopened.

## Technical Context

**Language/Version**: Python 3.11 (public floor); maintainer develops on 3.14.4; CI matrix covers 3.11 / 3.12 / 3.13 / 3.14 on `ubuntu-latest`.

**Primary Dependencies** (runtime unless noted; NO new mandatory runtime deps in this phase, per Principle V):

- Runtime: `numpy >= 1.24`, `plotly >= 5.18,<6.0.0` (upper bound NEW — ADR-001), `pydantic >= 2`, `pydantic-settings >= 2`, `click >= 8.1`, `pandas >= 2.0`, `openpyxl >= 3.1`, `rich >= 13.0`
- Dev (existing): `pytest`, `pytest-cov`, `pytest-mock`, `mypy`, `ruff`, `pre-commit`, `pandas-stubs`, `hypothesis`, `pip-audit`, `pyyaml`
- Docs extra (existing optional): `mkdocs-material >= 9.5`, `mkdocstrings[python] >= 0.25`
- **NEW** optional extras: `viz = ["kaleido>=0.2.1"]` (Plotly static-export backend — ADR-001); `dev` extended with `colorspacious` (colourblind palette verification — ADR-012)

**Storage**: no database. Files only. Generated visualisation HTML + static PNG/SVG written to user-specified output directories. Procedurally generated headline dataset committed at `examples/headline.csv`; demo command output written to `examples/output/` (gitignored).

**Testing**: existing pytest (103 passing, 1 skipped) + Hypothesis property suite. Strict TDD mode enabled (per project `CLAUDE.md`). New polish-phase test categories: reproducibility (FR-023 → SC-005), determinism (SC-006), drift (FR-004/013/014/016/026), a11y (FR-036 → SC-014), docs-build (FR-011/014).

**Target Platform**: Linux x86_64 (CI authoritative). macOS / Windows best-effort — pure-Python install, but not CI-verified at `v0.3.0-rc1`.

**Project Type**: pure-Python library with CLI entry point `bin-packer`. New polish artefacts (theme module, demo subcommand, dataset generator, headline dataset, docs site, README rewrite) ship in the same package + repository. The only new entry-point surface is `bin-packer demo` (Click subcommand registered alongside `pack | info | init`).

**Performance Goals**:

- Demo command end-to-end runtime: ≤60 seconds on commodity laptop (FR-021).
- Docs site cold-load homepage: ≤3 seconds on a 50 Mb/s connection (SC-003).
- Interactive visualisation HTML file size: ≤4.8 MB per bin (FR-010 baseline).
- Test suite runtime: ≤30 seconds on Linux CI (existing budget; MUST NOT regress).

**Constraints**:

- Zero new mandatory runtime dependencies; all polish deps under `[project.optional-dependencies]` to preserve FR-027 +5% install-footprint budget.
- `pip install bin-packer-3d` (no extras) install footprint ≤105% of v0.2.0.dev0 baseline (FR-027 + SC-007).
- Cross-Python compat: 3.11–3.14 (CI matrix).
- Constitution: all 8 Core Principles apply (4 NON-NEGOTIABLE). Contract Honesty (I) and Documentation (VI) are load-bearing for this phase.
- US5 cross-track: spec-02 MUST NOT modify `src/bin_packer_3d/algorithms/extreme_point.py` or `…/maximal_rectangles.py` (US5 creates these), or `src/bin_packer_3d/benchmark/runner.py` (US5 implements; spec-02 only relies on US5 Part 1 scaffolding).
- 103-passing test suite MUST NOT regress.
- Coverage on `src/bin_packer_3d/` MUST remain ≥90% (constitution §III).

**Scale/Scope**: single maintainer; typical reviewer reads README on GitHub then optionally clones; demo command runs on ≤200 boxes against a single bin (default 860×890×1040 mm); docs site rebuilds in ≤30 seconds; first-pass docs site has ~10 markdown pages.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Each principle from `.specify/memory/constitution.md` v1.0.1 is instantiated as a pass/fail gate. Gates marked **NON-NEG.** cannot be waived — a violation requires a constitution amendment, not a plan exception.

### G1 — Contract Honesty (Principle I, NON-NEG.)

**Pass criterion**: Every artefact this phase ships that asserts something about the runtime (README algorithm comparison table, README Highlights section, README Project Structure section, docs site algorithm pages, CLI `info` output) MUST source from a single registry (`ALGORITHMS` in `algorithms/__init__.py`) or compute its content from live repository state (test count, CI check count, Python version classifiers, etc.). Drift between any consumer and the runtime is a Contract Honesty violation and a merge blocker.

**How the plan satisfies it**:

- ADR-004 ratifies a **generated-section pattern** with `<!-- BEGIN/END -->` markers; `scripts/regenerate_readme.py` is the single regenerator that populates the algorithm comparison table, Highlights section, and Project Structure section.
- ADR-005 extends ADR-004 to the Highlights section — test count parsed from `pytest --collect-only -q | tail -1`, CI checks from `_ci-core.yml`, Python versions from `pyproject.toml` classifiers.
- ADR-003 mandates per-algorithm docs pages (`docs/algorithms/<key>.md`) with YAML front-matter `key:` + `complexity:` + `citation:` matching the registry entry. A drift test (`tests/integration/test_docs_build.py`) asserts every key has a page and every page's front-matter matches the registry.
- `tests/integration/test_readme_drift.py` re-runs the regenerator and asserts `git diff --exit-code README.md` is clean on every CI run.
- `tests/integration/test_highlights_drift.py` asserts the Highlights numbers match live state.
- `tests/integration/test_structure_drift.py` asserts the Project Structure block matches the actual filesystem.

**Status**: PASS (design enforces registry-driven sourcing at every consumer with automated drift detection).

### G2 — Test-First Discipline (Principle II, NON-NEG.)

**Pass criterion**: Every new test file is authored + red-verified locally BEFORE the implementation commit that turns it green. Strict TDD mode is enabled (project `CLAUDE.md`). Property-based tests inherited from spec-01 continue to cover algorithm invariants; this phase does not add new property tests but the existing suite MUST stay green.

**How the plan satisfies it**:

- Every new test file in §3 below (`test_visualization_theme.py`, `test_palette_colourblind.py`, `test_demo_command.py`, `test_dataset_generator.py`, `test_visualisation_e2e.py`, `test_docs_build.py`, `test_readme_drift.py`, `test_readme_alt_text.py`, `test_structure_drift.py`, `test_highlights_drift.py`) is paired with an implementation task that follows it in the task list (`/speckit-tasks` output).
- Commit messages on public-surface-touching commits include `verified red: pytest <file> exited 1 with N expected failures before this commit`.
- Phase A scaffolding writes the theme/palette/generator tests before their implementations.

**Status**: PASS (design enforces red→green pairing; commit-body convention codifies the verification).

### G3 — Automated Quality Gates (Principle III, NON-NEG.)

**Pass criterion**: CI runs on every push to `main` and every PR; executes lint (`ruff`), format-check (`ruff format --check`), strict static type check (`mypy --strict`), unit tests, integration tests, coverage (≥90% on `src/bin_packer_3d/`), and dependency vulnerability scan. New polish-phase CI jobs: docs-build (`mkdocs build --strict`), drift-tests (README + structure + highlights), install-footprint test.

**How the plan satisfies it**:

- Existing `_ci-core.yml` thin-dispatcher pattern (spec-01 ADR-003) extended with:
  - `docs-build` job: `pip install '.[docs]' && mkdocs build --strict`.
  - Drift-test jobs are part of the integration test suite (run by `pytest -m integration`); no separate workflow needed.
  - `install-footprint` job (NEW): on a clean Python venv, `pip install bin-packer-3d` (no extras), measure site-packages size, assert ≤105% of v0.2.0.dev0 baseline (committed as `tests/fixtures/install_footprint_baseline.json`).
- `pytest --cov-fail-under=90` hard-enforces the coverage floor (unchanged from spec-01).
- All new CI jobs added to `_ci-core.yml`'s aggregate `all_passed` output.

**Status**: PASS (all new gates wired into the existing reusable-core workflow).

### G4 — Reproducibility & Determinism (Principle IV)

**Pass criterion**: Every randomised process (the headline dataset generator, the deterministic colour assignment, the visualisation rendering) MUST produce bit-identical output for the same `(seed, input, configuration)`. Visualisation snapshot tests assert byte-identical HTML and pixel-equal static PNG across runs.

**How the plan satisfies it**:

- ADR-007 (palette): BLAKE2b hash → ColorBrewer Set3 index. BLAKE2b is deterministic across Python versions (no hash-randomisation interference, unlike `hash()`). Same `box_id` always produces the same colour.
- ADR-009 (dataset generator): seeded RNG; same seed → byte-identical CSV. Test `test_dataset_generator.py` asserts.
- ADR-001 (Plotly pinned `<6.0.0`) + Kaleido pinned: pixel-equal static PNG across runs guaranteed by version stability.
- ADR-011 (snapshot tests): expected outputs committed to `tests/fixtures/expected/`; comparison via stdlib `filecmp` and `hashlib` (no `pytest-snapshot` dep).
- `tests/integration/test_visualisation_e2e.py` runs identical input twice and asserts byte-identical HTML.

**Status**: PASS.

### G5 — Library Citizenship (Principle V)

**Pass criterion**: No `print()` outside `cli.py`; no `logging.basicConfig()`; no `os.chdir`, `sys.path` mutation, or env writes at import; no network or blocking I/O at import; errors via typed exceptions; never `sys.exit()` from library code.

**How the plan satisfies it**:

- New modules (`theme.py`, `palette.py`) follow the same conventions as existing library code. No `print()`; no logging configuration; no import-time side effects.
- Theme module is data-only (template constants + a pure function `apply_theme`). Palette module is pure (hash → index → colour).
- The `demo` CLI subcommand is in `cli.py` — Rich-backed output allowed there per Principle V.
- New optional deps (`kaleido`, `colorspacious`) are import-guarded: ImportError → clear instruction to install the extra. No silent fallback that changes runtime behaviour.

**Status**: PASS.

### G6 — Documentation as Artefact (Principle VI, NON-NEG.)

**Pass criterion**: Every public class and function carries a docstring; missing docstrings fail CI (existing ruff `D` ruleset). Documentation site builds in CI on every push to `main`. README passes the 30-second credibility test (FR-001 hero, FR-002 problem statement, FR-003 install/run, FR-004 algorithm table, FR-005 gallery — all per spec-02 user story 1). ADRs recorded for non-trivial decisions.

**How the plan satisfies it**:

- ADR-002 locks `mkdocs-material` + GitHub Pages + `actions/deploy-pages` (official) for the docs site. `docs-deploy.yml` workflow ships in Phase B.
- ADR-003 mandates hybrid algorithm-page strategy: `mkdocstrings` for API ref + hand-written prose for each algorithm page.
- Twelve ADRs (`research.md` ratifies them) record the non-trivial architectural decisions for this phase.
- README rewrite is a Phase B deliverable (US1).
- All new public symbols (`VisualisationStyle`, `apply_theme`, `colour_for_box`, demo subcommand) carry Google-format pydocstyle-compliant docstrings (FR-029).

**Status**: PASS.

### G7 — Privacy by Default (Principle VII)

**Pass criterion**: No identifiable real-world business, personal, or customer data in working tree or git history. The polish phase introduces only synthetic and procedurally-generated data; this principle is vacuously satisfied for new additions but inherited audit obligations from spec-01 (the per-file `DATASETS/AUDIT.md` review) remain in force.

**How the plan satisfies it**:

- `examples/headline.csv` is procedurally generated from a committed seed (ADR-009) — synthetic.
- `examples/small.csv`, `examples/stress.csv` are curated synthetic datasets (deterministic, no real-world business data).
- No new files in `DATASETS/` are introduced by this phase. The existing `DATASETS/AUDIT.md` (spec-01 deliverable) continues to govern.
- README does not embed any user-identifying screenshots; the comparison gallery uses only `examples/headline.csv`.

**Status**: PASS (vacuously for new additions; spec-01 audit obligations preserved).

### G8 — Performance Discipline (Principle VIII)

**Pass criterion**: Optimisations justified by measurement. Native extensions out of scope. The polish phase introduces no performance optimisation work; it only adds new code paths (theme application, demo command, dataset generation) whose performance must fit the documented budgets (FR-021 60-second demo budget; ≤30s test suite; etc.).

**How the plan satisfies it**:

- No native extensions, no JIT, no algorithm changes in spec-02 scope.
- Demo command's 60-second budget is enforced by `tests/integration/test_demo_command.py` (timeout fails the test).
- Dataset generator's ≤5-second budget enforced similarly.
- Visualisation theme application is `O(1)` (template assignment); palette lookup is `O(1)` per box (single hash).
- No performance claims in README or docs require a flamegraph; the project's existing complexity-class table (`O(n log n)` per algorithm) is the only performance claim, sourced from the registry per ADR-004.

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

**Overall**: All 8 principles PASS at plan authoring time. No amendments required. No complexity-tracking entries needed.

Phase-specific gates (G9 A11y baseline, G10 Tag ceremony) defined in `Speckit-context-prompts/spec-02-portfolio-polish/02-plan.md` § 4 are tracked at phase sign-off, not constitution-instantiation time.

## Project Structure

### Documentation (this feature)

```text
specs/002-portfolio-polish/
├── plan.md              # This file
├── research.md          # Phase 0 output — 12 LOCKED ADRs + 7 resolved open questions
├── data-model.md        # Phase 1 output — VisualisationStyle, DemoArtifact, HeadlineDatasetSeed + augmented Placement
├── quickstart.md        # Phase 1 output — 5-command reviewer flow
├── contracts/           # Phase 1 output — 6 contract files
│   ├── visualisation-theme.md
│   ├── demo-command.md
│   ├── headline-dataset.md
│   ├── docs-site.md
│   ├── algorithm-card-source.md
│   └── repository-structure.md
├── checklists/
│   └── requirements.md  # spec-quality checklist (from /speckit-specify)
└── tasks.md             # Phase 2 output (created by /speckit-tasks — NOT this command)
```

### Source Code (repository root)

Continuation of the existing single-package layout. Additions only — existing files not shown.

```text
src/bin_packer_3d/
├── visualization/
│   ├── plotter.py               # existing — extended to consume theme + emit static export
│   ├── theme.py                 # NEW — VisualisationStyle + Plotly template constants + apply_theme()
│   └── palette.py               # NEW — colour_for_box() with BLAKE2b → ColorBrewer Set3
├── cli.py                       # existing — adds `demo` Click subcommand (ADR-008)
└── __init__.py                  # existing — re-exports VisualisationStyle, apply_theme

tests/
├── unit/
│   ├── test_visualization_theme.py     # NEW — palette determinism, hover template, template-name assertion
│   ├── test_palette_colourblind.py     # NEW — colorspacious ΔE pass under deuteranopia + protanopia
│   ├── test_demo_command.py            # NEW — Click invocation, output structure, 60s budget
│   └── test_dataset_generator.py       # NEW — same seed → byte-identical CSV; ≥60% BFD utilisation
├── integration/
│   ├── test_visualisation_e2e.py       # NEW — pack → HTML + static PNG byte/pixel-equal across runs
│   ├── test_docs_build.py              # NEW — mkdocs build --strict passes; algorithm-page completeness
│   ├── test_readme_drift.py            # NEW — comparison table + Highlights + structure drift vs registry
│   ├── test_readme_alt_text.py         # NEW — every embedded image has alt text
│   ├── test_structure_drift.py         # NEW — Project Structure block matches filesystem
│   ├── test_highlights_drift.py        # NEW — numeric agreement with live repo state
│   └── test_install_footprint.py       # NEW — pip install size ≤105% of baseline (Gate G7)
├── fixtures/
│   ├── expected/                # NEW — snapshot test inputs (HTML, PNG, JSON)
│   └── install_footprint_baseline.json # NEW — v0.2.0.dev0 install footprint reference
└── property/                            # existing — no spec-02 additions

docs/
├── index.md                     # existing seed — expanded to mkdocs site landing
├── quickstart.md                # existing — updated to use examples/headline.csv
├── maintainers.md               # existing — adds Pages-setup + release-ceremony + snapshot-maintenance sections
├── about.md                     # NEW — canonical project identity (FR-024)
├── algorithms/                  # NEW — one .md per registered packer (registry-driven, ADR-003)
│   ├── index.md                 # NEW — comparison-table landing page
│   ├── bfd.md                   # NEW
│   ├── ffd.md                   # NEW
│   └── shelf.md                 # NEW
├── visualisation.md             # NEW — gallery + theme docs
├── configuration.md             # NEW — full PackerConfig reference
├── troubleshooting.md           # NEW
├── api/
│   └── index.md                 # NEW — mkdocstrings auto-generated
├── promo/                       # NEW — LinkedIn copy, social assets (optional; gitignored if private)
└── assets/                      # NEW — committed binaries
    ├── hero.gif                 # NEW — README hero (vhs-generated; FR-022)
    ├── gallery/                 # NEW — comparison gallery (one PNG per registered algorithm)
    │   ├── bfd.png
    │   ├── ffd.png
    │   └── shelf.png
    └── palette_colourblind_check.png   # NEW — palette verified under deuteranopia/protanopia (FR-037)

examples/                        # NEW at repo root — excluded from wheel via hatch.build.exclude
├── README.md                    # NEW — describes each dataset
├── headline.csv                 # NEW — generated, headline (≥60% BFD utilisation, FR-034)
├── headline.seed                # NEW — JSON seed file for reproducibility
├── small.csv                    # NEW — friendly intro dataset (hand-curated, deterministic)
└── stress.csv                   # NEW — edge case (handful of impossible-to-fit boxes)

scripts/                         # existing
├── audit_datasets.py            # existing (US4 pre-publish gate)
├── generate_headline_dataset.py # NEW — generator for headline.csv (ADR-009)
├── verify_palette_colourblind.py # NEW — produces palette_colourblind_check.png (ADR-012)
├── regenerate_readme.py         # NEW — autogenerates algorithm table + Highlights + Project Structure
├── render_demo_gif.tape         # NEW — vhs Tape script for hero.gif
└── render_demo_gif.py           # NEW — drives vhs (run manually by maintainer)

mkdocs.yml                       # NEW at repo root — MkDocs Material configuration

.github/
└── workflows/
    ├── ci.yml                   # existing — no surgery, calls _ci-core.yml
    ├── _ci-core.yml             # existing — adds docs-build + install-footprint jobs
    └── docs-deploy.yml          # NEW — deploys site to GitHub Pages on push to main (ADR-002)
```

**Structure Decision**: Single-project layout extended additively. No restructuring of `src/bin_packer_3d/`. New top-level `examples/` directory excluded from the wheel via `tool.hatch.build.exclude`. New `docs/` subtree fully populated. `mkdocs.yml` at repo root. The repository-surface disposition table (which top-level items stay, move, or are excluded) is documented in `docs/maintainers.md` § Repository structure and rendered as the README "Project Structure" section by `scripts/regenerate_readme.py`.

## Phased Milestones

Three phases — two for build-out (Phase A scaffolding, Phase B delivery) plus a Phase C release ceremony. Each phase ships a coherent batch; chained PRs apply if any batch exceeds the 400-line review budget (per the `chained-pr` skill in the project's skill registry).

### Phase A — Scaffolding (`v0.3.0-rc0` working state)

**User stories impacted**: foundations for US1, US2, US3, US5; US7 begins.

**Deliverables**:

1. `scripts/generate_headline_dataset.py` authored + `examples/headline.seed` + `examples/headline.csv` committed. `tests/unit/test_dataset_generator.py` green: byte-identical regeneration + ≥60% BFD utilisation on default bin dims (860×890×1040 mm).
2. `src/bin_packer_3d/visualization/theme.py` — `BIN_PACKER_3D_DARK` + `_LIGHT` Plotly templates + `apply_theme()` function. Unit tests assert template-name application + axis-label format.
3. `src/bin_packer_3d/visualization/palette.py` — `colour_for_box()` (BLAKE2b → Set3 index). Unit tests assert determinism + cycling beyond 12 boxes.
4. `scripts/verify_palette_colourblind.py` + `docs/assets/palette_colourblind_check.png` committed. ΔE threshold decided in research.md and documented.
5. `mkdocs.yml` at repo root; `docs/algorithms/{index,bfd,ffd,shelf}.md` scaffolded with placeholder prose + valid YAML front-matter; `docs/about.md` scaffolded.
6. `bin-packer demo` Click subcommand registered (scaffold only; `--help` works, real implementation in Phase B).
7. `pyproject.toml`:
   - `plotly` line: `"plotly>=5.18.0,<6.0.0"` (ADR-001 version pin).
   - New optional groups: `viz = ["kaleido>=0.2.1"]`; `dev` extended with `colorspacious`.
   - `tool.hatch.build.exclude` extended for `examples/`.
8. `tests/fixtures/install_footprint_baseline.json` committed — captured size of `pip install bin-packer-3d` (no extras) on the v0.2.0.dev0 baseline.

**Gate checks before Phase A sign-off**: G2 (test-first), G3 (coverage), G5 (theme + palette determinism), G7 (footprint baseline captured), G8 (`mkdocs build --strict` passes on scaffolds).

### Phase B — Delivery (`v0.3.0-rc0` working state)

**User stories covered**: US1 (README rewrite), US2 (Visualisation polish completed), US3 (Documentation site published), US4 (Repository surface cleanup), US5 (Demo command full), US7 (CV identity), a11y baseline (FR-036/037/038).

**Deliverables**:

1. **Visualisation polish completion** — `plotter.py` extended to call `apply_theme()`, emit static PNG/SVG via Kaleido (guarded by `try: import kaleido`), embed stats overlay annotation. `tests/integration/test_visualisation_e2e.py` green (byte-identical HTML across runs + pixel-equal PNG).
2. **README rewrite**:
   - Hero asset: vhs Tape file at `scripts/render_demo_gif.tape`; `docs/assets/hero.gif` committed.
   - Comparison gallery: `docs/assets/gallery/{bfd,ffd,shelf}.png` generated by `bin-packer demo` on `examples/headline.csv`.
   - Generated comparison table + Highlights section + Project Structure block via `scripts/regenerate_readme.py`. Drift tests green.
   - Problem-statement paragraph, "what this is", "Why I built this", link to docs site.
3. **Docs site**:
   - Algorithm-page prose authored (3 pages: bfd, ffd, shelf; each cites literature, follows FR-013 front-matter contract).
   - `docs/api/index.md` populated via `mkdocstrings`.
   - `docs/visualisation.md`, `docs/configuration.md`, `docs/troubleshooting.md` authored.
   - `docs-deploy.yml` workflow committed; GitHub Pages source set to "GitHub Actions" (one-time manual maintainer setup documented in `docs/maintainers.md`).
   - First deploy succeeds; URL `https://bruno-ghiberto.github.io/3D_BIN_PACKING/` reachable.
4. **Repository surface cleanup**:
   - `LINKEDIN.txt` decision applied (move to `docs/promo/` or remove).
   - README "Project Structure" section regenerated from filesystem; `test_structure_drift.py` green.
   - GitHub About block populated via documented `gh repo edit` command in `docs/maintainers.md`.
5. **Demo command full** — `bin-packer demo` implementation complete; output structure per ADR-008 + `contracts/demo-command.md`; integration test green; 60-second budget verified.
6. **CV identity** — `docs/about.md` populated with ≤80-word paragraph + ≤120-character tagline. README Highlights generated from live state.
7. **A11y completion** — alt text added to every README image; `test_readme_alt_text.py` green; palette verification artefact reaffirmed; mkdocs-material WCAG-AA contrast manually verified and documented in `docs/maintainers.md` § Accessibility verification.

**Gate checks before Phase B sign-off**: G1 (Contract Honesty drift tests green), G2 (test-first), G3 (coverage ≥90%), G4 (CI green including new docs-build + install-footprint jobs), G5 (determinism), G6 (reproducibility — every visual claim regeneratable), G7 (install footprint ≤105% baseline), G8 (docs build + algorithm-page completeness), G9 (a11y baseline complete).

### Phase C — Release Ceremony (~½ day)

**Deliverables**:

1. `hatch version 0.3.0rc1` — bumps `pyproject.toml`.
2. `CHANGELOG.md` — `[0.3.0-rc1]` section authored from `[Unreleased]`.
3. Final pre-merge CI run green (10+ checks including the new ones).
4. PR review + merge to `main` via merge-commit (preserves granular history per Git-Flow-lite).
5. From `main`: `git tag -a v0.3.0-rc1 -m "Portfolio polish release candidate" && git push --tags`.
6. `gh release create v0.3.0-rc1 --draft` with notes file at `specs/002-portfolio-polish/RELEASE_NOTES.md`.
7. Final SC verification: all 14 success criteria measured and recorded in `docs/maintainers.md` § v0.3.0-rc1 release evidence.

**Gate checks before Phase C sign-off**: G10 (tag ceremony — see plan-context-prompt § 4).

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

No violations. All 8 Core Principles PASS at plan authoring time. No constitution amendments required. No complexity-tracking entries needed.

## Risks, Non-Goals & Open Questions

### Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Kaleido accidentally promoted to mandatory dep, breaking FR-027 +5% budget | Medium | High | ADR-001 enforces `[viz]` optional group; `test_install_footprint.py` measures and asserts |
| mkdocs-material WCAG-AA default claim partially false at our contrast level | Low | Medium | Phase A manual verification; documented override path if shortfall surfaces |
| Plotly minor-version palette internals drift → screenshot pixel drift | Medium | Medium | Plotly pinned `>=5.18.0,<6.0.0`; snapshot tests catch drift |
| Procedural dataset generator can't hit ≥60% utilisation on default bin dims | Low | High | research.md Phase 0 evaluates candidates; bin-feasibility-driven recursive guillotine is the fallback (mathematically guaranteed feasible) |
| GitHub Pages first-deploy fails (settings not configured) | Medium | Low | One-time setup documented in `docs/maintainers.md`; first deploy is manual after settings change |
| `colorspacious` not installable on Python 3.14 | Low | Medium | Phase 0 validates install across 3.11–3.14; fallback: hand-curated palette swatch verified manually |
| US5 Part 2 lands mid-spec-02 introducing a new algorithm with no docs page → drift test blocks merge | Low | Low | Spec-02 owner adds a placeholder `docs/algorithms/<new>.md` stub on US5 Part 2 merge; full prose follows in US5 branch |
| `hatch version` misses secondary version-string locations (`__version__` in `__init__.py`) | Low | Low | Phase C ceremony test verifies `bin-packer --version` matches `pyproject.toml` |
| vhs Tape recording non-deterministic (timing-dependent) | Low | Low | Recording maintainer-manual on Linux; GIF committed and treated as frozen artefact |

### Non-Goals (explicit, out of scope)

- TUI (US6) — Deferred to v0.4 per `spec.md § Clarifications`.
- Streamlit / web frontend / hosted live demo beyond GitHub Pages.
- Spanish or other-language documentation.
- Automated WCAG-AA audit (axe-core / Pa11y / Lighthouse a11y) — manual verification only.
- PyPI publication — reserved for stable `v0.3.0` once US5 lands.
- Algorithm correctness improvements (spec-02 §5 out-of-scope).
- US5 Parts 2-3 work.
- Performance optimisation of existing packers.
- Multi-platform CI matrix (Windows, macOS).
- Custom domain for the docs site.
- PR-preview deployments for the docs site.

### Open Questions

All spec-level ambiguities were resolved by `/speckit-clarify` Session 2026-05-13 (5 clarifications recorded in `spec.md § Clarifications`). Plan-level open questions are resolved in `research.md` Phase 0:

1. Kaleido on Python 3.14 install footprint + compatibility — resolved in research.md.
2. `colorspacious` Python 3.14 support — resolved in research.md.
3. mkdocs-material WCAG-AA defaults verification — resolved in research.md (with fallback path documented).
4. Plotly version-pin choice — resolved in research.md (locked at `>=5.18.0,<6.0.0`).
5. Headline dataset generator algorithm choice — resolved in research.md (ADR-009 sub-decision).
6. Colourblind ΔE threshold — resolved in research.md.
7. GitHub Pages first-deploy procedure — resolved in research.md.

No clarification markers leak into this plan.md.

## Artefact Inventory

| Artefact | Path | Phase | Output of |
|---|---|---|---|
| Plan | `specs/002-portfolio-polish/plan.md` | (this) | `/speckit-plan` |
| Research | `specs/002-portfolio-polish/research.md` | Phase 0 | `/speckit-plan` |
| Data Model | `specs/002-portfolio-polish/data-model.md` | Phase 1 | `/speckit-plan` |
| Quickstart | `specs/002-portfolio-polish/quickstart.md` | Phase 1 | `/speckit-plan` |
| Contracts | `specs/002-portfolio-polish/contracts/*.md` (6 files) | Phase 1 | `/speckit-plan` |
| Tasks | `specs/002-portfolio-polish/tasks.md` | Phase 2 | `/speckit-tasks` (not this command) |

---

**End of implementation plan.**
