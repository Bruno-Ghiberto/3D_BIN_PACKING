---
description: "Task list for spec-02: Portfolio Polish of bin-packer-3d"
---

# Tasks: Portfolio Polish of `bin-packer-3d`

**Input**: Design documents from `/specs/002-portfolio-polish/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests**: Strict TDD mode is enabled (per project `CLAUDE.md`). Every new test file MUST be authored + red-verified locally before the implementation commit that turns it green. The `verified red: …` note belongs in the implementation commit body.

**Organization**: Tasks are grouped by user story (US1–US5, US7; US6 Deferred to v0.4 per `spec.md § Clarifications`). Each user story produces an independently testable increment.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: Which user story this task belongs to (US1, US2, US3, US4, US5, US7). Setup, Foundational, and Polish phases have no story label.
- Each task includes the exact file path(s) it touches.

## Path Conventions

- **Library source**: `src/bin_packer_3d/`
- **Tests**: `tests/{unit,integration,property,fixtures}/`
- **Scripts**: `scripts/`
- **Examples**: `examples/` (new in spec-02; excluded from wheel)
- **Docs source**: `docs/` (mkdocs-material site)
- **CI workflows**: `.github/workflows/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Toolchain updates, new optional dependency declarations, Plotly version pin, wheel-exclude updates. None of these touch user-story code — they unblock everything downstream.

- [x] T001 [P] Update `pyproject.toml`: pin Plotly to `"plotly>=5.18.0,<6.0.0"` (ADR-001) in `[project] dependencies`
- [x] T002 [P] Update `pyproject.toml`: add new optional group `viz = ["kaleido>=0.2.1"]` under `[project.optional-dependencies]` (ADR-001)
- [x] T003 [P] Update `pyproject.toml`: extend the existing `dev` extras list under `[project.optional-dependencies]` with `colorspacious` (ADR-012)
- [x] T004 [P] Update `pyproject.toml`: extend `[tool.hatch.build.exclude]` with `"examples/"` so the new `examples/` directory does not ship in the wheel
- [x] T005 [P] Update `.gitignore`: add `examples/output/` (demo command default output; ignored)
- [x] T006 Append a placeholder `[Unreleased]` block to `CHANGELOG.md` for spec-02 entries (entries will be added per task during Phases 3–8)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure every user story depends on — the headline dataset, the visualisation theme, the deterministic palette, the public API re-exports, and the install-footprint baseline. Strict TDD applies: every test file is committed in a red state and verified before the implementation that turns it green.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [x] T007 [P] Create empty `examples/README.md` placeholder (final content added in US5)
- [x] T008 [P] Create empty directory `tests/fixtures/expected/` (will hold visualisation snapshots in US2)
- [x] T009 Capture install-footprint baseline on a clean Python 3.11 venv: `pip install bin-packer-3d` (no extras), measure `du -sb` on `site-packages/bin_packer_3d/` AND on the venv's `site-packages/` total, record both to `tests/fixtures/install_footprint_baseline.json` (depends on T001–T004 being merged so the baseline matches the new pyproject configuration)
- [x] T010 [P] Write red test `tests/integration/test_install_footprint.py` asserting `pip install bin-packer-3d` (no extras) install footprint ≤105% of the value committed in `tests/fixtures/install_footprint_baseline.json` (depends on T009)
- [x] T011 [P] Write red test `tests/unit/test_dataset_generator.py` asserting (a) same seed → byte-identical CSV, (b) generated dataset achieves ≥60% BFD utilisation on default bin dims (860×890×1040 mm), (c) generator completes in ≤5 seconds
- [x] T012 Author `scripts/generate_headline_dataset.py` implementing bin-feasibility-driven recursive guillotine cuts (ADR-009) — turns T011 green
- [x] T013 Author seed file `examples/headline.seed` with documented JSON schema (`seed: 42, target_utilisation: 0.65, bin_dimensions: [860.0, 890.0, 1040.0], n_boxes: 50, min_box_volume_mm3: 50000.0`) — depends on T012
- [ ] T014 Run `python scripts/generate_headline_dataset.py --seed examples/headline.seed --out examples/headline.csv` and commit the result — depends on T012, T013
- [ ] T015 [P] Write red test `tests/unit/test_visualization_theme.py` asserting (a) `apply_theme(fig, "dark")` sets `fig.layout.template` to `BIN_PACKER_3D_DARK`, (b) light variant similarly, (c) axis-label format follows `{name} (mm)` template, (d) `apply_theme` is idempotent
- [ ] T016 Author `src/bin_packer_3d/visualization/theme.py` exporting `VisualisationStyle` (frozen dataclass), `BIN_PACKER_3D_DARK`, `BIN_PACKER_3D_LIGHT`, and `apply_theme(fig, variant="dark")` (ADR-006) — turns T015 green
- [ ] T017 [P] Write red test `tests/unit/test_palette_colourblind.py` asserting (a) `colour_for_box(box_id)` is deterministic across calls, (b) palette indices cycle correctly for >12 unique IDs, (c) pairwise CIELAB ΔE under deuteranopia + protanopia simulation (`colorspacious`) ≥ 15 (ADR-007 + ADR-012)
- [ ] T018 [P] Author `scripts/verify_palette_colourblind.py` that renders the palette under normal + deuteranopia + protanopia simulation via `colorspacious`, computes pairwise ΔE, asserts ≥ 15 threshold, and emits the montage to `docs/assets/palette_colourblind_check.png`
- [ ] T019 Author `src/bin_packer_3d/visualization/palette.py` exporting `SET3: tuple[str, ...]` (12 ColorBrewer Set3 hex strings) and `colour_for_box(box_id, palette=SET3) -> str` using BLAKE2b hash → modulo index (ADR-007) — turns T017 green
- [ ] T020 Run `python scripts/verify_palette_colourblind.py` and commit `docs/assets/palette_colourblind_check.png` — depends on T018, T019
- [ ] T021 [P] Update `src/bin_packer_3d/__init__.py`: re-export `VisualisationStyle`, `BIN_PACKER_3D_DARK`, `BIN_PACKER_3D_LIGHT`, `apply_theme`, `colour_for_box` from the top-level package; extend `__all__`
- [ ] T022 [P] Augment `src/bin_packer_3d/models/placement.py`: add `@cached_property colour(self) -> str` that calls `palette.colour_for_box(self.box.identifier)` with a local import to avoid circular dependency (data-model.md § Augmented)
- [ ] T023 [P] Write `tests/unit/test_placement_colour.py` asserting (a) same placement → same colour across calls, (b) different placement IDs → potentially different colours, (c) the `colour` attribute is NOT serialised in `placements.csv` (existing exporter unchanged)
- [x] T024 Update `_ci-core.yml`: add the new `install-footprint` job invoking `pytest tests/integration/test_install_footprint.py`; wire it into the aggregate `all_passed` output

**Checkpoint**: Foundation ready — every user story may now begin in parallel. Examples directory has `headline.csv` + seed. Theme, palette, and palette-verification artefact in place. Install-footprint guard active.

---

## Phase 3: User Story 1 — First-Screen README Credibility (Priority: P1) 🎯 MVP

**Goal**: Rewrite the README so a 30-second skim communicates the project's problem domain, available algorithms, install command, run command, and one visual of a packed bin. Hero asset (animated GIF), algorithm comparison table (registry-sourced, generated section), "Why I built this" paragraph, screenshot gallery, problem-statement paragraph all present.

**Independent Test**: Render the README on GitHub in a desktop browser. A first-time reader scrolling only the first viewport-and-a-half should within 60 seconds identify: (a) problem domain, (b) the three available algorithms by name, (c) the install command, (d) the run command, (e) at least one visual of a packed bin.

### Tests for User Story 1 (Strict TDD) ⚠️

- [ ] T025 [P] [US1] Write red test `tests/integration/test_readme_drift.py` asserting that `python scripts/regenerate_readme.py` produces no changes (i.e. `git diff --exit-code README.md` is clean after re-run)
- [ ] T026 [P] [US1] Write red test `tests/integration/test_readme_alt_text.py` asserting every embedded image syntax (`![alt](path)`) in `README.md` has a non-empty `alt` field — parse via stdlib `re` (FR-036)
- [ ] T027 [P] [US1] Write red test `tests/integration/test_highlights_drift.py` asserting the regenerated Highlights block matches numerically with live repo state (test count from `pytest --collect-only -q`, CI check count from `_ci-core.yml`, Python versions from `pyproject.toml` classifiers, algorithm count from `ALGORITHMS` registry, license from pyproject)

### Implementation for User Story 1

- [ ] T028 [US1] Author `scripts/regenerate_readme.py` with three generators (algorithm comparison table → ALGORITHMS registry, Highlights section → live repo state per ADR-005, Project Structure section → top-level filesystem scan per FR-016) and CLI argument `--check` for drift-test mode. The script writes between explicit `<!-- BEGIN: SECTION -->` / `<!-- END: SECTION -->` markers (ADR-004) — turns T027 green for highlights once README is rewritten in T031
- [ ] T029 [US1] Author `scripts/render_demo_gif.tape` — a `vhs` Tape script that records the 5-command flow from `quickstart.md` (install + info + pack + demo + open). Tape sized for 800×500 viewport, 8 fps, ~30 seconds total
- [ ] T030 [US1] Maintainer manual step: run `vhs scripts/render_demo_gif.tape -o docs/assets/hero.gif` on Linux; commit `docs/assets/hero.gif`. Documented in `docs/maintainers.md` § Hero asset regeneration (this step is NOT automated in CI — vhs requires a TTY)
- [ ] T031 [US1] Rewrite `README.md`: hero `![Demo of bin-packer-3d packing 50 boxes into a single 860×890×1040 mm bin](docs/assets/hero.gif)` at the top (alt text mandatory per FR-036); problem-statement paragraph (≤200 words plain language); "What this is" paragraph; install one-liner `pip install 'bin-packer-3d[viz]'`; run one-liner `bin-packer demo` (or per-strategy `bin-packer pack`); `<!-- BEGIN: ALGORITHMS_TABLE -->` / `<!-- END: ALGORITHMS_TABLE -->` markers; comparison-gallery section (placeholder paths to `docs/assets/gallery/{bfd,ffd,shelf}.png`, each with alt text); `<!-- BEGIN: HIGHLIGHTS -->` / `<!-- END: HIGHLIGHTS -->` markers; `<!-- BEGIN: PROJECT_STRUCTURE -->` / `<!-- END: PROJECT_STRUCTURE -->` markers; "Why I built this" closing paragraph; link to published docs site
- [ ] T032 [US1] Run `python scripts/regenerate_readme.py` to populate the three generated sections. Commit the populated README — turns T025 green
- [ ] T033 [US1] Generate the three comparison-gallery PNGs by invoking the existing `bin-packer pack` CLI three times against `examples/headline.csv` (one per algorithm) with `--visualize` and Kaleido output. Move each `bin_1.png` to `docs/assets/gallery/<key>.png`. Note: this works once US2 has shipped the Kaleido-static-export branch (T037); if US1 ships first, use a temporary placeholder image and replace it post-US2. (Cross-story note: prefer to ship US1 + US2 in the same PR batch.)
- [ ] T034 [US1] Verify all three drift tests pass: `pytest tests/integration/test_readme_drift.py tests/integration/test_readme_alt_text.py tests/integration/test_highlights_drift.py -v`

**Checkpoint**: README passes the 30-second credibility test. Comparison table + Highlights + Project Structure regenerated from runtime. Hero GIF embeds in GitHub README. SC-001 measurable.

---

## Phase 4: User Story 2 — Polished, Reproducible Visualisations (Priority: P1)

**Goal**: Plotter applies the branded theme (`apply_theme`) and deterministic palette to every emitted figure; embeds a stats overlay panel; emits a static PNG (or SVG) alongside every interactive HTML; produces byte-identical HTML and pixel-equal PNG across runs.

**Independent Test**: Run `bin-packer pack examples/headline.csv --strategy bfd --visualize -o /tmp/test-viz` twice on a fresh checkout. Compare the two `bin_1.html` files byte-for-byte (`cmp` exits 0) and the two `bin_1.png` files byte-for-byte. Open `bin_1.html` in a browser; confirm (a) branded title + axis labels in mm, (b) stats overlay panel reports algorithm + boxes placed + utilisation% + runtime, (c) every box's colour is reproducible.

### Tests for User Story 2 (Strict TDD) ⚠️

- [ ] T035 [P] [US2] Write red test `tests/integration/test_visualisation_e2e.py` asserting (a) two consecutive `bin-packer pack` runs on `examples/headline.csv` with the same `--strategy bfd` produce byte-identical `bin_*.html`, (b) the corresponding `bin_*.png` files are pixel-equal (compare via `hashlib.sha256` of the raw bytes; Kaleido + Plotly are pinned so byte-equality is achievable), (c) the rendered HTML contains the project's branded title format and stats overlay
- [ ] T036 [P] [US2] Generate snapshot fixtures: run `bin-packer pack examples/headline.csv --strategy bfd --visualize -o tests/fixtures/expected/bfd/`; commit `tests/fixtures/expected/bfd/bin_1.html` and `tests/fixtures/expected/bfd/bin_1.png` as the canonical expected output (depends on T037 having shipped)

### Implementation for User Story 2

- [ ] T037 [US2] Extend `src/bin_packer_3d/visualization/plotter.py`: after figure construction, call `apply_theme(fig, variant="dark")`; for each placement, assign `marker.color = placement.colour` (consuming the new `Placement.colour` property from T022); add an `Annotation`-based stats overlay reporting algorithm name, boxes placed (`N/total`), overall utilisation %, and runtime in ms (FR-007, FR-008)
- [ ] T038 [US2] Extend `plotter.py`: add static-export branch guarded by `try: import kaleido` (ADR-001); on success, emit `bin_<N>.png` (default) or `bin_<N>.svg` (when `static_format == "svg"`); on `ImportError`, raise `RuntimeError("Static export requires 'kaleido'. Install with: pip install 'bin-packer-3d[viz]'")` (FR-009, contract `visualisation-theme.md`)
- [ ] T039 [US2] Extend the `bin-packer pack` Click command (in `src/bin_packer_3d/cli.py`) with `--static-format` option (`png` default, `svg` opt-in) — passes through to the plotter's static-export branch
- [ ] T040 [US2] Verify snapshot tests green: `pytest tests/integration/test_visualisation_e2e.py -v` — turns T035 green
- [ ] T041 [US2] Update `CHANGELOG.md` `[Unreleased]` with bullet: "viz: branded Plotly theme, deterministic colours, static export via Kaleido, stats overlay"

**Checkpoint**: Visualisations are reproducible, branded, and exportable to static formats. SC-006 (deterministic colours) measurable. SC-005 (visual claim reproducibility) supported.

---

## Phase 5: User Story 3 — Published Public Documentation Site (Priority: P1)

**Goal**: A `mkdocs-material` documentation site published to GitHub Pages at `https://bruno-ghiberto.github.io/3D_BIN_PACKING/` rebuilds on every push to `main`. Contains: Quickstart, Algorithms (one prose page per registered packer, registry-sourced), API reference (mkdocstrings), Configuration, Visualisation gallery, Troubleshooting, About.

**Independent Test**: Visit the published URL. Homepage loads in ≤3 seconds on a 50 Mb/s connection. Navigation shows all top-level sections. Each algorithm page lists its complexity matching the `ALGORITHMS` registry entry. README contains a prominent link to the site.

### Tests for User Story 3 (Strict TDD) ⚠️

- [ ] T042 [P] [US3] Write red test `tests/integration/test_docs_build.py` asserting (a) `mkdocs build --strict` exits 0, (b) every key in `ALGORITHMS` has a corresponding `docs/algorithms/<key>.md`, (c) each page's YAML front-matter `key:` matches a registry entry, (d) each front-matter `complexity:` matches the registry entry's `complexity_class`

### Implementation for User Story 3

- [ ] T043 [US3] Create `mkdocs.yml` at repo root: Material theme (default palette — preserves WCAG-AA per ADR-002 + ADR-012), `mkdocstrings[python]` plugin, navigation per `plan.md § Project Structure / Documentation layout`
- [ ] T044 [US3] Author `docs/algorithms/bfd.md` with front-matter (`key: bfd`, `complexity: "O(n log n)"`, `citation: "..."`) + prose intro + Complexity reasoning + "When to use" + Reference + embedded `![BFD packing example](../assets/gallery/bfd.png)` (alt text mandatory)
- [ ] T045 [P] [US3] Author `docs/algorithms/ffd.md` (same shape as T044)
- [ ] T046 [P] [US3] Author `docs/algorithms/shelf.md` (same shape as T044)
- [ ] T047 [US3] Author `docs/algorithms/index.md` — landing page for the algorithms section; embeds the same comparison table as the README (regenerated section); add `<!-- BEGIN: ALGORITHMS_TABLE -->` / `<!-- END: ALGORITHMS_TABLE -->` markers; extend `scripts/regenerate_readme.py` (or factor into a shared helper) so the docs algorithm-index table stays in sync with the README table
- [ ] T048 [P] [US3] Author `docs/api/index.md` — a single page with mkdocstrings directives auto-generating API reference from `bin_packer_3d` docstrings
- [ ] T049 [P] [US3] Author `docs/visualisation.md` — explains the theme system, palette, deterministic colour assignment, links to the comparison gallery PNGs, embeds the colourblind-check artefact (`palette_colourblind_check.png`)
- [ ] T050 [P] [US3] Author `docs/configuration.md` — full reference of `PackerConfig` fields (strategy, bin_dimensions, allow_rotation, seed, constraints) with examples and defaults; lists environment-variable overrides
- [ ] T051 [P] [US3] Author `docs/troubleshooting.md` — common install / runtime / visualisation failure modes with documented resolutions (matches the troubleshooting table in `quickstart.md`)
- [ ] T052 [US3] Update existing `docs/quickstart.md` — supersede with the 5-command flow from `specs/002-portfolio-polish/quickstart.md` (install with `[viz]`, info, pack against examples/headline.csv, demo, open viz)
- [ ] T053 [US3] Extend `docs/maintainers.md` with three new sections: § Pages setup (one-time manual: Settings → Pages → Source = "GitHub Actions"), § Accessibility verification (manual mkdocs-material WCAG-AA check procedure per ADR-012), § Snapshot maintenance (procedure for regenerating `tests/fixtures/expected/*` snapshots when intentional theme changes happen)
- [ ] T054 [US3] Author `.github/workflows/docs-deploy.yml` — triggers on `push: branches: [main]`; permissions `pages: write` + `id-token: write` + `contents: read`; jobs `build` (`pip install '.[docs]'` + `mkdocs build --strict` + `actions/upload-pages-artifact@<pinned-sha>`) and `deploy` (`actions/deploy-pages@<pinned-sha>`)
- [ ] T055 [US3] Update `_ci-core.yml`: add a new `docs-build` job invoking `pip install '.[docs]' && mkdocs build --strict`; wire into the aggregate `all_passed` output
- [ ] T056 [US3] Verify docs build test green: `pytest tests/integration/test_docs_build.py -v` — turns T042 green
- [ ] T057 [US3] Update `CHANGELOG.md` `[Unreleased]` with bullet: "docs: published documentation site at github.io"
- [ ] T058 [US3] Maintainer manual step (post-merge): enable GitHub Pages via repo Settings → Pages → Source = "GitHub Actions"; verify first deploy succeeds at `https://bruno-ghiberto.github.io/3D_BIN_PACKING/`. Documented in `docs/maintainers.md` § Pages setup.

**Checkpoint**: Docs site live, registry-driven, WCAG-AA. SC-003 (homepage ≤3s) + SC-004 (100% registered algorithms have docs) measurable. README links to it.

---

## Phase 6: User Story 4 — Tidy Repository Surface and Structure Map (Priority: P1)

**Goal**: Every top-level repository entry is either explained in the README's "Project Structure" section or relocated/excluded. The README block is generated from the actual filesystem and verified by drift test.

**Independent Test**: A first-time reader clones the repo and runs `ls`. Every visible top-level entry maps to one of the categories (library source, tests, docs, examples/data, build/CI, governance/metadata) — verified by inspecting the README "Project Structure" section.

### Tests for User Story 4 (Strict TDD) ⚠️

- [ ] T059 [P] [US4] Write red test `tests/integration/test_structure_drift.py` asserting (a) the README's `<!-- BEGIN: PROJECT_STRUCTURE -->` block matches the regenerator output bytewise, (b) every top-level filesystem entry (excluding `.git/`, hidden VCS / IDE artefacts) is accounted for in the disposition table from `contracts/repository-structure.md`, (c) no orphan files at the repository root

### Implementation for User Story 4

- [ ] T060 [US4] Apply disposition for `LINKEDIN.txt`: move to `docs/promo/linkedin.txt` (kept as maintainer reference, gitignored if private — see Phase 8) OR remove entirely. Whichever is chosen, document the choice in `docs/maintainers.md` § Repository structure
- [ ] T061 [US4] Update `pyproject.toml`'s `[tool.hatch.build.exclude]` list to additionally exclude any newly-introduced root-level paths that should not ship in the wheel (cross-check: `legacy/`, `Speckit-context-prompts/`, `examples/`, `docs/`, `tests/`, `benchmark/`, `DATASETS/*.xlsx` already excluded; verify `LINKEDIN.txt` if retained)
- [ ] T062 [US4] Run `python scripts/regenerate_readme.py` to refresh the Project Structure block from the live filesystem (must run AFTER T060 so the disposition change is reflected)
- [ ] T063 [US4] Extend `docs/maintainers.md` with a new section: § Repository metadata. Document the exact `gh repo edit Bruno-Ghiberto/3D_BIN_PACKING --description ... --homepage ... --add-topic ...` command sequence per `Speckit-context-prompts/spec-02-portfolio-polish/02-plan.md` § 11.2
- [ ] T064 [US4] Maintainer manual step (post-merge): execute the documented `gh repo edit` command to populate the GitHub About block (tagline, ≥3 topics, docs-site homepage). Verify against SC-010
- [ ] T065 [US4] Verify structure-drift test green: `pytest tests/integration/test_structure_drift.py -v` — turns T059 green
- [ ] T066 [US4] Update `CHANGELOG.md` `[Unreleased]` with bullet: "repo: top-level structure cleaned and documented in README"

**Checkpoint**: No mystery top-level entries. README Project Structure block is the canonical reference. GitHub About block populated. SC-011 measurable.

---

## Phase 7: User Story 5 — Curated Examples and One-Command Demo (Priority: P2)

**Goal**: A `bin-packer demo` Click subcommand runs every registered algorithm against `examples/headline.csv` in under 60 seconds, emits per-algorithm HTML + static PNG + `placements.csv` + `summary.json`, and writes a top-level `comparison.md` summary. The `examples/` directory contains 3+ curated datasets with documented intent.

**Independent Test**: Run `bin-packer demo -o /tmp/demo` on a fresh install. The command exits 0 in ≤60 seconds. Output directory has one subdir per registered algorithm key; each contains four expected files; `comparison.md` exists at the root with metrics for every algorithm.

### Tests for User Story 5 (Strict TDD) ⚠️

- [ ] T067 [P] [US5] Write red test `tests/integration/test_demo_command.py` asserting (a) `bin-packer demo -o <tmpdir>` exits 0 in ≤60 seconds (pytest timeout), (b) `<tmpdir>` contains one subdir per `ALGORITHMS` key, (c) each subdir contains `bin_*.html`, `bin_*.png`, `placements.csv`, `summary.json`, (d) `<tmpdir>/comparison.md` exists and references every algorithm

### Implementation for User Story 5

- [ ] T068 [US5] Register the `demo` Click subcommand in `src/bin_packer_3d/cli.py` per ADR-008: `@cli.command()` decorator, three options (`--dataset` default `examples/headline.csv`, `-o/--output-dir` default `examples/output`, `--static-format` choice `[png,svg]` default `png`), docstring "Run every registered algorithm against the headline dataset"
- [ ] T069 [US5] Implement the `demo` command body: iterate `ALGORITHMS.keys()`; for each, load dataset via existing loader, pack via the algorithm, emit per-bin HTML + static export via the visualisation layer (US2 capabilities), write `placements.csv` via existing exporter, write `summary.json` (`DemoArtifact` serialisation per `data-model.md`); finally write `comparison.md` with the metrics table sorted by `bins_used` ascending then `utilisation` descending
- [ ] T070 [P] [US5] Author `examples/small.csv` — a friendly intro dataset (8-15 boxes, easy fit, single bin, clear visualisation for first impression). Hand-curated deterministic CSV; document intent in `examples/README.md` (next task)
- [ ] T071 [P] [US5] Author `examples/stress.csv` — an edge-case dataset (a handful of boxes deliberately too large or odd-shaped so the packer rejects some). Hand-curated deterministic CSV
- [ ] T072 [US5] Update `examples/README.md`: describe each of the three committed datasets (`headline.csv`, `small.csv`, `stress.csv`) — dimensions, intent, expected qualitative result; explain that the demo command defaults to `headline.csv`
- [ ] T073 [US5] Verify demo-command test green: `pytest tests/integration/test_demo_command.py -v` — turns T067 green
- [ ] T074 [US5] Update `CHANGELOG.md` `[Unreleased]` with bullet: "cli: `bin-packer demo` subcommand runs every registered algorithm against a curated headline dataset"

**Checkpoint**: One-command demo works end-to-end. Examples directory curated. SC-002 (≤60s demo, ≥60% BFD utilisation) measurable.

---

## Phase 8: User Story 7 — CV-Facing Project Identity (Priority: P3)

**Goal**: A canonical project-identity file (`docs/about.md`) holds a ≤80-word paragraph + a ≤120-character tagline + a technical-highlights bullet list. The README's "Highlights" section is regenerated from live state. Optional `docs/promo/` directory holds LinkedIn post drafts.

**Independent Test**: Open `docs/about.md`. Within 30 seconds, copy: a ≤80-word paragraph, a one-line tagline, and a bullet list of technical highlights — ready for CV / LinkedIn paste.

### Tests for User Story 7 (Strict TDD) ⚠️

- [ ] T075 [P] [US7] Write red test `tests/unit/test_about_content.py` asserting (a) `docs/about.md` exists, (b) it contains a ≤80-word paragraph block (count words between two specific markdown headings, e.g., `## Project paragraph (≤80 words)`), (c) it contains a tagline of ≤120 characters under a `## Tagline (≤120 chars)` heading, (d) it contains a "Technical highlights" bulleted list of ≥5 entries

### Implementation for User Story 7

- [ ] T076 [US7] Author `docs/about.md` with three sections: `## Project paragraph (≤80 words)` (CV-paste-ready description naming the project, the problem, the algorithms, the engineering practices), `## Tagline (≤120 chars)` (one-line punchy summary suitable for GitHub About + LinkedIn headline), `## Technical highlights` (bulleted list: algorithm count + test count + property-based testing + strict mypy + 10-check CI + supported Python versions + license) — turns T075 green
- [ ] T077 [P] [US7] Author `docs/promo/linkedin-post-draft.md` (optional, maintainer may choose to gitignore via `.gitignore` if private) — a draft LinkedIn post copy citing the project, the release tag, and the docs URL; with a placeholder for the screenshot
- [ ] T078 [US7] Run `python scripts/regenerate_readme.py` to refresh the Highlights block from the live state (the regenerator sources from pyproject + workflows + registry per ADR-005). Drift test (`test_highlights_drift.py` from T027) now passes against final state
- [ ] T079 [US7] Update `CHANGELOG.md` `[Unreleased]` with bullet: "docs: canonical project identity (about.md) + LinkedIn promo draft"

**Checkpoint**: CV-ready paragraph and tagline committed. GitHub About block (T064 from US4) sources from this. SC-012 measurable.

---

## Phase 9: Polish & Release Ceremony

**Purpose**: Final cross-cutting verifications, release version bump, tag, GitHub release draft, success-criterion measurements.

- [ ] T080 [P] Run full test suite + coverage: `pytest --cov=src/bin_packer_3d --cov-fail-under=90 --cov-report=term`. Confirm coverage ≥90% (Gate G3) and total tests ≥ 103 + N (N counts new tests authored in spec-02). Record results in `docs/maintainers.md` § v0.3.0-rc1 release evidence
- [ ] T081 [P] Run install-footprint check on a clean venv: `pip install bin-packer-3d` (no extras), measure, confirm ≤105% of baseline (Gate G7). Record in release-evidence section
- [ ] T082 [P] Run docs site lighthouse-style cold-load timing measurement (open `https://bruno-ghiberto.github.io/3D_BIN_PACKING/` in a fresh browser session on a 50 Mb/s connection; record DOMContentLoaded; assert ≤3s). Record in release-evidence section. (Manual maintainer task — not CI-automated.)
- [ ] T083 [P] Run the full quickstart.md flow on a clean checkout: install with `[viz]` → info → pack → demo → open browser. Verify every command exits 0 and produces the documented output (SC-001 validation). Record any deviations in release-evidence section
- [ ] T084 [P] Run mkdocs-material WCAG-AA manual verification per `docs/maintainers.md` § Accessibility verification (browser DevTools or Lighthouse a11y panel on homepage, an algorithm page, the API reference page; confirm contrast ratios pass AA thresholds for normal + large text). Record outcomes in `docs/maintainers.md` § v0.3.0-rc1 release evidence (Gate G9)
- [ ] T085 Author `specs/002-portfolio-polish/RELEASE_NOTES.md`: opens with a one-paragraph summary of the polish phase; lists each shipped user story (US1, US2, US3, US4, US5, US7) with one-line summary; lists the new CLI surface (`bin-packer demo`, `--static-format` on `pack`), new public API symbols (`VisualisationStyle`, `apply_theme`, `colour_for_box`), new optional dep groups; links to the published docs site and the spec
- [ ] T086 Bump version: `hatch version 0.3.0rc1` (ADR-010). Verifies `pyproject.toml` now declares `version = "0.3.0rc1"`
- [ ] T087 Update `CHANGELOG.md`: move every `[Unreleased]` bullet authored in earlier tasks (T041, T057, T066, T074, T079, T088) into a new `[0.3.0-rc1] — 2026-MM-DD` section; reset `[Unreleased]` to an empty placeholder
- [ ] T088 Commit + open final PR. Verify CI green on every check (lint, format, type, unit, integration, property, docs-build, install-footprint, pre-commit-parity, pip-audit, aggregate)
- [ ] T089 Merge PR to `main` via merge-commit (preserves granular history, per Git-Flow-lite)
- [ ] T090 From `main`: `git checkout main && git pull && git tag -a v0.3.0-rc1 -m "Portfolio polish release candidate" && git push --tags` (Gate G10)
- [ ] T091 Draft GitHub Release: `gh release create v0.3.0-rc1 --draft --title "v0.3.0-rc1 — Portfolio Polish (Release Candidate)" --notes-file specs/002-portfolio-polish/RELEASE_NOTES.md`. Maintainer reviews + publishes
- [ ] T092 Final success-criteria audit: tick every SC-001..SC-014 in `docs/maintainers.md` § v0.3.0-rc1 release evidence with the measured value + link to the artefact that proves it (test run, screenshot, command output, etc.)

**Checkpoint**: spec-02 closed. `v0.3.0-rc1` published. Docs site live. README first-screen credible. All 8 constitutional gates + G9 + G10 pass.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1, T001–T006)**: No dependencies; can start immediately. T001–T005 are all `[P]` (different file scopes within `pyproject.toml` are coordinated via a single commit, but the conceptual tasks are independent).
- **Foundational (Phase 2, T007–T024)**: Depends on Setup completion. Internal dependencies:
  - T009 depends on T001–T004 (pyproject must reflect new extras + Plotly pin before measuring footprint).
  - T010 depends on T009 (test reads the baseline JSON).
  - T012 depends on T011 (TDD: test red → impl green).
  - T013, T014 depend on T012.
  - T016 depends on T015.
  - T019 depends on T017 + T018.
  - T020 depends on T018 + T019.
  - T021, T022 depend on T016 + T019.
  - T024 depends on T009 + T010.
- **User Stories (Phases 3–8)**: All start after Foundational completion. Most stories can proceed in parallel given different developers. Notable inter-story dependencies:
  - **US1 (T033) depends on US2 (T038)** for the Kaleido static-export branch that produces the README gallery PNGs. Recommended: ship US1 + US2 in the same PR batch. Workaround if US1 needs to ship first: commit placeholder PNGs and replace post-US2.
  - **US3 (T044, T045, T046)** embeds the gallery PNGs from US2 — same dependency. The algorithm-page docs depend on US2's PNGs being committed under `docs/assets/gallery/`.
  - **US4 (T062)** depends on US1 (T028) — the regenerator script is authored in US1.
  - **US7 (T078)** depends on US1 (T028) — same regenerator.
  - US5 is fully independent — depends only on Foundational.
- **Polish (Phase 9, T080–T092)**: Depends on all user stories complete. T080–T085 can run in parallel; T086–T092 are sequential (version bump → changelog → PR → merge → tag → release → audit).

### Within Each User Story

- Strict TDD applies: every test task is committed in a red state and is verified before the implementation task that turns it green. The verification line goes in the implementation commit body (e.g., `verified red: pytest tests/integration/test_readme_drift.py exited 1 with 1 expected failure before this commit`).
- Models / scripts before consumers (e.g., regenerator T028 before the README rewrite T031 that depends on its output).
- Story complete before moving to next priority.

### Parallel Opportunities

- **Setup phase**: T001–T005 all `[P]` (independent edits to pyproject.toml + gitignore).
- **Foundational phase**: T007, T008, T010, T011, T015, T017, T018, T021, T022, T023 are all `[P]` — different files, no inter-dependencies.
- **US1 internal**: T025, T026, T027 (the three test files) can run in parallel before any implementation.
- **US2 internal**: T035 + T036 are parallel (different fixtures).
- **US3 internal**: T044, T045, T046 (three algorithm pages) + T048, T049, T050, T051 (four other docs pages) are all `[P]`. T054 (workflow) and T055 (CI core extension) can also run in parallel.
- **US5 internal**: T070, T071 (curated CSVs) are parallel.
- **Polish phase**: T080–T084 (measurements + audits) all `[P]`.

### Cross-Story Parallelism

Once Foundational completes (T024 green), user stories can be assigned in parallel to different developers:

- Developer A: US1 (README rewrite) + dependent US4 (structure cleanup) + US7 (CV identity)
- Developer B: US2 (visualisation polish) — unblocks Developer A's gallery work mid-stream
- Developer C: US3 (docs site)
- Developer D: US5 (demo + examples)

Each story checkpoint is an independent integration point.

---

## Parallel Example: Foundational Phase

```bash
# After Setup completes (T001–T006 merged), launch foundational tests + scripts in parallel:
Task: "Write tests/unit/test_dataset_generator.py (red)"     # T011
Task: "Write tests/unit/test_visualization_theme.py (red)"   # T015
Task: "Write tests/unit/test_palette_colourblind.py (red)"   # T017
Task: "Author scripts/verify_palette_colourblind.py"         # T018

# Once T011/T015/T017 are committed red, launch the implementations sequentially per file:
# (Tests for different modules are still parallel: T012 + T016 + T019 don't share files.)
Task: "Author scripts/generate_headline_dataset.py"          # T012 (green for T011)
Task: "Author src/bin_packer_3d/visualization/theme.py"      # T016 (green for T015)
Task: "Author src/bin_packer_3d/visualization/palette.py"    # T019 (green for T017)
```

## Parallel Example: User Story 3 (Docs Site)

```bash
# All authoring tasks for separate docs files are parallel:
Task: "Author docs/algorithms/bfd.md"           # T044
Task: "Author docs/algorithms/ffd.md"           # T045
Task: "Author docs/algorithms/shelf.md"         # T046
Task: "Author docs/api/index.md (mkdocstrings)" # T048
Task: "Author docs/visualisation.md"            # T049
Task: "Author docs/configuration.md"            # T050
Task: "Author docs/troubleshooting.md"          # T051
```

---

## Implementation Strategy

### MVP First (US1 + US2 + US3 + US4 — the P1 bundle)

The four P1 stories ship together as the credibility minimum for `v0.3.0-rc1`. Recommended order within the bundle:

1. Complete Phase 1 (Setup) + Phase 2 (Foundational).
2. Phase 4 (US2 — Visualisation polish) FIRST among the P1 stories — its Kaleido static-export branch unblocks US1's gallery PNGs and US3's algorithm-page screenshots.
3. Phase 3 (US1 — README rewrite) — depends on US2's PNGs.
4. Phase 5 (US3 — Docs site) — depends on US2's PNGs.
5. Phase 6 (US4 — Repo surface) — depends on US1's regenerator script.
6. **STOP and VALIDATE**: Confirm SC-001, SC-002, SC-003, SC-004, SC-005, SC-006, SC-011 measurable.
7. Open the P1-bundle PR; merge; if shipping at this point is the chosen scope, proceed to Phase 9 release ceremony.

### Incremental Delivery (P1 → P2 → P3)

After the P1 bundle merges and is testable as an MVP:

1. Phase 7 (US5 — Demo + examples) — independent of all P1 stories given Foundational; ships as the second batch.
2. Phase 8 (US7 — CV identity) — small batch; could merge with P2 or as its own.
3. Phase 9 (release ceremony) only after every desired story is complete.

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together (forced bottleneck — everything else depends on this).
2. Once Foundational is green:
   - Developer A: US2 → US1 → US4 (sequential chain due to gallery dependency).
   - Developer B: US3 (docs site) — independent.
   - Developer C: US5 (demo) — independent.
   - Developer D: US7 (CV identity) — independent.
3. Stories complete and integrate via separate PRs against `develop`.

---

## Notes

- **Strict TDD verification**: Every `green` implementation commit body must include the literal line `verified red: pytest <test_path> exited 1 with N expected failures before this commit` (substituting the actual test path and failure count). This is the project's commit-body convention per the constitution + project `CLAUDE.md`.
- **No `Co-Authored-By` lines**: Per project `CLAUDE.md`, never add `Co-Authored-By:` or AI attribution to commits.
- **Conventional commits**: All commits MUST follow Conventional Commits format (e.g., `feat(viz): add branded theme + deterministic palette`, `test(docs): assert algorithm-page completeness`).
- **PR strategy**: Per the `chained-pr` skill in the project skill registry, chained PRs apply if any batch exceeds 400 lines. Phase B (US1 + US2 + US3 + US4) is likely large enough to warrant chained PRs; sub-batch in the order recommended in the Implementation Strategy.
- **US5 cross-track collision avoidance**: spec-02 MUST NOT touch `src/bin_packer_3d/algorithms/extreme_point.py`, `…/maximal_rectangles.py`, or `src/bin_packer_3d/benchmark/runner.py`. The first two don't yet exist (US5 Part 2 creates them); the third is owned by the parallel US5 branch. If US5 Part 2 merges to `develop` mid-spec-02, registry-driven sourcing (FR-004, FR-013) auto-incorporates the new algorithm into the README table, docs site, and demo command output — but a corresponding `docs/algorithms/<new>.md` stub MUST be added (otherwise drift test T042 blocks the merge).
- **Manual steps are documented, not automated**: T030 (vhs hero recording), T058 (one-time Pages source setting), T064 (gh repo edit for About block), T084 (manual WCAG verification), T086–T092 (release ceremony) are all maintainer-manual steps. They are tracked as tasks in `tasks.md` for completeness but their execution is not part of CI.
- **Avoid same-file conflicts**: Tasks marked `[P]` operate on distinct files. The few coordination points (e.g., multiple tasks modifying `pyproject.toml` in Setup) are intentionally bundled into a single commit at sign-off so they don't conflict.
