# Plan Context — Phase 02 Plan: Portfolio Polish of `bin-packer-3d`

> Hand this document AND `specs/002-portfolio-polish/spec.md` to `speckit.plan`
> as the planning input. The spec is the WHAT (38 FRs · 14 SCs · 7 user stories;
> US6 Deferred). This document tells the plan phase HOW the maintainer wants
> the WHAT delivered: technology choices already made, architecture preferences
> ratified as ADRs, sequencing hints, gates, contracts, testing strategy, and
> the operational plan for the `v0.3.0-rc1` release ceremony.
>
> **Input spec**: `specs/002-portfolio-polish/spec.md` (fully clarified — 0
> outstanding `[NEEDS CLARIFICATION]` markers; 5 decisions recorded in the
> spec's § Clarifications section, dated 2026-05-13).
> **Constitution**: `.specify/memory/constitution.md` (v1.0.1, 8 principles,
> 4 NON-NEGOTIABLE).
> **Reference plan**: `specs/001-public-release-hardening/plan.md` — use as
> a structural blueprint; this plan inherits its phasing style and ADR shape.

---

## 0. Brief

Take `bin-packer-3d` at `v0.2.0.dev0` — a working library with three
registered algorithms (`bfd`, `ffd`, `shelf`), 103 passing tests, a 10-check
CI pipeline, `mypy --strict` clean, and MIT license — and produce a
presentation-layer polish pass that ships under tag `v0.3.0-rc1`. The
maintainer (sole: Bruno Ghiberto) wants the repository, its generated
artefacts, and its on-screen experience to read as credible portfolio work
to three reader profiles: a recruiter who scans the README for 30 seconds,
a senior engineer who spends 5–10 minutes on README plus source, and an
operations-research practitioner who spends 10–20 minutes on the docs site.

This is a **level-up pass on a working codebase**, not feature completion.
US5 Parts 2-3 (Extreme Point + Maximal Rectangles + benchmark CI gate)
continues on its parallel branch (`feature/us5-extreme-point-benchmark`).
The polish pass MUST NOT block, depend on, or anticipate that work.

The plan MUST consume the post-clarify spec verbatim. Five clarifications
are locked (see spec § Clarifications, Session 2026-05-13):

1. **TUI Deferred** to a v0.4 candidate; US6 closed; FR-035 a future-phase
   placeholder. No TUI dependencies, no TUI tests, no TUI entry point in
   spec-02.
2. **Headline dataset procedurally generated** (FR-034) — commit a generator
   script + seed file + the resulting CSV; regenerating from the committed
   seed yields byte-identical output. Literature-cited datasets (BR1..BR8)
   are reserved for US5.
3. **End-of-phase version `v0.3.0-rc1`** (FR-033) — bump `pyproject.toml`
   to `0.3.0rc1` and cut a git tag at PR merge. No PyPI publication. The
   stable `v0.3.0` tag is reserved for when US5 Parts 2-3 also land.
4. **README gallery is comparison-style** (FR-005) — one curated headline
   dataset packed by every registered algorithm, side by side. Visual
   difference between images = algorithmic difference, directly readable.
5. **A11y baseline** (FR-036/037/038) — alt text on every README image,
   colourblind-safe visualisation palette verified against
   deuteranopia/protanopia simulation, mkdocs-material WCAG-AA defaults
   preserved in the docs site. Full automated WCAG audit deferred.

The plan MUST NOT reopen these decisions.

---

## 1. Primary Input References

The `/speckit.plan` invocation MUST read all four inputs before emitting
any artefact. Do not paraphrase or summarise — use the source documents
directly.

| Artefact | Path | Role |
|---|---|---|
| Feature spec | `specs/002-portfolio-polish/spec.md` | 38 FRs · 14 SCs · 7 user stories — authoritative scope |
| Constitution | `.specify/memory/constitution.md` | 8 principles (4 NON-NEGOTIABLE) — quality thresholds |
| Plan template | `.specify/templates/plan-template.md` | Output structure — every placeholder must be resolved |
| Reference plan | `specs/001-public-release-hardening/plan.md` | Structural blueprint + ADR style |

**Environment baseline (maintainer machine + CI)**:

- Python: maintainer develops on 3.14.4 (CPython); public floor `>=3.11`; CI matrix 3.11 / 3.12 / 3.13 / 3.14 on `ubuntu-latest`
- OS: Linux x86_64 (Fedora 43, kernel 6.19)
- Package manager: `pip` (uv compatible; no lockfile committed)
- VCS: Git; remote at `github.com/Bruno-Ghiberto/3D_BIN_PACKING`
- Strict TDD mode is **enabled** (per project `CLAUDE.md`): Red → confirm red → Green → Refactor, with red verified locally before green ships. Every new test file MUST land in a commit that pairs (or precedes) the implementation it covers, with the verification noted in the commit body.

---

## 2. Technical Context

Fills the `[NEEDS CLARIFICATION]` placeholders in
`.specify/templates/plan-template.md` § Technical Context verbatim.

**Language/Version**: Python 3.11 (public floor, existing); maintainer develops on 3.14.4; CI matrix 3.11 / 3.12 / 3.13 / 3.14 on Linux.

**Primary Dependencies** (existing — NO new mandatory runtime deps in this phase, per Principle V):

- Runtime: `numpy >= 1.24`, `plotly >= 5.18` (to be pinned more tightly — see ADR-001), `pydantic >= 2`, `pydantic-settings >= 2`, `click >= 8.1`, `pandas >= 2.0`, `openpyxl >= 3.1`, `rich >= 13.0`
- Dev (existing): `pytest`, `pytest-cov`, `pytest-mock`, `mypy`, `ruff`, `pre-commit`, `pandas-stubs`, `hypothesis`, `pip-audit`, `pyyaml`
- Docs extra (existing optional): `mkdocs-material >= 9.5`, `mkdocstrings[python] >= 0.25`

**New optional dependencies introduced in this phase** (all under `[project.optional-dependencies]` to preserve FR-027 +5% install-footprint budget):

- `kaleido` — Plotly static-export backend; group `viz` (ADR-001).
- `colorspacious` — Python colourblind-simulation library; group `dev` (ADR-012).

**Storage**: no database. Files only. Generated visualisation HTML + static PNG/SVG written to user-specified output directories. The procedurally generated headline dataset and curated examples committed under `examples/`. Demo command output written to `examples/output/` (gitignored).

**Testing**: existing pytest (102+1 passing, hypothesis property suite). Strict TDD mode enabled. New polish-phase test categories:

- Reproducibility tests (FR-023 → SC-005)
- Determinism tests (SC-006)
- Drift tests (FR-004, FR-013, FR-014, FR-016, FR-026)
- A11y baseline tests (FR-036 → SC-014)
- Docs-build integration test (FR-011, FR-014)

**Target Platform**: Linux x86_64 (CI authoritative). macOS / Windows best-effort — pure-Python install, but not CI-verified at `v0.3.0-rc1`.

**Project Type**: pure-Python library with CLI entry point `bin-packer`. New polish artefacts (visualisation theme, demo command, dataset generator, headline dataset, docs site, README rewrite) ship in the same package + repository. No new entry points beyond a `bin-packer demo` subcommand registered with Click (ADR-008).

**Performance Goals**:

- Demo command end-to-end runtime: ≤60 seconds on commodity laptop (FR-021).
- Docs site cold-load homepage: ≤3 seconds on a 50 Mb/s connection (SC-003).
- Interactive visualisation HTML file size: ≤4.8 MB per bin (FR-010 baseline).
- Test suite runtime: ≤30 seconds on Linux CI (existing budget; MUST NOT regress).

**Constraints**:

- Zero new mandatory runtime dependencies. All polish deps under optional groups.
- `pip install bin-packer-3d` (no extras) install footprint ≤105% of the v0.2.0.dev0 baseline (FR-027 + SC-007).
- Cross-Python compat: 3.11–3.14 (CI matrix).
- Constitution: all 8 Core Principles apply (4 NON-NEGOTIABLE). Contract Honesty (I) and Documentation (VI) are load-bearing for this phase.
- US5 cross-track: spec-02 MUST NOT modify `src/bin_packer_3d/algorithms/extreme_point.py`, `…/maximal_rectangles.py` (created by US5), or the benchmark engine code (`src/bin_packer_3d/benchmark/runner.py`).
- 103-passing test suite MUST NOT regress.
- Coverage on `src/bin_packer_3d/` MUST remain ≥90% (constitution §III).

**Scale/Scope**: single maintainer. Typical reviewer reads README on GitHub then optionally clones. Demo command runs on ≤200 boxes against a single bin (default 860×890×1040 mm). Docs site rebuilds in ≤30 seconds.

---

## 3. Project Structure Decision

**Selected option**: Continuation of the existing single-package layout from
`001-public-release-hardening`. No restructuring of `src/bin_packer_3d/`.
New polish-phase artefacts are added additively across `src/`, `tests/`,
`docs/`, `examples/` (new), `scripts/` (existing), and `.github/workflows/`.

### Source layout (additions only — existing files not shown)

```text
src/bin_packer_3d/
├── visualization/
│   ├── plotter.py               # existing — extended to consume theme + emit static export
│   ├── theme.py                 # NEW — VisualisationStyle dataclass + Plotly template authoring
│   └── palette.py               # NEW — deterministic colour assignment (hash → palette index)
├── cli.py                       # existing — adds `demo` subcommand
└── __init__.py                  # existing — re-exports VisualisationStyle, apply_theme

tests/
├── unit/
│   ├── test_visualization_theme.py     # NEW — palette determinism, hover template
│   ├── test_palette_colourblind.py     # NEW — deuteranopia/protanopia ΔE pass
│   ├── test_demo_command.py            # NEW — Click invocation, output structure
│   └── test_dataset_generator.py       # NEW — seed → byte-identical CSV
├── integration/
│   ├── test_visualisation_e2e.py       # NEW — pack → HTML + static export reproducible
│   ├── test_docs_build.py              # NEW — mkdocs build --strict passes; algo pages complete
│   ├── test_readme_drift.py            # NEW — comparison table + highlights drift vs registry
│   ├── test_readme_alt_text.py         # NEW — every embedded image has alt text
│   ├── test_structure_drift.py         # NEW — Project Structure section matches filesystem
│   └── test_highlights_drift.py        # NEW — numeric agreement with live repo state
└── property/                            # existing — no new property tests in spec-02

docs/
├── index.md                     # existing seed — expanded to mkdocs site landing
├── quickstart.md                # existing — updated to use examples/headline.csv
├── maintainers.md               # existing — adds Pages-setup + release-ceremony sections
├── about.md                     # NEW — canonical project identity (FR-024)
├── algorithms/                  # NEW — one .md per registered packer (registry-driven)
│   ├── index.md                 # NEW — comparison-table landing page
│   ├── bfd.md                   # NEW
│   ├── ffd.md                   # NEW
│   └── shelf.md                 # NEW
├── visualisation.md             # NEW — gallery + theme docs
├── configuration.md             # NEW — full PackerConfig reference
├── troubleshooting.md           # NEW
├── api/
│   └── index.md                 # NEW — mkdocstrings auto-generated
├── promo/                       # NEW — LinkedIn copy, optional artefacts (gitignored if private)
└── assets/                      # NEW — committed binaries
    ├── hero.gif                 # NEW — README hero recording (vhs-generated)
    ├── gallery/                 # NEW — comparison gallery (one PNG per registered algorithm)
    │   ├── bfd.png
    │   ├── ffd.png
    │   └── shelf.png
    └── palette_colourblind_check.png   # NEW — palette verified under deuteranopia/protanopia

examples/                        # NEW at repo root — excluded from wheel
├── README.md                    # NEW — describes each dataset
├── headline.csv                 # NEW — generated, headline (≥60% BFD utilisation)
├── headline.seed                # NEW — JSON seed file for reproducibility
├── small.csv                    # NEW — friendly intro dataset (hand-curated, deterministic)
└── stress.csv                   # NEW — edge case (handful of impossible-to-fit boxes)

scripts/                         # existing
├── audit_datasets.py            # existing (US4 pre-publish gate)
├── generate_headline_dataset.py # NEW — generator for headline.csv (ADR-009)
├── verify_palette_colourblind.py # NEW — produces palette_colourblind_check.png (ADR-012)
├── regenerate_readme.py         # NEW — autogenerates comparison table + highlights + structure
└── render_demo_gif.py           # NEW — drives vhs to produce hero.gif (run manually)

mkdocs.yml                       # NEW at repo root — MkDocs Material configuration

.github/
└── workflows/
    ├── ci.yml                   # existing — no surgery, calls _ci-core.yml
    ├── _ci-core.yml             # existing — adds docs-build + drift-test jobs
    └── docs-deploy.yml          # NEW — deploys site to GitHub Pages on push to main
```

### Top-level repository disposition (FR-016, FR-017)

| Entry | Today | Disposition | Why |
|---|---|---|---|
| `src/` | source | stays at root | library citizenship |
| `tests/` | tests | stays at root | conventional |
| `docs/` | docs seed | stays at root, expanded | site source |
| `examples/` | (NEW) | added at root | curated demo data |
| `scripts/` | existing | stays at root | maintainer tooling |
| `DATASETS/` | mixed legacy + audit | stays at root, documented in README | data archive; loader still reads from here for backwards compat |
| `legacy/` | preserved Alpha scripts | stays at root, documented in README "Project Structure" | Library Citizenship — preserved for audit trail; excluded from wheel |
| `Speckit-context-prompts/` | planning prompts | stays at root, documented in README "Project Structure" | maintainer artefacts (visible signal of SDD discipline); excluded from wheel |
| `LINKEDIN.txt` | one-off scratch | **removed** or moved under `docs/promo/` | does not earn root placement |
| `benchmark/` | empty placeholder from US5 Part 1 | stays; managed by US5 track | scope crossover — no spec-02 modifications |
| `.specify/` | SDD config | stays at root | SpecKit machinery |
| `.serena/` | Serena config | stays at root | tooling |

**Structure Decision**: Single-project layout extended additively. Hatch
`tool.hatch.build.exclude` updated to include `examples/`, `LINKEDIN.txt`
(if retained), and any new top-level artefacts not part of the wheel.

---

## 4. Constitutional Gates

Each gate maps to a constitution principle. Gates MUST be checked before
each phase sign-off. Failed gates block progression.

| Gate | Principle | Pass criterion | Fail action |
|---|---|---|---|
| **G1 — Contract Honesty (registry sync)** | I. Contract Honesty (NON-NEG) | README comparison table, docs `/algorithms/<key>.md` pages, and CLI `bin-packer info` output ALL source from `ALGORITHMS` registry; CI drift-test blocks merge if any disagree with runtime | Block merge |
| **G2 — Test-First** | II. Test-First (NON-NEG) | Every new test file authored + red-verified locally before its implementation commit; verification noted in commit body | Block PR review |
| **G3 — Coverage floor** | III. Quality Gates (NON-NEG) | `pytest --cov=src/bin_packer_3d --cov-fail-under=90` passes on merge commit | Block merge |
| **G4 — CI green** | III. Quality Gates (NON-NEG) | All 10+ CI checks pass (existing + new docs-build + new drift-tests) | Block merge |
| **G5 — Determinism** | IV. Reproducibility | Two consecutive runs on identical input produce byte-identical HTML and pixel-equal static PNG | Block release |
| **G6 — Reproducibility** | IV. Reproducibility | Every visual claim in README is reproducible via a documented command (FR-023, SC-005); CI test exercises each | Block merge |
| **G7 — Install footprint** | V. Library Citizenship | `pip install bin-packer-3d` (no extras) install footprint ≤105% of v0.2.0.dev0 baseline; measured on a clean CI matrix run | Block release |
| **G8 — Docs gate** | VI. Documentation (NON-NEG) | `mkdocs build --strict` exits 0; every `ALGORITHMS` registry key has a corresponding `docs/algorithms/<key>.md` (FR-014) | Block merge |
| **G9 — A11y baseline** | VI. Documentation (NON-NEG, extended) | 100% of README-embedded images have alt text (FR-036); palette colourblind-safe verification artefact committed (FR-037); mkdocs-material WCAG-AA contrast verified (FR-038) | Block release |
| **G10 — Tag ceremony** | (release) | `pyproject.toml` reads `version = "0.3.0rc1"`; git tag `v0.3.0-rc1` exists on merge commit; release notes drafted citing spec-02 user stories | Required for release |

---

## 5. Architectural Decisions (LOCKED — to be ratified as ADRs in research.md)

These decisions are LOCKED by the maintainer in this plan-context document.
SpecKit MUST record each as an ACCEPTED ADR in
`specs/002-portfolio-polish/research.md` with status `ACCEPTED`. Do not
propose alternatives or invite further user input.

---

### ADR-001 — Plotly static-export backend + version pin

**Decision**: **Kaleido**, declared under `[project.optional-dependencies]`
group `viz`. **Plotly pinned** to `>=5.18.0,<6.0.0` to lock major-version
behaviour and minimise risk of palette-internals drift between minor versions.

**Rationale**: Kaleido is the canonical Plotly static-export backend
(cross-platform, deterministic byte output). Its install footprint (~50 MB)
would breach FR-027's +5% budget if added to mandatory deps, so it MUST be
optional. The Plotly upper bound on the major version locks colour-handling
internals; a future Plotly 6.x upgrade is a separate spec.

**Implementation scope**:

- `pyproject.toml` gains `[project.optional-dependencies]` group `viz = ["kaleido>=0.2.1"]`.
- `plotly` line updated: `"plotly>=5.18.0,<6.0.0"`.
- `src/bin_packer_3d/visualization/plotter.py` static-export branch:
  - `try: import kaleido` guarded; on `ImportError`, emit a clear error message: `"Static export requires 'kaleido'. Install with: pip install 'bin-packer-3d[viz]'"`.
- README install snippet shows both forms: `pip install bin-packer-3d` (no static) and `pip install 'bin-packer-3d[viz]'` (with static).

---

### ADR-002 — Documentation framework, host, and deploy mechanism

**Decision**: **MkDocs Material** for build; **GitHub Pages** for hosting;
**GitHub-official `actions/deploy-pages`** for deployment (NOT
`peaceiris/actions-gh-pages`).

**Rationale**: `mkdocs-material` is already declared as an optional
dependency and its default theme is documented WCAG-AA conformant. GitHub
Pages is the zero-infrastructure choice. The official `actions/deploy-pages`
action integrates with the Pages "GitHub Actions" source mode without
maintaining a `gh-pages` branch, which keeps the repo tree cleaner.

**Implementation scope**:

- `mkdocs.yml` at repo root with the Material theme, `mkdocstrings[python]`
  plugin, and the navigation specified in §3.
- Workflow `docs-deploy.yml`:
  - Trigger: `push: branches: [main]`.
  - Permissions: `pages: write`, `id-token: write`, `contents: read`.
  - Jobs: `build` (runs `mkdocs build --strict`, uploads `site/` as a Pages artefact) and `deploy` (uses `actions/deploy-pages`).
- One-time maintainer setup documented in `docs/maintainers.md` § Pages setup:
  - GitHub repo Settings → Pages → Source = "GitHub Actions".

---

### ADR-003 — Algorithm-page strategy (docs/algorithms/)

**Decision**: **Hybrid** — `mkdocstrings` for API reference page;
**hand-written prose** for each `docs/algorithms/<key>.md` page; **drift
test** asserts each registry key has a matching `.md` file and that the
`.md` front-matter's `key:` field matches a registry entry.

**Rationale**: `mkdocstrings` alone produces dry, docstring-derived pages
that don't convey intuition for OR practitioners. Hand-written prose adds
the operations-research framing (complexity reasoning, typical-use-case,
literature citation). The drift test enforces registry-doc agreement
(Principle I) without requiring full autogeneration.

**Implementation scope**:

- Each `docs/algorithms/<key>.md` page MUST have YAML front-matter:
  ```yaml
  ---
  key: bfd                       # MUST match an ALGORITHMS registry key
  complexity: "O(n log n)"
  citation: "Crainic et al., 2008"
  ---
  ```
- `tests/integration/test_docs_build.py`:
  - Assert every key in `ALGORITHMS` has a corresponding `docs/algorithms/<key>.md`.
  - Assert each front-matter `key:` field is a valid registry entry.
  - Assert front-matter `complexity:` matches the registry entry's `complexity_class`.

---

### ADR-004 — Comparison-table sync (README)

**Decision**: **Generated-section pattern** — README contains a block
between explicit markers:

```markdown
<!-- BEGIN: ALGORITHMS_TABLE (auto-generated by scripts/regenerate_readme.py; do not edit) -->
... regen output ...
<!-- END: ALGORITHMS_TABLE -->
```

A `scripts/regenerate_readme.py` script reads `ALGORITHMS` and overwrites
the block. A CI test (`tests/integration/test_readme_drift.py`) runs the
script and asserts the README has not changed (`git diff --exit-code`).

**Rationale**: Principle I (Contract Honesty) demands runtime is the truth.
A drift test alone allows local divergence between commits; generation
forces alignment at every commit. The markered-block pattern is industry
standard (e.g., `<!-- automd:badges -->`).

**Implementation scope**:

- `scripts/regenerate_readme.py` is the single source for table generation; it can also regenerate the Highlights section (ADR-005) and the Project Structure tree (FR-016).
- README is initially authored with placeholder markers; the script populates them.
- Pre-commit hook (optional) runs the script on README edits.

---

### ADR-005 — Highlights-section sync (README)

**Decision**: **Generated-section pattern**, same shape as ADR-004. Sources:

- Algorithm count: `len(ALGORITHMS)`
- Test count: parsed from `pytest --collect-only -q | tail -1`
- CI check count: parsed from `.github/workflows/_ci-core.yml` `jobs:` keys
- Supported Python versions: parsed from `pyproject.toml` `classifiers`
- License: parsed from `pyproject.toml` `license`
- Property-based-testing presence: presence of `tests/property/` directory
- Strict-mypy presence: presence of `[tool.mypy] strict = true` (it's a global toggle in pyproject already)

**Rationale**: Same Contract Honesty anchor as ADR-004. FR-026 explicitly
calls drift between the Highlights numbers and the live repo state a
Contract Honesty violation.

**Implementation scope**:

- `scripts/regenerate_readme.py` extends to populate the Highlights block too.
- `tests/integration/test_highlights_drift.py` asserts post-regen diff is empty.

---

### ADR-006 — Visualisation theme implementation pattern

**Decision**: **Plotly figure template** (`go.layout.Template`) authored
once in `src/bin_packer_3d/visualization/theme.py`. Two variants: `dark`
and `light`. Apply via `fig.update_layout(template="bin_packer_3d_dark")`.
Per-call overrides are limited to run-specific data (figure title, stats
overlay values).

**Rationale**: Clean separation of concerns. Theme is a single source of
truth. Light/dark variants without code duplication. Plotly's template
mechanism is the idiomatic way to brand figures.

**Implementation scope**:

- `theme.py` exports:
  - `BIN_PACKER_3D_DARK: go.layout.Template`
  - `BIN_PACKER_3D_LIGHT: go.layout.Template`
  - `apply_theme(fig: go.Figure, variant: Literal["dark", "light"] = "dark") -> go.Figure`
- `plotter.py` calls `apply_theme(fig)` after figure construction; theme controls colours, axis labels (mm units), hover template, stats-overlay layout.

---

### ADR-007 — Deterministic colour assignment

**Decision**: **Stable hash of box identifier → palette index**. Hash
function: **BLAKE2b** (stdlib `hashlib.blake2b`, 16-bit digest sufficient
for indexing). Base palette: **ColorBrewer "Set3"** (12 colours,
qualitative, documented colourblind-friendly). For inputs with >12 boxes,
cycle the palette via modulo indexing. The colourblind-safety claim MUST
be re-verified empirically (ADR-012) — ColorBrewer's documentation rates
Set3 as "colourblind safe" with caveats; the verification artefact is the
ground truth.

**Rationale**: BLAKE2b is deterministic across Python versions (no
hash-randomisation interference). ColorBrewer palettes are widely cited
for cartographic / visualisation work. The empirical verification (ADR-012)
keeps the spec honest — "colourblind-safe" is not a fact we assert without
evidence.

**Implementation scope**:

- `palette.py`:
  - `SET3: tuple[str, ...]` — hex strings, 12 entries from ColorBrewer Set3.
  - `colour_for_box(box_id: str, palette: tuple[str, ...] = SET3) -> str`:
    1. Compute `digest = hashlib.blake2b(box_id.encode("utf-8"), digest_size=2).digest()`.
    2. `index = int.from_bytes(digest, "big") % len(palette)`.
    3. Return `palette[index]`.
- Test: same box ID → same colour; collisions exist (palette has 12 slots) but are stable.

---

### ADR-008 — Demo command surface

**Decision**: **`bin-packer demo`** — a new Click subcommand registered
alongside the existing `pack | info | init`. Output directory defaults to
`examples/output/` (gitignored). Optional `--dataset PATH` argument
overrides the headline dataset.

**Rationale**: Consistent with the existing CLI pattern. Reuses the
existing packer / loader / metrics code without duplication. No new tool
surface (no Makefile, no standalone script). Iterates `ALGORITHMS` registry,
honouring the "every registered algorithm" semantics of FR-020.

**Implementation scope**:

- `cli.py` gains:
  ```python
  @cli.command()
  @click.option("--dataset", type=click.Path(exists=True), default="examples/headline.csv")
  @click.option("-o", "--output-dir", type=click.Path(), default="examples/output")
  @click.option("--static-format", type=click.Choice(["png", "svg"]), default="png")
  def demo(dataset: Path, output_dir: Path, static_format: str) -> None:
      """Run every registered algorithm against the headline dataset."""
  ```
- Output structure:
  ```text
  examples/output/
  ├── bfd/
  │   ├── bin_1.html
  │   ├── bin_1.png
  │   ├── placements.csv
  │   └── summary.json
  ├── ffd/
  │   └── ... (same shape)
  ├── shelf/
  │   └── ...
  └── comparison.md            # per-algorithm metrics table
  ```
- 60-second budget enforced via `tests/integration/test_demo_command.py`.

---

### ADR-009 — Headline dataset generator approach

**Decision**: **Deterministic seeded generator**. Concrete algorithm is
chosen during the plan's `research.md` Phase 0 investigation; the OUTPUT
CONTRACT is fixed here. Candidate algorithms include (a) bin-feasibility-
driven recursive guillotine cuts of a target-utilisation packed bin, (b)
rejection sampling from a parameter distribution, (c) parameter-driven
synthetic generator adapted from Martello-Vigo. The plan picks one with
justification.

**Output contract** (LOCKED):

- Script path: `scripts/generate_headline_dataset.py`.
- Input: a seed file (`examples/headline.seed`) containing
  `{seed: int, target_utilisation: float, bin_dimensions: [l,w,h], n_boxes: int, min_box_volume_mm3: float}`.
- Output: `examples/headline.csv` matching `DATASETS/sample_boxes.csv` columns (`ITEM,W,H,L,CANTIDAD,CAJA,DESCRIPCION`).
- Byte-identical regeneration: same seed → same CSV bytes (verified by `tests/unit/test_dataset_generator.py`).
- Acceptance: generated dataset packed by BFD on default bin dims (860×890×1040 mm) MUST yield overall utilisation ≥60% (verified by integration test).
- Pure-Python implementation; no native deps.

**Rationale**: Multiple valid generator algorithms exist; the maintainer
prefers the plan phase choose one with explicit research-backed reasoning
rather than the plan-context-prompt prescribing a specific algorithm. The
contract (deterministic, ≥60% BFD util, byte-identical regen) is the
load-bearing constraint.

---

### ADR-010 — Versioning workflow + tag ceremony

**Decision**: **`hatch version 0.3.0rc1`** for the `pyproject.toml` bump;
**manual `git tag -a v0.3.0-rc1`** on the merge commit; release notes
drafted from `CHANGELOG.md` `[Unreleased]` entries via `gh release create
--draft`. No PyPI publish.

**Rationale**: Hatch is the existing build backend (`[build-system]
requires = ["hatchling"]`). `hatch version` updates `pyproject.toml`
cleanly without sed/manual-edit risk. Manual tag preserves the lightweight
release flow established by the v0.1.0 / v0.2.0 history. PyPI publishing
is FR-033-scoped out.

**Implementation scope**:

- Phase C ceremony (§6 below) details the exact command sequence.
- `docs/maintainers.md` § Release ceremony updated.

---

### ADR-011 — Reproducibility test pattern (snapshot tests)

**Decision**: **Snapshot tests** for visual-claim reproducibility (FR-023
→ SC-005). Expected outputs stored under `tests/fixtures/expected/<test
_name>.{html,png,json}`. Compare bytes for HTML and JSON; pixel-equal for
PNG (Kaleido + Plotly are pinned per ADR-001, so pixel-equality is
achievable). Snapshots regeneratable via `pytest --snapshot-update`
(gated to maintainer-only via documented procedure; CI never updates).

**Rationale**: Strongest guarantee for Principle IV (Reproducibility).
Snapshot bytes are the actual artefact, not a proxy.

**Implementation scope**:

- `tests/fixtures/expected/` directory populated by Phase A.
- `pytest-snapshot` is not added as a dep — snapshots compared with stdlib `filecmp` and `hashlib` (pure-Python, zero dep cost).
- Documented regeneration procedure in `docs/maintainers.md` § Snapshot maintenance.

---

### ADR-012 — Accessibility verification tooling

**Decision**: **`colorspacious`** Python library for colourblind palette
verification (deuteranopia + protanopia simulation). Pure-Python, supports
3.11–3.14, declared under `[project.optional-dependencies]` group `dev`.

Verification script `scripts/verify_palette_colourblind.py`:

1. Render each palette colour as a sample swatch.
2. Apply `colorspacious` simulation for deuteranopia and protanopia.
3. Compute pairwise ΔE distance between adjacent palette entries under each simulation.
4. Assert minimum ΔE ≥ a documented threshold (e.g., 15 in CIELAB; threshold chosen during research.md Phase 0).
5. Emit `docs/assets/palette_colourblind_check.png` — a montage showing the palette under normal vision, deuteranopia simulation, and protanopia simulation.

`mkdocs-material` WCAG-AA verification: **manual** for this phase.
Documented procedure in `docs/maintainers.md` § Accessibility verification
(check the default theme's contrast ratios with a browser tool against
WCAG-AA criteria; record any overrides). Full automated audit (axe-core,
Pa11y) is explicitly deferred per spec § Assumptions.

**Rationale**: `colorspacious` is mature, well-cited, lightweight. Pure-
Python keeps the cross-Python compat trivial. Manual mkdocs-material
verification is acceptable because the default theme is documented as
AA-conformant; spec-02 doesn't introduce theme overrides, so risk is low.

---

## 6. Phased Implementation Strategy

Three phases — two for build-out (Phase A scaffolding, Phase B delivery)
plus a Phase C release ceremony. Each phase ships a coherent batch; chained
PRs if any batch exceeds the 400-line review budget (per `chained-pr` skill
in the project's skill registry).

---

### Phase A — Scaffolding (`v0.3.0-rc0` working state; ~1 calendar week)

**User stories impacted**: foundations for US1, US2, US3, US5, US7.

**Deliverables**:

1. `scripts/generate_headline_dataset.py` authored; `examples/headline.seed` + `examples/headline.csv` committed; `tests/unit/test_dataset_generator.py` green (byte-identical regen; ≥60% BFD utilisation).
2. `src/bin_packer_3d/visualization/theme.py` — `BIN_PACKER_3D_DARK` + `_LIGHT` templates + `apply_theme()`. Unit tests pass.
3. `src/bin_packer_3d/visualization/palette.py` — `colour_for_box()` with BLAKE2b + ColorBrewer Set3. Unit tests pass.
4. `scripts/verify_palette_colourblind.py` + `docs/assets/palette_colourblind_check.png` committed. ΔE threshold documented.
5. `mkdocs.yml` at repo root; `docs/algorithms/{index,bfd,ffd,shelf}.md` scaffolded with placeholder prose; `docs/about.md` scaffolded.
6. `bin-packer demo` Click subcommand registered (scaffold only; `--help` works, real implementation in Phase B).
7. `pyproject.toml`:
   - `plotly` line: `"plotly>=5.18.0,<6.0.0"`
   - new optional groups: `viz = ["kaleido>=0.2.1"]`, extends `dev` with `colorspacious`
   - `tool.hatch.build.exclude` extended for `examples/` (so it doesn't ship in wheel).

**Gate checks before Phase A sign-off**: G2 (test-first), G3 (coverage), G5 (theme determinism), G7 (footprint baseline taken — actual measurement happens in Phase B), G8 (`mkdocs build --strict` passes on scaffolds).

---

### Phase B — Delivery (`v0.3.0-rc0` working state; ~2 calendar weeks)

**User stories covered**: US1 (README rewrite), US2 (Visualisation polish completed), US3 (Documentation site published), US4 (Repository surface cleanup), US5 (Demo command full), US7 (CV identity).

**Deliverables**:

1. **Visualisation polish completion** — `plotter.py` calls `apply_theme()`, emits static export via Kaleido (per `--static-format` flag), embeds stats overlay annotation. Snapshot tests in `tests/integration/test_visualisation_e2e.py` green.
2. **README rewrite**:
   - Hero asset: vhs Tape file authored at `scripts/render_demo_gif.tape`; `scripts/render_demo_gif.py` invokes vhs; `docs/assets/hero.gif` committed.
   - Comparison gallery: `docs/assets/gallery/{bfd,ffd,shelf}.png` generated by demo command on `examples/headline.csv`.
   - Generated comparison table + Highlights section + Project Structure block via `scripts/regenerate_readme.py`. Drift tests green.
   - Problem-statement paragraph, "what this is", "Why I built this", links to docs site.
3. **Docs site**:
   - Algorithm-page prose authored (3 pages, each citing literature).
   - `docs/api/index.md` populated with mkdocstrings.
   - `docs/visualisation.md` (gallery + theme overview), `docs/configuration.md`, `docs/troubleshooting.md`.
   - `docs-deploy.yml` workflow committed; GitHub Pages source set to "GitHub Actions" (one-time manual setup documented).
   - First deploy succeeds; URL `https://bruno-ghiberto.github.io/3D_BIN_PACKING/` reachable.
4. **Repository surface cleanup**:
   - `LINKEDIN.txt` decision applied (move to `docs/promo/` or remove).
   - README "Project Structure" section regenerated from filesystem.
   - GitHub About block populated via documented `gh repo edit` command in `docs/maintainers.md` § Repository metadata.
5. **Demo command full** — `bin-packer demo` implementation complete; output structure as ADR-008; integration test green; 60-second budget verified.
6. **CV identity** — `docs/about.md` populated with ≤80-word paragraph + ≤120-char tagline. README Highlights generated.
7. **A11y completion** — alt text added to every README image; `test_readme_alt_text.py` green; palette verification artefact reaffirmed.
8. **Drift tests** — `test_readme_drift.py`, `test_highlights_drift.py`, `test_structure_drift.py`, `test_docs_build.py` all green.

**Gate checks before Phase B sign-off**: G1, G2, G3, G4, G5, G6, G7 (final footprint measurement), G8, G9.

---

### Phase C — Release Ceremony (~½ day)

**Deliverables**:

1. `hatch version 0.3.0rc1` (bumps `pyproject.toml`).
2. `CHANGELOG.md` — `[0.3.0-rc1]` section authored from `[Unreleased]`.
3. Final pre-merge CI run green (10+ checks).
4. PR review + merge to `main` via merge-commit (preserving granular history per the Git-Flow-lite branching strategy).
5. From `main`: `git tag -a v0.3.0-rc1 -m "Portfolio polish release candidate" && git push --tags`.
6. `gh release create v0.3.0-rc1 --draft --title "v0.3.0-rc1 — Portfolio Polish (Release Candidate)" --notes-file specs/002-portfolio-polish/RELEASE_NOTES.md`.
7. Final SC verification: all 14 success criteria measured + recorded in `docs/maintainers.md` § v0.3.0-rc1 release evidence.

**Gate checks before Phase C sign-off**: G10.

---

## 7. Data Model Outline

Fills `data-model.md`. Carries forward all existing entities from spec-01;
only new and modified entities are described here.

### Carried forward (no spec-02 changes)

`Box`, `Bin`, `Placement`, `PackerConfig`, `PackingResult`, `LoadReport`,
`RejectedRow`, `ColumnMapping`, `AlgorithmMetadata`, `BenchmarkResult`,
`ConstraintVisitor` and its concretions, `StructuredAdapter`.

### New entities

**`VisualisationStyle`** (`src/bin_packer_3d/visualization/theme.py`):

```python
@dataclass(frozen=True)
class VisualisationStyle:
    name: Literal["bin_packer_3d_dark", "bin_packer_3d_light"]
    palette: tuple[str, ...]            # 12 hex strings, ColorBrewer Set3
    axis_label_format: str              # "{name} (mm)"
    hover_template: str                 # Plotly hover string with %{customdata.*} fields
    title_format: str                   # "{algorithm} — {dataset}"
    stats_overlay_layout: dict[str, Any]
```

**`DemoArtifact`** (per-algorithm output record, in-memory only — not
persisted as a Python object; manifest written to `summary.json`):

```python
@dataclass(frozen=True)
class DemoArtifact:
    source_dataset: Path
    algorithm: str
    html_paths: tuple[Path, ...]        # one per bin used
    static_export_paths: tuple[Path, ...] # one per bin, format determined by --static-format
    placements_csv: Path
    summary_json: Path
    metrics: PackingResult              # reuses existing summary type
    timestamp: datetime
```

**`HeadlineDatasetSeed`** (anchors the seed-file format; not a Python
import — JSON on disk at `examples/headline.seed`):

```json
{
  "seed": 42,
  "target_utilisation": 0.65,
  "bin_dimensions": [860.0, 890.0, 1040.0],
  "n_boxes": 50,
  "min_box_volume_mm3": 50000.0
}
```

### Augmented existing entities

- `Placement`: GAINS a derived `colour: str | None` attribute, computed
  via `palette.colour_for_box(box.identifier)`. Computed lazily; not
  persisted in `placements.csv`. Plotter reads this for figure rendering.

---

## 8. Quickstart Contract

Fills `quickstart.md`. The reviewer-facing flow MUST work verbatim after
`pip install 'bin-packer-3d[viz]'` (the `[viz]` extra is what unlocks
static export; the basic install still works without it).

```bash
# 1. Install with viz extras (Kaleido for static export + interactive HTML)
pip install 'bin-packer-3d[viz]'

# 2. Verify CLI + see registered algorithms
bin-packer info

# 3. Pack the bundled headline dataset with BFD (the strongest of the three)
bin-packer pack examples/headline.csv --strategy bfd --visualize -o /tmp/pack-bfd

# 4. Run the demo command — exercise all three registered algorithms on the same data
bin-packer demo -o /tmp/demo

# 5. Open the generated visualisation in your browser
xdg-open /tmp/demo/bfd/bin_1.html   # Linux
# open /tmp/demo/bfd/bin_1.html      # macOS
# start /tmp/demo/bfd/bin_1.html     # Windows
```

Each command MUST:

- Exit 0 on a clean install (no missing deps, no import errors).
- Produce its primary output (terminal table, or HTML + PNG files) within ≤60 seconds.
- Print `--help` output that names the purpose, required flags, and at least one example.

`quickstart.md` includes the five commands with annotated expected output
snippets (text only — no embedded screenshots, for reproducibility).

---

## 9. Public Contracts Surface

Fills `contracts/`. Enumerates polish-phase public surfaces. Each contract
file documents the stability promise and breaking-change policy.

### `contracts/visualisation-theme.md`

- Public exports from `bin_packer_3d.visualization`:
  - `VisualisationStyle` (frozen dataclass)
  - `BIN_PACKER_3D_DARK` (constant)
  - `BIN_PACKER_3D_LIGHT` (constant)
  - `apply_theme(fig: go.Figure, variant: Literal["dark", "light"] = "dark") -> go.Figure`
- Static export emits per-bin PNG (default) or SVG (`--static-format svg`).
- Stable across Plotly minor versions (Plotly pinned per ADR-001).
- Breaking-change policy: theme dict shape change OR palette identity change is a major version bump.

### `contracts/demo-command.md`

- Invocation: `bin-packer demo [-o OUTPUT_DIR] [--dataset PATH] [--static-format png|svg]`.
- Defaults: output dir `examples/output/`, dataset `examples/headline.csv`, static format `png`.
- Iterates `ALGORITHMS` registry; emits one subdir per registered key.
- Per-algorithm subdir contents: `bin_*.html`, `bin_*.{png,svg}`, `placements.csv`, `summary.json`.
- Repo-root output: `comparison.md` (per-algorithm metrics table in Markdown).
- 60-second total budget enforced; integration test fails if exceeded.

### `contracts/headline-dataset.md`

- Generator: `scripts/generate_headline_dataset.py` (not part of the wheel).
- Invocation: `python scripts/generate_headline_dataset.py --seed examples/headline.seed --out examples/headline.csv`.
- Seed file format (JSON): see §7 above.
- Output CSV columns (header row): `ITEM,W,H,L,CANTIDAD,CAJA,DESCRIPCION` (matching `DATASETS/sample_boxes.csv`).
- Byte-identical regeneration guarantee under the same seed.
- Generated dataset MUST yield ≥60% BFD utilisation on default bin dims.

### `contracts/docs-site.md`

- URL: `https://bruno-ghiberto.github.io/3D_BIN_PACKING/` (default GitHub Pages — custom domain out of scope).
- Top-level sections: Quickstart · Algorithms · Visualisation · Configuration · API Reference · Troubleshooting · About.
- Algorithm pages: 1 per `ALGORITHMS` key; each MUST have YAML front-matter `key:`, `complexity:`, `citation:` matching registry data.
- Build: `mkdocs build --strict` (zero warnings).
- Deploy: GitHub-official `actions/deploy-pages` on push to `main`.

### `contracts/algorithm-card-source.md`

- Source of truth: `ALGORITHMS` registry in `src/bin_packer_3d/algorithms/__init__.py`.
- Consumers (ALL MUST source from registry, not hardcoded):
  - README comparison table (generated section, ADR-004)
  - CLI `bin-packer info` (existing)
  - Docs site `docs/algorithms/index.md` table (regenerated by docs-build, OR included as a fragment from the same regenerator)
  - Docs site per-algorithm pages (via drift-test enforcement, ADR-003)
- Tests assert all consumers agree on every CI run.

### `contracts/repository-structure.md`

- README "Project Structure" section is regenerated from the actual filesystem on every `scripts/regenerate_readme.py` invocation.
- Drift test (`test_structure_drift.py`) asserts post-regen diff is empty.
- Top-level disposition table (§3 above) is the canonical reference; the README section is its rendered form.

---

## 10. Testing Strategy

Strict TDD enforced (project `CLAUDE.md` flag, ratified by Constitution §II
NON-NEG). Every new test file authored + red-verified locally before the
implementation commit that turns it green. Verification noted in commit
body.

### Test pyramid (additions to existing suite)

```
                ┌──────────────────┐
                │   Property       │  existing — no spec-02 additions
                │                  │
                ├──────────────────┤
                │   Integration    │  NEW: visualisation_e2e, docs_build,
                │   (~70 % polish) │  readme_drift, alt_text, structure_drift,
                │                  │  highlights_drift, demo_command
                ├──────────────────┤
                │   Unit           │  NEW: theme, palette_colourblind,
                │   (~30 % polish) │  dataset_generator
                └──────────────────┘
```

### Test categories for polish phase

1. **Theme + palette unit tests** (`tests/unit/test_visualization_theme.py`)
   - Same input → same colour assignment (BLAKE2b stable).
   - Palette index cycles correctly for >12 boxes.
   - `apply_theme()` returns the figure with the expected template name set.

2. **Colourblind verification test** (`tests/unit/test_palette_colourblind.py`)
   - `colorspacious` ΔE distance under deuteranopia and protanopia ≥ documented threshold.
   - Verification artefact (`docs/assets/palette_colourblind_check.png`) is regeneratable byte-equal.

3. **Demo command integration test** (`tests/integration/test_demo_command.py`)
   - `bin-packer demo` exits 0 in ≤60s on fresh install.
   - Output dir contains one subdir per registered algorithm.
   - Each subdir has the four expected files (`bin_*.html`, `bin_*.{png,svg}`, `placements.csv`, `summary.json`).
   - `comparison.md` at output root exists and references every algorithm.

4. **Headline dataset generator test** (`tests/unit/test_dataset_generator.py`)
   - Same seed → byte-identical CSV output.
   - Generated dataset achieves ≥60% BFD utilisation on default bin dims.
   - Generator completes in ≤5s.

5. **Visualisation reproducibility test** (`tests/integration/test_visualisation_e2e.py`)
   - Two runs on identical input produce byte-identical HTML.
   - Static-export PNG is pixel-equal across runs (Kaleido + Plotly pinned).
   - Snapshots in `tests/fixtures/expected/`.

6. **Docs build test** (`tests/integration/test_docs_build.py`)
   - `mkdocs build --strict` exits 0.
   - Every `ALGORITHMS` key has a corresponding `docs/algorithms/<key>.md`.
   - Each algorithm page front-matter `key:` matches an existing registry entry.
   - Each front-matter `complexity:` matches the registry entry's `complexity_class`.

7. **README drift test** (`tests/integration/test_readme_drift.py`)
   - Run `scripts/regenerate_readme.py`; assert `git diff --exit-code README.md` is clean.

8. **Highlights drift test** (`tests/integration/test_highlights_drift.py`)
   - Regenerated Highlights block matches numerically with live repo state (test count, CI check count, etc.).

9. **A11y alt-text test** (`tests/integration/test_readme_alt_text.py`)
   - Every embedded image in README has a non-empty alt text. Parse markdown via `re` (no new dep needed for simple Markdown image syntax).

10. **Repository structure drift test** (`tests/integration/test_structure_drift.py`)
    - Regenerated "Project Structure" block matches actual top-level filesystem layout.

### Coverage threshold (Gate G3)

`pytest --cov=src/bin_packer_3d --cov-fail-under=90` (constitution §III floor).

### Pre-commit parity

Existing `pre-commit-parity` job in `_ci-core.yml` continues to enforce
CI-local parity. New hooks in `.pre-commit-config.yaml` (optional, for
maintainer convenience): one that runs `scripts/regenerate_readme.py` on
README edits, gated to `manual` stage.

---

## 11. Operational Plans

### 11.1 Documentation deployment (US3 → FR-011 → ADR-002)

- One-time maintainer manual action (documented in `docs/maintainers.md` § Pages setup):
  1. GitHub repo Settings → Pages → Source = **GitHub Actions** (not "Deploy from a branch").
  2. No custom domain (out of scope for spec-02).
- CI workflow `docs-deploy.yml`:
  - Trigger: `push: branches: [main]`.
  - Permissions: `pages: write`, `id-token: write`, `contents: read`.
  - Jobs:
    - `build` — `pip install '.[docs]'` + `mkdocs build --strict` + `actions/upload-pages-artifact`.
    - `deploy` — `actions/deploy-pages` (needs: `build`).
- Build-failure semantics: a failing `mkdocs build --strict` blocks deploy (Gate G8); the live site does NOT roll back automatically — it remains at the last successful build.

### 11.2 GitHub About block (US4 → FR-018)

Manual configuration via `gh repo edit`, documented in
`docs/maintainers.md` § Repository metadata:

```bash
gh repo edit Bruno-Ghiberto/3D_BIN_PACKING \
  --description "3D Bin Packing solver — three O(n log n) heuristic algorithms with interactive 3D visualisation" \
  --homepage "https://bruno-ghiberto.github.io/3D_BIN_PACKING/" \
  --add-topic "bin-packing" \
  --add-topic "3d-packing" \
  --add-topic "operations-research" \
  --add-topic "python" \
  --add-topic "optimisation" \
  --add-topic "logistics"
```

The description text is sourced from `docs/about.md` (single source of
truth — no drift between About block, README hero, and CV paragraph).

### 11.3 Release ceremony (Phase C → FR-033 → ADR-010)

```bash
# 0. On feature/002-portfolio-polish (or final-batch branch):

# 1. Bump version
hatch version 0.3.0rc1

# 2. Update CHANGELOG.md — move [Unreleased] entries to a new [0.3.0-rc1] section
$EDITOR CHANGELOG.md

# 3. Commit + push + open PR
git add pyproject.toml CHANGELOG.md
git commit -m "chore(release): bump to v0.3.0-rc1"
git push -u origin HEAD

# 4. Open final PR → review → CI green → merge to develop (Git-Flow-lite)
# 5. PR develop → main → merge with merge-commit (preserves granular history)

# 6. From main:
git checkout main && git pull
git tag -a v0.3.0-rc1 -m "Portfolio polish release candidate"
git push --tags

# 7. Draft GitHub Release
gh release create v0.3.0-rc1 --draft \
  --title "v0.3.0-rc1 — Portfolio Polish (Release Candidate)" \
  --notes-file specs/002-portfolio-polish/RELEASE_NOTES.md
```

`RELEASE_NOTES.md` template lives at the spec folder; the plan SHOULD
generate it as part of Phase C tasks.

### 11.4 US5 cross-track integration (parallel-branch hygiene)

- `feature/us5-extreme-point-benchmark` is the parallel track for the OR algorithm work.
- spec-02 branch (`002-portfolio-polish`) is cut from `develop` after the latest US5 Part 1 merge (commit `08ca725` or later).
- spec-02 MUST NOT touch:
  - `src/bin_packer_3d/algorithms/extreme_point.py` (US5 creates)
  - `src/bin_packer_3d/algorithms/maximal_rectangles.py` (US5 creates)
  - `src/bin_packer_3d/benchmark/runner.py` (US5 implements; spec-02 only relies on scaffolding from US5 Part 1)
- If US5 Part 2 lands mid-spec-02:
  - The registry-driven sourcing (FR-004, FR-013, ADR-004, ADR-007) automatically incorporates the new algorithm into the README comparison table, docs site, and demo command output.
  - No spec-02 amendments required; drift tests will catch any consumer that still hardcodes the algorithm list.
- Integration via `develop` per Git-Flow-lite. The polish PR merges to `develop`; final develop→main rollup at release time.

---

## 12. Risks, Non-Goals & Open Questions

### Risks (with mitigations)

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Kaleido adds runtime weight that breaks FR-027 +5% budget if accidentally promoted to mandatory dep | Medium | High | ADR-001 enforces `[viz]` optional group; CI test (`test_install_footprint.py`) measures `pip install bin-packer-3d` (no extras) size and asserts ≤105% baseline |
| mkdocs-material's "WCAG-AA default" claim is partially false at the contrast level we use | Low | Medium | Phase A research task verifies manually; documented override path in `docs/maintainers.md` if shortfall surfaces |
| Plotly minor-version palette internals drift between 5.x releases → screenshot pixel drift | Medium | Medium | Plotly pinned `>=5.18.0,<6.0.0`; upgrade-test procedure documented; snapshot tests catch drift on every CI run |
| Procedural dataset generator cannot achieve ≥60% utilisation on default bin dims with a tractable algorithm | Low | High | Phase 0 research evaluates 2–3 candidate algorithms; bin-feasibility-driven recursive guillotine is the fallback (mathematically guaranteed feasible) |
| GitHub Pages first-deploy fails because "Pages source = GitHub Actions" not yet set | Medium | Low | One-time maintainer manual setup documented in `docs/maintainers.md` § Pages setup; first deploy is a manual trigger after settings change |
| `colorspacious` not installable on Python 3.14 | Low | Medium | Phase 0 research task validates install across 3.11–3.14; fallback: snapshot-only verification with a hand-curated palette swatch |
| US5 Part 2 lands mid-spec-02 and surfaces an algorithm whose docs page is missing → drift test blocks merge | Low | Low | Phase B integration adds a placeholder `docs/algorithms/<new>.md` stub immediately on US5 Part 2 merge; full prose follows in the US5 branch |
| `hatch version` bumps cause secondary version-string changes the maintainer misses (e.g. `__version__` in `__init__.py`) | Low | Low | Phase C ceremony test verifies `bin-packer --version` matches `pyproject.toml` value |
| vhs Tape recording produces non-deterministic GIF (timing-dependent) | Low | Low | Recording is maintainer-manual on Linux; GIF committed to repo and treated as a frozen artefact, not regenerated per CI |

### Non-Goals (explicit, out of scope)

- TUI (US6) — Deferred to v0.4 per spec § Clarifications.
- Streamlit / web frontend / hosted live demo beyond GitHub Pages.
- Spanish or other-language documentation.
- Automated WCAG-AA audit (axe-core / Pa11y / Lighthouse a11y) — manual verification only this phase.
- PyPI publication — reserved for stable `v0.3.0` once US5 lands.
- Algorithm correctness improvements (out of spec, per §5 of context prompt).
- US5 Parts 2-3 work.
- Performance optimisation of existing packers.
- Multi-language CI matrix (Windows, macOS) — pure-Python install is best-effort, not CI-verified.
- Custom domain for the docs site.
- PR-preview deployments for the docs site.

### Open Questions

All spec-level ambiguities were resolved by `/speckit-clarify` Session
2026-05-13 (5 clarifications, recorded in spec § Clarifications). The
following PLAN-level open questions are surfaced explicitly for `/speckit-
plan`'s Phase 0 research:

1. **Kaleido on Python 3.14** — verify install footprint (target: ≤55 MB unpacked on a default Kaleido install) and 3.14 compatibility (Kaleido pre-built wheels available?).
2. **`colorspacious` Python 3.14 support** — verify install + import on the matrix.
3. **mkdocs-material WCAG-AA defaults** — manually verify current state at target version; document any overrides if shortfalls surface.
4. **Plotly version-pin choice** — pick the minor version with the longest expected colour-internals stability. Candidates: `>=5.18.0,<6.0.0` (broad), `>=5.24.0,<5.25.0` (narrow, latest verified at writing time). Decide in Phase 0.
5. **Headline dataset generator algorithm choice** — evaluate three candidates (bin-feasibility-driven recursive guillotine; rejection sampling from a parameter distribution; adapted Martello-Vigo). Pick one and justify; record as a sub-ADR in `research.md`.
6. **Colourblind ΔE threshold** — settle on a CIELAB ΔE minimum for adjacent palette entries under deuteranopia/protanopia simulation (literature suggests ΔE ≥ 10 for noticeable difference; ≥ 15 for "clear separation"). Pick ≥ 15 unless research surfaces a better number.
7. **GitHub Pages first-deploy procedure** — confirm exact one-time settings; document in `docs/maintainers.md` § Pages setup.

No clarification markers should leak into the resulting `plan.md`. All
seven items above are research-resolvable (no further user input expected).

---

## 13. Branch, Commit, Integration

- **Branch**: `002-portfolio-polish`, cut from `develop` (not from `main`), per the existing Git-Flow-lite branching strategy established by spec-01.
- **Commit format**: Conventional Commits. Per project `CLAUDE.md`: **NEVER** add `Co-Authored-By:` or AI attribution to commits.
- **PR strategy**: Per the `chained-pr` skill in the project's skill registry, chained PRs apply if any batch exceeds the 400-line review budget. Phases A / B / C each ship as their own PR ideally; if Phase B exceeds budget, sub-batch it (e.g., visualisation polish → README rewrite → docs site → demo command → CV identity).
- **Final PR**: closes spec-02; cuts `v0.3.0-rc1` tag per the Phase C ceremony.
- **Strict TDD posture**: Every new test file's red state is verified locally and noted in the commit body (e.g., "verified red: pytest tests/unit/test_visualization_theme.py exited 1 with 3 expected failures before this commit"). Green commit pairs the implementation with the test transitions.

---

**End of plan context.**
