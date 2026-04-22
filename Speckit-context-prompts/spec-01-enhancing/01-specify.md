# Specify Context — Phase 01: Public-Release Hardening of `bin-packer-3d`

> Hand this entire document to `speckit.specify` as the feature description.
> SpecKit will turn it into a rigorous spec.md with user stories, functional
> requirements, success criteria, and edge cases. The goal of THIS document is
> to give SpecKit enough signal that it never has to emit `[NEEDS CLARIFICATION]`.

---

## 0. Brief

Take the existing `bin-packer-3d` project — a Python implementation of the 3D
Bin Packing Problem currently in `0.1.0 / Alpha` — and harden it into a credible
public open-source release that can be cited on a CV and stand up to inspection
by a senior engineer or hiring manager. This is a **level-up pass on a working
codebase**, not a greenfield rewrite.

The user is the sole maintainer (Bruno Ghiberto). The project is already on
GitHub at `https://github.com/Bruno-Ghiberto/3D_BIN_PACKING`. After this change
ships, a recruiter or peer reviewer landing on the repo should within 30 seconds
conclude: *"this person knows how to ship serious software."*

---

## 1. Project Background

The 3D Bin Packing Problem (3D-BPP) is a classic NP-hard combinatorial
optimisation problem with applications in logistics, container loading,
warehousing, and palletisation. Given a set of rectangular boxes and bins of
fixed dimensions, the goal is to pack all boxes into the minimum number of bins
respecting geometric, non-overlap, and orientation constraints.

This project implements heuristic solvers for that problem with interactive 3D
visualisation. It originated as a personal/work-driven exploration (early
commits in Spanish referencing partial implementations) and was later refactored
into a packaged Python module.

Current latest commit: `905a2f6 feat: Complete professional 3D Bin Packing solver implementation`.

---

## 2. Current State Snapshot

A faithful baseline. The spec must NOT assume anything beyond what is described
here.

### 2.1 Repository layout

```
3D_BIN_PACKING/
├── src/bin_packer_3d/        # Modern packaged implementation (current)
│   ├── algorithms/           # base.py, ffd.py, shelf.py
│   ├── models/               # box.py, bin.py, placement.py
│   ├── visualization/        # plotter.py (Plotly)
│   ├── data/                 # loaders.py (CSV / Excel via pandas)
│   ├── utils/                # metrics.py
│   ├── cli.py                # Click-based CLI: `bin-packer pack|info|init`
│   ├── config.py             # Pydantic v2 settings (PackerConfig, Settings)
│   ├── __init__.py           # public API surface
│   └── __main__.py           # `python -m bin_packer_3d`
├── tests/
│   ├── conftest.py           # fixtures: sample_box, sample_boxes, default_config
│   ├── unit/                 # test_models.py, test_algorithms.py
│   └── integration/          # test_packing.py (workflow + overlap + bounds)
├── DATASETS/                 # sample_boxes.csv + 3 .xlsx files (mixed content)
├── CODE/                     # ⚠️  LEGACY scripts with hardcoded Windows paths
│   ├── MAIN.py               # references C:\Users\bghiberto\... paths
│   ├── AdvancedHeuristicPacker.py
│   ├── Plotter.py
│   └── Utils.py
├── pyproject.toml            # hatchling build, ruff/black/mypy/pytest config
├── MANIFEST.in
├── README.md                 # has badges; CV-shaped but generic
├── LICENSE                   # MIT
├── .env.example              # access denied during audit; assumed present
└── .gitignore
```

### 2.2 What works today

- Two algorithms implemented and passing unit tests:
  - **First-Fit Decreasing (FFD)** with guillotine-style 3D space splitting
    (`src/bin_packer_3d/algorithms/ffd.py`).
  - **Shelf-based packer** with ceiling-locking shelves
    (`src/bin_packer_3d/algorithms/shelf.py`).
- `Box` model with 6-orientation enumeration deduplicated for cubes
  (`src/bin_packer_3d/models/box.py:77`).
- `Bin` model with utilisation + weight constraint
  (`src/bin_packer_3d/models/bin.py`).
- `Placement` model with overlap detection + dict export
  (`src/bin_packer_3d/models/placement.py`).
- Pydantic v2 typed settings with env-var support, `BIN_PACKER_` prefix,
  nested delimiter `__` (`src/bin_packer_3d/config.py`).
- Click + Rich CLI: `bin-packer pack <file>`, `info`, `init`
  (`src/bin_packer_3d/cli.py`).
- Plotly 3D visualisation per bin with mesh + wireframe + hover info
  (`src/bin_packer_3d/visualization/plotter.py`).
- pandas-based CSV / Excel loader with quantity expansion
  (`src/bin_packer_3d/data/loaders.py`).
- Pytest suite with unit + integration markers, mypy strict, ruff, black
  configured in `pyproject.toml`. README claims **39 tests passing**.
- Modern packaging: hatchling backend, `pyproject.toml` only, declares
  `bin-packer` console script.

### 2.3 What is broken, missing, or embarrassing

The spec must explicitly address each of these. Severity in brackets.

**Contract integrity (HIGH severity — public credibility risk)**
- `PackerConfig.strategy` is typed `Literal["ffd", "bfd", "shelf", "extreme_points"]`
  (`src/bin_packer_3d/config.py:34`) but `bfd` and `extreme_points` have NO
  implementation. The CLI restricts to `ffd|shelf` so it doesn't crash, but the
  public API silently lies about supported strategies.
- README references "Multiple Packing Algorithms" plural and lists only two —
  acceptable, but the type signature still lies.
- `Box.weight` defaults to 0.0 and is consumed by `Bin.can_fit_weight` with no
  documented semantics for "no weight given". Loaders only read a hardcoded
  `PESO` column outside the configurable mapping.

**CI / CD / Quality automation (HIGH)**
- No `.github/` directory at all. No GitHub Actions, no Dependabot, no CodeQL,
  no issue templates, no PR template, no funding file.
- `pre-commit` is in dev deps but there is no `.pre-commit-config.yaml`.
- Coverage tooling configured in `pyproject.toml` but no badge backed by real
  measurements, no Codecov / Coveralls integration.
- No automated release workflow. PyPI publish is manual / non-existent.
- Tests claim "39 passing" in README but there is no automation that re-asserts
  this on every push.

**Repository hygiene (HIGH)**
- The `CODE/` directory is dead code preserved at the repo root. It contains
  hardcoded Windows paths (`C:\Users\bghiberto\source\repos\...`) and references
  files (`asignacion_cajas_final.csv`) that don't exist in `DATASETS/`. A
  visitor opens the repo, sees `CODE/`, opens `MAIN.py`, and concludes the
  project is broken.
- `DATASETS/` mixes a clean `sample_boxes.csv` with three Spanish-named `.xlsx`
  files (`PACKING LIST.xlsx`, `PACKING LIST-11.xlsx`, `DIMENSIONES CAJAS-NORMALIZADO.xlsx`,
  `PESO_P.T.xlsx`) that look like artefacts from the original real-world job.
  No metadata explains what each is or whether it should be in version control.
- `__init__.py:31` declares `__author__ = "Bruno"` (no surname) while
  `pyproject.toml` has the full identity. Inconsistent.
- Mixed-language history in commit log (Spanish + English). Not blocking but
  signals the transition from personal script to public project — the README
  and policy docs must be uniformly English to set the public tone.

**Documentation (HIGH)**
- No `docs/` directory. No documentation site (Sphinx, MkDocs, or otherwise).
- No `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `CHANGELOG.md`, `SECURITY.md`.
- No architectural diagram or Architecture Decision Records (ADRs).
- No example Jupyter notebook or scripted tutorial.
- README "Author" section has bare GitHub link, no project narrative or
  motivation, no benchmark results, no comparison with literature or competing
  Python libraries (`py3dbp`, `rectpack`, `BinPacking3D`).
- No documented coordinate convention diagram (the code uses X=length,
  Y=width, Z=height — non-obvious and easy to get wrong).

**Algorithm portfolio (MEDIUM — primary CV signal)**
- Only two heuristics. A serious portfolio for an NP-hard problem would
  showcase several families:
  - First-Fit Decreasing (HAVE)
  - Best-Fit Decreasing (declared, NOT implemented)
  - Shelf with multiple variants (HAVE one)
  - Extreme Point heuristic (declared, NOT implemented) — this is *the* modern
    reference baseline (Crainic, Perboli, Tadei 2008)
  - Maximal Rectangles / Skyline approaches
  - Optional metaheuristic stub (Genetic Algorithm, Simulated Annealing,
    Iterated Local Search) for future extension
- No benchmarking against the canonical literature instances (Bischoff &
  Ratcliff 1995 BR1–BR8, Loh & Nee 1992, Davies & Bischoff 1999).
- No `bin-packer compare` CLI command for side-by-side algorithm runs.
- No load-stability constraint (boxes floating in mid-air is currently legal).
- No "this side up" / fragility / load-bearing constraint framework.

**Observability & code smells (MEDIUM)**
- `print()` statements in `data/loaders.py:113` and `visualization/plotter.py:226`
  instead of structured logging.
- `data/loaders.py:112` swallows row-level exceptions with `print` and `continue`
  — silent data quality bugs in production usage.
- `visualization/plotter.py:224` uses `plotly.offline.pyo.plot` — older API,
  modern equivalent is `fig.write_html()`.
- No determinism: tests that rely on box ordering have no seed control, making
  future regression debugging harder.

**Distribution (LOW for now, MEDIUM for "public credibility")**
- Not published to PyPI. README says "Clone and `pip install -e`" — fine for a
  toy project, weak signal for a "professional" project.
- No Docker image / `Dockerfile`.
- No `CHANGELOG.md` and no semantic versioning policy documented.

---

## 3. Strategic Goals

In priority order. The spec must produce user stories that map cleanly onto
these goals.

1. **Make the project's contract honest.** Nothing the public API or
   documentation claims should be unimplemented or untested.
2. **Demonstrate engineering discipline through automation.** CI, coverage,
   linting, type-checking, and release pipelines that any reviewer can verify
   without running the code locally.
3. **Tell a credible technical story.** The README, docs, and ADRs collectively
   communicate the engineering judgment behind the project.
4. **Show algorithmic depth.** At least one additional, named, literature-grade
   heuristic ships, with reproducible benchmark numbers against published
   instances.
5. **Be installable by a stranger in one command.** Either via PyPI or a
   documented `pipx`/`uv tool install` path.
6. **Make it pleasant to extend.** A new contributor can add a new algorithm by
   implementing one base class and dropping a file — no configuration scattered
   across the repo.

---

## 4. Non-Goals (out of scope for THIS spec)

State these explicitly so SpecKit doesn't try to scope them in.

- A web UI / SaaS deployment.
- Distributed / parallelised packing across machines.
- Exact / branch-and-bound ILP solvers (Gurobi, CPLEX, OR-Tools CP-SAT).
- Reinforcement-learning packers.
- Multi-bin-type heterogeneous fleet optimisation (one bin geometry per run is
  acceptable for this phase; framework should not preclude it later).
- A REST API (deferred; might appear in a later spec phase).
- Internationalisation of the CLI / documentation (English only).
- Backwards compatibility with the legacy `CODE/` scripts — they will be
  removed or relocated.
- Native extensions (Rust, C, or Cython-compiled units). Deferred until a
  measured profile identifies a hot path that NumPy / Numba / Cython cannot
  resolve. When that threshold is crossed, the preferred path is Rust via
  PyO3 + maturin, introduced in a later spec phase with an Architecture
  Decision Record documenting the flamegraph, the measured speedup vs.
  cheaper alternatives, and the wheel-distribution plan.

---

## 5. Constraints & Non-Negotiables

- **Language:** Python 3.11+ as the public floor (update `requires-python`
  from current `>=3.10` to `>=3.11` in implementation); maintainer develops
  on Python 3.14.3; CI matrix covers 3.11, 3.12, 3.13, 3.14.
- **License:** MIT (already in `LICENSE` and `pyproject.toml`).
- **Maintainer:** single solo maintainer; tooling must not require paid
  services beyond GitHub free tier and PyPI free tier.
- **Build system:** stay on hatchling unless there's a compelling reason to
  switch (don't switch for taste).
- **Type safety:** mypy strict mode must continue to pass.
- **Style:** ruff + black, line length 100 (already configured).
- **Public API stability:** any breaking change to the existing public
  `bin_packer_3d` symbols must be either (a) documented in CHANGELOG with a
  deprecation path, or (b) justified explicitly because the project is still
  pre-1.0.
- **Dataset privacy:** any `.xlsx` files in `DATASETS/` that contain
  identifiable real-world business data must be either anonymised or removed
  from version history before this spec is implemented. Treat as a security /
  privacy concern, not a hygiene one.
- **No vendor lock-in:** no proprietary CI providers, no closed-source
  observability tools.

---

## 6. Target Audiences

The spec must keep all four audiences in mind because they will read different
artefacts and judge the project on different criteria.

| Audience | Primary artefact | Judgment criterion |
|---|---|---|
| Recruiter / hiring manager | README, GitHub repo summary | "Does this look like real engineering?" |
| Senior engineer reviewing a CV | README, ADRs, CI workflow files, test suite | "Would I trust this person on my team?" |
| Operations-research practitioner | Benchmark results, algorithm docs, source | "Are the algorithms correctly implemented?" |
| Open-source contributor | CONTRIBUTING.md, issue templates, test suite | "Can I add a feature in an evening?" |

---

## 7. Capability Workstreams — User Stories with Priority

SpecKit's spec format organises by independently-shippable user stories with
priorities (P1 = MVP for credibility, P2 = depth, P3 = polish). Each story
below should map to one user story in the generated spec, with FRs and
acceptance criteria that exercise the listed capabilities.

### US1 — Contract Integrity & Algorithm Honesty (P1)

**As a** library user
**I want** the public API to truthfully reflect what is implemented
**so that** I can trust type signatures, documentation, and CLI options.

Capabilities the spec must require:
- The `PackerConfig.strategy` accepted values match exactly the set of
  registered, working algorithms at runtime.
- A registry / discovery mechanism for algorithms so adding one updates the
  accepted strategy values automatically (single source of truth).
- The CLI `info` command lists every algorithm the runtime can actually
  execute, with its complexity class and brief description, generated from the
  registry — not hardcoded.
- Configurable column mappings in the CSV/Excel loader cover EVERY column the
  loader reads (currently `CAJA`, `DESCRIPCION`, `PESO` are hardcoded outside
  the `DataConfig` mapping).
- `Box.weight` semantics are documented: what does `0.0` mean? "Unknown" vs
  "weightless" must be distinguishable in the model.

Acceptance criteria the spec must encode:
- Attempting to instantiate `PackerConfig(strategy="bfd")` must either succeed
  (because BFD is now implemented per US5) or fail with a clear error listing
  the available strategies — never silently accepted.
- A test exists that asserts `PackerConfig.model_fields["strategy"]` accepted
  values equal the algorithm registry's keys.

### US2 — Continuous Integration Pipeline (P1)

**As a** reviewer of the public repository
**I want to** see green checks on every push and pull request
**so that** I can trust the project enforces its own quality bar.

Capabilities:
- A CI pipeline that runs on every push to `main` and every pull request,
  across the supported Python versions (3.11, 3.12, 3.13, 3.14), on at
  minimum Linux.
- Pipeline stages: install, lint (ruff), format check (black), type-check
  (mypy strict), unit tests, integration tests, coverage report.
- Coverage uploaded to a public, free service (Codecov or equivalent) with a
  badge in the README that reflects the latest measured value.
- A separate workflow for security: dependency scanning (Dependabot or
  equivalent) and at least one static-analysis sweep (CodeQL or equivalent).
- Pre-commit hook configuration (`.pre-commit-config.yaml`) that mirrors the CI
  lint/format/type-check stages so contributors fail fast locally.
- Issue templates (bug report, feature request) and a pull request template.

Acceptance criteria:
- Opening a pull request triggers the pipeline.
- A PR that introduces a ruff violation, a type error, or a test failure is
  blocked from merging.
- README displays at least: build status badge, coverage badge, Python version
  badge, license badge, PyPI version badge (once US8 is done).

### US3 — Public-Facing Documentation (P1)

**As a** newcomer to the repository
**I want** documentation that explains the problem, the architecture, and how
to use the library
**so that** I can understand and contribute without reading every source file.

Capabilities:
- A documentation site generated from the repo (MkDocs Material or Sphinx are
  both acceptable; the spec should not prescribe), published via GitHub Pages
  on every push to `main`.
- Site sections at minimum:
  - Problem introduction (3D-BPP, NP-hard, real-world relevance, with a
    diagram of the coordinate convention used by this library).
  - Quickstart (CLI + Python API).
  - Algorithm reference (one page per algorithm: complexity, pseudo-code,
    citation to literature, when to choose it).
  - Constraints reference (rotation, weight, future stability).
  - Benchmark results page (tables produced by US5).
  - API reference auto-generated from docstrings.
  - Contributing guide.
- Repository top-level files:
  - `CONTRIBUTING.md` (how to set up the dev environment, run tests, add an
    algorithm, submit a PR).
  - `CHANGELOG.md` following Keep-a-Changelog format with an `Unreleased`
    section.
  - `CODE_OF_CONDUCT.md` (Contributor Covenant or equivalent).
  - `SECURITY.md` (how to report a vulnerability).
- README rewritten to:
  - Open with a hero image / GIF showing a Plotly visualisation.
  - State the problem in two sentences.
  - Show benchmark headline numbers ("X% better utilisation than baseline Y on
    instance Z").
  - List a maximum of five canonical use cases.
  - Link to the docs site for everything else.
- An ADR directory (`docs/adr/`) with at least three accepted ADRs documenting
  past or new decisions: choice of Pydantic for config, choice of Plotly for
  visualisation, coordinate convention.
- An `examples/` directory with at least one runnable example notebook
  (or scripted equivalent) loading sample data, running multiple algorithms,
  and producing comparison plots.

Acceptance criteria:
- The docs site is reachable from a link in the README and builds in CI.
- Every public class and function has a docstring; pydocstyle (or ruff's `D`
  ruleset) enforces this in CI.
- A first-time contributor following CONTRIBUTING.md can go from `git clone` to
  passing `pytest` in under five commands.

### US4 — Repository Hygiene (P1)

**As a** visitor browsing the file tree
**I want** every file in the repository to have a clear purpose
**so that** I don't lose trust on first inspection.

Capabilities:
- The legacy `CODE/` directory is either removed entirely or relocated to a
  clearly-marked `legacy/` (or `archive/`) folder with a `README.md` explaining
  it is preserved for historical reference and is not part of the supported
  package.
- All hardcoded absolute paths are removed from any retained legacy file.
- `DATASETS/` contains only files that are (a) meant to be public and (b)
  referenced from documentation or tests. Each file has an accompanying
  description in `DATASETS/README.md` explaining its origin, schema, and
  intended use.
- Any business-confidential `.xlsx` files are removed from the working tree
  AND scrubbed from git history (BFG or `git filter-repo`) before this story
  is closed.
- `__init__.py` author metadata matches `pyproject.toml`.
- A `.editorconfig` file standardises whitespace expectations across editors.
- A `py.typed` marker is shipped in the wheel so downstream users get type
  inference (PEP 561).

Acceptance criteria:
- `git ls-files` returns no path containing absolute Windows / Unix
  user-specific paths.
- A reviewer can open every directory at the repo root and identify its
  purpose from its top-level README within 10 seconds.

### US5 — Algorithm Portfolio Expansion + Benchmarks (P2)

**As an** operations-research-minded reviewer
**I want** the library to implement and benchmark multiple competitive
heuristics
**so that** I can judge the implementer's algorithmic depth.

Capabilities:
- Implement at least the following additional algorithms, each as its own
  module under `src/bin_packer_3d/algorithms/`:
  - **Best-Fit Decreasing (BFD)** — currently a phantom in the type signature.
  - **Extreme Point heuristic** (Crainic, Perboli, Tadei, 2008) — the modern
    reference baseline; without it, an OR reviewer dismisses the project.
  - One additional family of the implementer's choice — for example a
    Maximal-Rectangles / Skyline approach, or a Layer-building heuristic.
- A pluggable algorithm registry (per US1) so each new algorithm registers
  itself once and becomes available to the CLI, config, and tests
  automatically.
- A `bin-packer benchmark` CLI command that:
  - Runs every registered algorithm against one or more benchmark instances.
  - Emits a comparison table (utilisation %, bins used, success rate, runtime)
    in plain text, JSON, and Markdown formats.
  - Supports a `--seed` option for reproducibility on any randomised
    components.
- Benchmark instances:
  - Bundle at least one published reference instance set (Bischoff & Ratcliff
    1995 is the canonical one for 3D-BPP and is freely available; OR-Library
    hosts it). If license permits redistribution, ship inside the repo;
    otherwise, ship a downloader script.
  - Include the existing `sample_boxes.csv` as a smoke-test instance.
- A benchmark results page in the docs site (see US3) that is generated from
  the benchmark runner output, dated, and committed alongside each release.
- Property-based tests using Hypothesis for invariants that must hold for
  every algorithm:
  - No two placements in the same bin overlap.
  - All placements lie within their bin's bounds.
  - Every box's volume in the result equals its volume in the input.
  - Box count in the result equals box count in the input minus unpacked.

Acceptance criteria:
- The benchmark CLI command runs end-to-end on the bundled instances in CI on
  every push (perhaps as a slow-marker test) and produces a JSON artefact.
- At least one of the new algorithms must demonstrate measurably different
  behaviour from FFD on at least one bundled instance — documented in the
  benchmark page.
- Hypothesis tests run in CI and cover all registered algorithms via
  parametrisation.

### US6 — Observability & Diagnostics (P2)

**As a** library user debugging an unexpected packing outcome
**I want** structured, controllable logging
**so that** I can understand what the algorithm did without modifying source.

Capabilities:
- All `print()` calls in library code replaced with `logging` at appropriate
  levels.
- Library follows the standard "library logging" pattern: a module-level
  logger, a `NullHandler` attached, no global configuration imposed on
  consumers.
- The CLI configures a Rich-based handler for nice output, controlled by
  `--verbose` / `--quiet` flags and the existing `debug` setting in `Settings`.
- Row-level errors in the loader are no longer swallowed: they are logged at
  WARNING and accumulated into a structured `LoadReport` returned alongside
  the boxes (or raised as an exception group at strict mode).
- A `bin-packer pack --explain` mode that, in addition to the result, emits a
  per-box trace (which bin / shelf was tried, why the placement failed if it
  did) at DEBUG level.

Acceptance criteria:
- No `print` calls in `src/bin_packer_3d/` except in the CLI module's
  user-facing output (acceptable to use Rich `console.print` there).
- Loader tests assert that malformed rows produce a logged warning AND appear
  in a returned `LoadReport`, not silent skips.

### US7 — Constraint Framework Foundation (P2)

**As an** advanced user
**I want** to express common real-world constraints declaratively
**so that** the solver respects them without me hacking the algorithm code.

Capabilities (foundational only — full implementation can come later):
- A `Constraint` abstraction (interface or strategy) that each algorithm
  consults before accepting a candidate placement.
- At least these two constraints implemented and covered by tests:
  - **Allowed orientations per box** — generalising the current
    `allow_rotation` flag from "all 6 or none" to "any subset of 6", expressed
    on the `Box` model (`Box.allowed_orientations`).
  - **Maximum stacking weight** per box — a placement is rejected if placing
    something on top would exceed the supporting box's load capacity. (Even
    if no algorithm yet does true stability checking, the data model and
    constraint should exist.)
- The architecture must allow a future `StabilityConstraint` (boxes must rest
  on the bin floor or on top of another box) to be added without changing
  algorithm interfaces.

Acceptance criteria:
- A box created with `allowed_orientations={"flat"}` is never placed on its
  side by any algorithm.
- A box with a `max_supported_weight` of 5 kg never has a placement above it
  whose total weight exceeds 5 kg, in tests covering every algorithm.
- An ADR documents the constraint architecture and explicitly lists which
  constraints are deferred to later phases.

### US8 — Distribution & Release Engineering (P3)

**As a** prospective user
**I want** to install the library with a single command from PyPI
**so that** I do not have to clone a repository.

Capabilities:
- The package is published to PyPI under the name `bin-packer-3d` (or
  alternative if taken — verify availability and document the chosen name).
- A release workflow on GitHub Actions that, when triggered by a Git tag
  matching `v*.*.*`, builds the wheel + sdist, runs the full test suite, and
  uploads to PyPI using a trusted publisher (OIDC; no long-lived API tokens in
  repo secrets).
- Versioning policy documented: SemVer, with the meaning of MAJOR / MINOR /
  PATCH for this project explicitly defined (e.g., MAJOR = breaking public API
  change, MINOR = new algorithm or new constraint, PATCH = bug fix).
- A `Dockerfile` (multi-stage, non-root runtime) and a published image
  (GitHub Container Registry is acceptable) that exposes the CLI as the
  entrypoint.
- A `CHANGELOG.md` updated as part of every release; an `Unreleased` section
  collects in-flight changes.

Acceptance criteria:
- `pipx install bin-packer-3d` (or equivalent) on a clean machine yields a
  working `bin-packer` command.
- `docker run ghcr.io/<owner>/bin-packer-3d:<tag> info` prints algorithm info.
- The most recent release tag has a matching CHANGELOG entry and a published
  PyPI artefact whose version matches.

### US9 — Interactive Demo (P3)

**As a** casual visitor or recruiter
**I want** to see the algorithms in action without installing anything
**so that** I get an immediate sense of the project's value.

Capabilities:
- One of:
  - A static GitHub Pages demo page hosting a representative Plotly HTML
    output (cheapest), OR
  - A small Streamlit / Gradio app deployable to a free-tier host (HF Spaces,
    Streamlit Community Cloud) where a user can upload a CSV, pick an
    algorithm, and see the visualisation.
- The README's hero image / GIF (US3) links to this demo if it exists.

Acceptance criteria:
- A reviewer can click one link from the README and within 30 seconds see
  packed boxes rendered in 3D, without local install.

---

## 8. Cross-Cutting Quality Bars

These apply to every workstream and the spec should mention them explicitly.

- **Testing:** unit + integration + property-based tests; coverage minimum
  threshold (the spec proposes 90% line coverage for `src/bin_packer_3d/`,
  enforced in CI).
- **Type safety:** every new public symbol carries type hints; mypy strict
  remains green.
- **Determinism:** any randomness in algorithms or tests is seeded; results
  are reproducible from a `(seed, input, config)` triple.
- **Performance regressions:** the existing `test_large_dataset` (100 boxes,
  <5s budget) is preserved; the spec proposes adding a coarse benchmark
  workflow that warns if any algorithm regresses by >25% on a fixed instance.
- **Internationalisation:** all user-facing strings are English; all in-code
  comments and identifiers are English (current code is mixed with Spanish
  artefacts in DATASETS).
- **License hygiene:** any third-party benchmark instances bundled in the
  repository carry their original license headers; CONTRIBUTING.md states the
  contribution license (MIT) explicitly.

---

## 9. Edge Cases & Risks the Spec Must Address

SpecKit's "Edge Cases" section should at minimum cover these:

- An empty box list passed to `pack()` must return an empty `PlacementResult`
  with `success_rate == 100.0` (current behaviour — must remain a tested
  contract).
- A box larger than the bin in every orientation must be returned in
  `unpacked_boxes`, never crash. Currently handled in FFD; must be tested for
  every algorithm in the registry.
- A CSV with missing optional columns (CAJA, DESCRIPCION, PESO) must load
  correctly — no `KeyError`.
- A CSV with malformed rows produces structured warnings, not silent skips.
- Floating-point coordinate boundaries (`x0 == x1` after split) must not
  create zero-volume placements.
- Cube boxes (`width == height == length`) must yield exactly one orientation,
  not six. (Already true; preserve as a regression test.)
- Bin weight constraint with `max_weight == None` means "unlimited", not
  "zero" — must be documented and tested.
- A benchmark run on an unknown algorithm name in the registry must list
  available names in the error message.
- Concurrent invocations of `Plotter3D` must not collide on output paths —
  output filenames must include the bin id and ideally a timestamp / run id.

---

## 10. Success Criteria — Project-Level Definition of Done

The phase is done when **all** of the following hold:

1. A first-time visitor lands on the GitHub repo and within 30 seconds sees:
   green CI badge, coverage badge ≥ 90%, PyPI version badge, hero
   visualisation, problem statement, install command.
2. `pipx install bin-packer-3d && bin-packer pack DATASETS/sample_boxes.csv`
   on a fresh machine produces a result and an HTML visualisation.
3. The docs site is live, linked from the README, and contains an algorithm
   reference page for every algorithm the runtime can execute.
4. No file in `src/bin_packer_3d/` contains a `print()` call that should be
   logging.
5. The `PackerConfig.strategy` `Literal` exactly matches the algorithm
   registry's keys, asserted by an automated test.
6. At least one published reference benchmark instance is run in CI on every
   push, with results stored as a build artefact.
7. The legacy `CODE/` directory is removed from `main` (history may be
   preserved on a tag).
8. All `.xlsx` files in `DATASETS/` are either anonymised or removed AND
   scrubbed from git history if they contained business-confidential data.
9. CHANGELOG.md, CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md, ADRs ≥3
   exist and are linked from the README.
10. The most recent commit on `main` is associated with a Git tag that
    matches a published PyPI release.

---

## 11. Inputs Available to the Spec Author

When SpecKit generates the spec, it can refer to (and the planning phase
afterwards will need to read) the following:

- `pyproject.toml` — current packaging, tooling config, and declared deps.
- `src/bin_packer_3d/` — current implementation, all 12 modules.
- `tests/` — current 39-test suite (estimated; CI will produce the exact
  number).
- `README.md` — current narrative to be rewritten.
- `CODE/` — legacy reference implementation that motivated the rewrite.
- `DATASETS/sample_boxes.csv` — known-good canonical input.
- Git log — context on the project's evolution from script to package.

---

## 12. Narrative the Final README Must Tell

A second-level summary the spec should ensure the rewritten README delivers:

> *"3D Bin Packing is a classic NP-hard problem. This library implements a
> portfolio of competitive heuristics — FFD, BFD, Shelf-based, Extreme Point —
> with a pluggable architecture, interactive 3D visualisation, and reproducible
> benchmarks against published reference instances. Built with modern Python
> tooling (Pydantic, Click, Plotly, Hypothesis), strict typing, and full CI/CD.
> Install in one command, get a 3D HTML visualisation in two."*

If the rewritten README does not deliver that paragraph in substance, the spec
has failed.

---

## 13. Suggested User-Story Priority Map (for SpecKit)

| Priority | Stories |
|---|---|
| P1 (must ship together as MVP for credibility) | US1, US2, US3, US4 |
| P2 (depth & engineering signal) | US5, US6, US7 |
| P3 (polish & reach) | US8, US9 |

The spec should make P1 stories independently shippable and verifiable. P2 may
depend on P1 (for example, US5 requires US1's algorithm registry). P3 may
depend on P2 (US8's PyPI release should not happen before US5's benchmarks
exist, so the first PyPI version represents the full, credible portfolio).

---

*End of context. Hand this entire document to `speckit.specify`.*
