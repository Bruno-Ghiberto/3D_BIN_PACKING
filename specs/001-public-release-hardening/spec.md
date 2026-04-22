# Feature Specification: Public-Release Hardening of `bin-packer-3d`

**Feature Branch**: `001-public-release-hardening`
**Created**: 2026-04-22
**Status**: Draft
**Input**: User description: "Read @Speckit-context-prompts/spec-01-enhancing/01-specify.md" — turn the existing alpha `bin-packer-3d` Python package into a credible public open-source release that stands up to inspection by a senior engineer, a hiring manager, and an operations-research practitioner.

## User Scenarios & Testing *(mandatory)*

<!--
  User stories are prioritized as independently-testable journeys.
  Priorities: P1 = MVP-for-credibility (must ship together), P2 = depth, P3 = polish & reach.
  Each story delivers a standalone slice of value that can be developed, tested,
  and demonstrated on its own — even though P1 stories should bundle at release
  time to meet the first-impression goal.
-->

### User Story 1 - Contract Integrity & Algorithm Honesty (Priority: P1)

As a library user importing and using `bin_packer_3d`, I want the public API, configuration schema, CLI options, and documentation to faithfully describe exactly what the runtime can do, so that I never discover phantom strategies, silently-dropped errors, or documentation that lies at runtime.

**Why this priority**: The current public configuration accepts strategy names (`bfd`, `extreme_points`) that have no implementation. A public API that lies erodes trust faster than a missing feature; this is the most fundamental credibility defect in the codebase and everything else rests on it.

**Independent Test**: Instantiate the packer configuration with every advertised strategy name and verify each either executes or fails with a clear error listing runtime-registered algorithms; invoke the CLI `info` command and verify its output lists every algorithm the runtime can execute (and only those); load a CSV that omits optional columns and verify the loader honours the declared column mapping without raising `KeyError`.

**Acceptance Scenarios**:

1. **Given** the library is installed, **When** a user enumerates the accepted strategy values on the packer configuration, **Then** every returned value names an algorithm discoverable and executable at runtime — no phantom entries.
2. **Given** a new algorithm is added to the runtime registry, **When** a user reads the accepted strategy values or the CLI `info` output, **Then** the new algorithm appears automatically without additional changes to the configuration or CLI code.
3. **Given** the CLI `info` command is invoked, **When** the output is read, **Then** each registered algorithm is listed with its complexity class and a short description, sourced from the registry rather than hardcoded.
4. **Given** a CSV or Excel file lacks any combination of optional columns (identifier, description, weight), **When** the loader reads it, **Then** loading completes without `KeyError`, honouring the declared column mapping.
5. **Given** a box is constructed without an explicit weight, **When** its weight is later inspected, **Then** the state "weight unknown" is distinguishable from the state "weight is zero" — no ambiguous `0.0` default.
6. **Given** the documentation describes an algorithm or configuration option, **When** the library runtime is queried for the same, **Then** an automated test asserts the two agree.

---

### User Story 2 - Continuous Integration Pipeline (Priority: P1)

As a reviewer (hiring manager, senior engineer, or open-source contributor) landing on the public repository, I want to see green status checks on every commit and pull request plus live quality badges in the README, so that I can trust the project enforces its own stated quality bar without running the code locally.

**Why this priority**: README claims of passing tests are unverifiable without CI. CI is the primary engineering-discipline signal a senior reviewer looks for; without it, every other quality claim is taken on faith.

**Independent Test**: Open a draft pull request that deliberately introduces a lint violation, a type error, and a failing test; verify each one independently blocks the PR from merging. Render the README in a browser and confirm the badges reflect the most recent build state.

**Acceptance Scenarios**:

1. **Given** a commit is pushed to `main` or a pull request is opened, **When** the pipeline runs, **Then** it installs dependencies and executes linting, format checking, strict static type checking, unit tests, and integration tests across the supported Python versions (3.11, 3.12, 3.13, 3.14) on Linux at minimum.
2. **Given** a pull request introduces a lint violation, a type error, or a failing test, **When** the pipeline completes, **Then** the PR is blocked from merging (branch protection is enforced).
3. **Given** the pipeline produces a coverage report, **When** the report is uploaded, **Then** a publicly visible free-tier coverage service displays the current coverage percentage and a README badge reflects it live.
4. **Given** a contributor installs the project's local pre-commit configuration, **When** they commit code, **Then** the same linting, formatting, and type checks that run in CI run locally first.
5. **Given** a dependency is added with a known vulnerability, **When** automated dependency scanning runs, **Then** the vulnerability is surfaced as an alert on the repository.
6. **Given** the README is rendered on GitHub, **When** a reviewer scans it, **Then** live badges for build status, coverage, supported Python versions, license, and (after US8) PyPI version are visible and current.
7. **Given** a visitor opens the `.github/` directory, **When** they look for contribution templates, **Then** they find bug-report, feature-request, and pull-request templates.

---

### User Story 3 - Public-Facing Documentation (Priority: P1)

As a newcomer landing on the repository (recruiter, senior engineer, operations-research practitioner, or would-be contributor), I want published documentation that explains the problem domain, the library's architecture, its algorithms, and how to use it, so that I can evaluate, adopt, or contribute to the library without reading source code.

**Why this priority**: Documentation separates "GitHub hobby project" from "credible library." Without it, none of the engineering work inside the package is legible from the outside. A senior reviewer will not read source to reconstruct your intent.

**Independent Test**: From a fresh browser session, open the repository root; within 60 seconds locate the documentation-site link from the README; open the site and verify the required sections are present and that the auto-generated API reference covers every public symbol.

**Acceptance Scenarios**:

1. **Given** a first-time visitor opens the README, **When** they scroll through it, **Then** they see, in order: a hero visualisation (image or GIF), a two-sentence problem statement, headline benchmark numbers, a maximum of five canonical use cases, and a single link to the documentation site.
2. **Given** the documentation site is published, **When** a visitor browses it, **Then** it contains at minimum: a problem introduction (3D-BPP, NP-hardness, real-world relevance, a diagram of the coordinate convention used), a quickstart for both the CLI and the Python API, one reference page per algorithm (complexity class, pseudo-code, literature citation, guidance on when to use it), a constraints reference, a benchmark results page, an auto-generated API reference, and a contributing guide.
3. **Given** the README is inspected, **When** its narrative is read, **Then** it delivers in substance the promise: "a portfolio of competitive heuristics with a pluggable architecture, interactive 3D visualisation, and reproducible benchmarks against published reference instances, installable in one command."
4. **Given** a new contributor follows `CONTRIBUTING.md`, **When** they execute its setup steps, **Then** they go from `git clone` to a passing test run in five commands or fewer.
5. **Given** the repository root is inspected, **When** a visitor lists the top-level files, **Then** `CONTRIBUTING.md`, `CHANGELOG.md` (Keep-a-Changelog format with an `Unreleased` section), `CODE_OF_CONDUCT.md` (Contributor Covenant or equivalent), and `SECURITY.md` are present and linked from the README.
6. **Given** the `docs/adr/` directory is opened, **When** a reviewer browses it, **Then** at least three accepted Architecture Decision Records document decisions such as the configuration approach, the choice of visualisation technology, and the coordinate convention.
7. **Given** the repository ships an `examples/` directory, **When** a user opens an example, **Then** they find a runnable notebook (or scripted equivalent) that loads sample data, runs multiple algorithms, and produces comparison plots.
8. **Given** docstring enforcement is configured, **When** CI runs, **Then** missing docstrings on public classes or functions fail the build.

---

### User Story 4 - Repository Hygiene (Priority: P1)

As a visitor browsing the repository's file tree for the first time, I want every top-level file and directory to have a clear, explicable purpose, so that I do not lose trust on first inspection.

**Why this priority**: The legacy `CODE/` directory with hardcoded Windows paths and the unexplained Spanish-named business `.xlsx` files in `DATASETS/` are first-impression destroyers. Hygiene must ship alongside the other P1 stories; without it, the P2/P3 work cannot rescue the first-impression verdict.

**Independent Test**: Run `git ls-files` and grep for user-specific absolute paths; open every top-level directory and verify it has a purpose description; inspect `DATASETS/` and verify every file is either public-safe with a description, or removed; check git history for any business-confidential artefacts that once existed.

**Acceptance Scenarios**:

1. **Given** a reviewer runs `git ls-files`, **When** the output is inspected, **Then** no tracked file contains hardcoded, user-specific, absolute paths.
2. **Given** the legacy reference implementation is inspected, **When** a visitor opens the repository, **Then** it either no longer exists, or it is moved to a clearly-marked `legacy/` (or `archive/`) folder with a README stating it is preserved for historical reference and is not part of the supported package.
3. **Given** `DATASETS/` is inspected, **When** a visitor opens it, **Then** every file has a description in `DATASETS/README.md` explaining its origin, schema, and intended use.
4. **Given** any `.xlsx` file in `DATASETS/` contained business-confidential or personally identifiable data, **When** the release ships, **Then** those files are anonymised or removed from the working tree AND scrubbed from git history.
5. **Given** the package metadata is inspected, **When** a reviewer compares the author field in the package `__init__` with `pyproject.toml`, **Then** they match exactly.
6. **Given** a downstream consumer installs the package, **When** they import it in a type-checked project, **Then** type hints are available (the `py.typed` marker is shipped in the wheel per PEP 561).
7. **Given** a contributor opens the project in any standard editor, **When** they start editing, **Then** an `.editorconfig` file at the repository root codifies consistent whitespace behaviour.

---

### User Story 5 - Algorithm Portfolio Expansion & Reproducible Benchmarks (Priority: P2)

As an operations-research-literate reviewer evaluating the project's algorithmic depth, I want the library to implement and benchmark multiple competitive heuristics against published reference instances with reproducible results, so that I can judge the implementer's competence and not dismiss the project as a homework assignment.

**Why this priority**: Depth of algorithm portfolio is the primary CV signal for an OR audience. Depends on US1's algorithm registry — without the registry, adding algorithms does not solve the contract-integrity problem.

**Independent Test**: Run `bin-packer benchmark --seed <N>` against a bundled reference instance; verify the output includes a comparison table with utilisation %, bins used, success rate, and runtime for at least three algorithms (FFD, BFD, Extreme Point, plus one additional family); rerun with the same seed and verify the result is bit-identical.

**Acceptance Scenarios**:

1. **Given** the library is installed, **When** a user runs `bin-packer benchmark` against a bundled instance, **Then** a comparison table is produced for every registered algorithm covering utilisation %, bins used, success rate, and runtime.
2. **Given** the benchmark runner supports multiple output formats, **When** a user requests plain text, JSON, or Markdown, **Then** the runner emits the requested format without loss of information.
3. **Given** a fixed random seed is passed via `--seed`, **When** the benchmark is re-executed with the same seed, input, and configuration, **Then** the result is bit-identical.
4. **Given** at least one published reference instance set is available, **When** the benchmark runs in CI on every push, **Then** a JSON artefact is stored on the build and a dated results entry is reflected on the benchmark documentation page.
5. **Given** property-based tests are configured, **When** the test suite runs against every registered algorithm, **Then** these invariants always hold: no two placements in the same bin overlap; every placement lies within its bin's bounds; the total volume of placed + unpacked boxes equals the input volume; the placed-box count plus the unpacked-box count equals the input count.
6. **Given** Best-Fit Decreasing and the Extreme Point heuristic are implemented, **When** the benchmark runs on at least one bundled instance, **Then** at least one new algorithm demonstrates a utilisation delta of ≥5 percentage points OR a bin-count delta of ≥1 bin from First-Fit Decreasing — documented on the benchmark documentation page.
7. **Given** a user passes an unknown algorithm name to the benchmark command, **When** the command runs, **Then** the error message lists every registered algorithm name.

---

### User Story 6 - Observability & Diagnostics (Priority: P2)

As a library user debugging an unexpected packing outcome, I want structured, controllable logging at the library level and an explain mode at the CLI level, so that I can understand what the algorithm did without patching source code.

**Why this priority**: Once the contract is honest (US1), users need visibility into algorithm decisions to trust and extend the library. Essential for adoption; not strictly required for a credible first-impression demo.

**Independent Test**: Run the CLI with a malformed CSV and verify the loader emits WARNING-level logs and returns a structured load report listing every bad row; run `bin-packer pack --explain` against a small instance and verify a per-box placement trace appears at DEBUG level.

**Acceptance Scenarios**:

1. **Given** the library is used as a dependency in a larger application, **When** it logs internal events, **Then** it follows the standard library-logging convention: a module-level logger with a `NullHandler` attached and no global logging configuration imposed on the consumer.
2. **Given** the CLI is invoked with `--verbose` or `--quiet`, **When** packing runs, **Then** the effective log level is adjusted to the requested setting (DEBUG / INFO / WARNING / ERROR).
3. **Given** the library source is inspected, **When** a reviewer searches for `print(`, **Then** none remain in `src/bin_packer_3d/` except Rich-backed user-facing output inside the CLI module.
4. **Given** a CSV contains malformed rows, **When** the loader reads the file, **Then** each bad row produces a WARNING log AND is surfaced in a structured load report returned to the caller — bad rows are never silently skipped.
5. **Given** `bin-packer pack --explain` is invoked, **When** DEBUG-level logging is enabled, **Then** a per-box trace is emitted documenting which bin and shelf was tried and, for failed placements, why.

---

### User Story 7 - Constraint Framework Foundation (Priority: P2)

As an advanced user with real-world packing rules, I want to express common constraints (allowed orientations per box, maximum stacking weight) declaratively and have every algorithm respect them, so that the solver honours my constraints without my patching algorithm code.

**Why this priority**: Positions the library for real-world problems (fragility, "this-side-up", future stability constraints). Foundational only in this phase — the full constraint taxonomy is deferred to a later spec phase.

**Independent Test**: Create a box with `allowed_orientations = {"flat"}` and run every registered algorithm; assert no placement puts the box on its side. Create a supporting box with a `max_supported_weight` and place heavier boxes on top; assert rejection by every algorithm.

**Acceptance Scenarios**:

1. **Given** a box is created with `allowed_orientations = {"flat"}`, **When** any algorithm places it, **Then** the placement orientation is always "flat" — never rotated onto its side.
2. **Given** a supporting box has `max_supported_weight = 5.0`, **When** any algorithm attempts to place boxes above it whose cumulative weight would exceed 5.0 units, **Then** those placements are rejected.
3. **Given** the architecture is inspected, **When** a reviewer reads the algorithm base class and the constraint abstraction, **Then** adding a new constraint (e.g. a future stability constraint) requires no changes to any algorithm's placement loop beyond consulting the constraint registry.
4. **Given** the `docs/adr/` directory is opened, **When** a reviewer reads the constraint-architecture ADR, **Then** it names the constraints implemented in this phase and explicitly lists which constraints are deferred to later phases.

---

### User Story 8 - Distribution & Release Engineering (Priority: P3)

As a prospective user discovering the library, I want to install it from PyPI with a single command or pull a container image, so that I do not have to clone the repository to try the library.

**Why this priority**: Reach amplifier. Depends on US5 — the first public release should ship with the full algorithm portfolio so that the public version represents the credible final state, not an interim one.

**Independent Test**: On a clean machine, run `pipx install bin-packer-3d` (or the documented alternative name if `bin-packer-3d` is unavailable on PyPI) and execute `bin-packer pack DATASETS/sample_boxes.csv`; verify a result and an HTML visualisation are produced. Separately, run `docker run <registry>/<image>:<tag> info` and verify it prints algorithm information.

**Acceptance Scenarios**:

1. **Given** a release tag matching `v*.*.*` is pushed, **When** the release workflow runs, **Then** it builds the wheel and sdist, runs the full test suite, and uploads the artefacts to PyPI via a trusted publisher (OIDC) — no long-lived API tokens in repository secrets.
2. **Given** a user on a clean machine runs `pipx install bin-packer-3d` (or the documented alternative if the primary name is taken), **When** the install completes, **Then** the `bin-packer` command is available and prints usage information.
3. **Given** a container image is published to a public free-tier registry, **When** a user runs `docker run <image> info`, **Then** the container prints registered algorithm information and exits cleanly.
4. **Given** a reviewer reads the versioning policy, **When** they look at the release notes or a dedicated policy document, **Then** the meaning of MAJOR / MINOR / PATCH is explicitly defined for this project.
5. **Given** a release exists, **When** a reviewer compares the Git tag, the `CHANGELOG.md` entry, and the PyPI artefact version, **Then** all three versions match.
6. **Given** the Dockerfile is inspected, **When** a reviewer reviews it, **Then** it uses a multi-stage build and runs as a non-root user.

---

### User Story 9 - Interactive Demo (Priority: P3)

As a casual visitor or recruiter who does not install software, I want to click one link from the README and see the algorithms running in 3D, so that I get an immediate sense of the project's value.

**Why this priority**: Maximum reach per unit of effort. Lowest priority because the README hero visualisation already captures most of this value; a click-through demo is the amplifier, not the foundation.

**Independent Test**: Open the README in a fresh browser, click the single demo link, and within 30 seconds confirm that a 3D packing visualisation is rendered — without any local install step.

**Acceptance Scenarios**:

1. **Given** the README contains a single demo link, **When** a visitor clicks it, **Then** they land on either a static GitHub Pages demo page hosting a representative 3D visualisation, or a free-tier hosted small app (e.g. Streamlit Community Cloud, HF Spaces) that renders a packing result.
2. **Given** the interactive option is shipped, **When** a visitor uploads a sample CSV, selects an algorithm, and runs it, **Then** a 3D visualisation is rendered within 30 seconds.

---

### Edge Cases

The system MUST handle the following edge conditions correctly, each verified by an automated test:

- **Empty input**: an empty box list passed to the packing entry point returns an empty result with `success_rate == 100.0` — preserved as a regression test (this is already current behaviour).
- **Oversized box**: a box larger than the bin in every orientation is returned in the `unpacked_boxes` field — never crashes — for every registered algorithm.
- **Optional columns missing**: a CSV with any combination of optional columns missing (identifier, description, weight) loads without error, consuming the configured column mapping.
- **Malformed rows**: a CSV with malformed rows produces structured WARNING logs and is reported in the returned load report — never silently skipped.
- **Zero-volume placements**: floating-point boundary conditions (`x0 == x1` after a guillotine split) must not produce zero-volume placements.
- **Cube boxes**: boxes with equal width, height, and length yield exactly one orientation, not six (this is already current behaviour; preserved as a regression test).
- **Unlimited bin weight**: a bin with `max_weight == None` means "unlimited capacity", not "zero capacity" — documented and tested.
- **Unknown algorithm name**: invoking a benchmark or CLI command with an unrecognised algorithm name produces an error message listing every available algorithm name.
- **Concurrent visualiser invocations**: two parallel runs must not collide on output paths — filenames include the bin identifier and a run-specific marker (timestamp or run id).

## Clarifications

### Session 2026-04-22

- Q: Privacy disposition of `DATASETS/*.xlsx` files (PACKING LIST.xlsx, PACKING LIST-11.xlsx, DIMENSIONES CAJAS-NORMALIZADO.xlsx, PESO_P.T.xlsx) — scrub all, keep all, replace with synthetic, or per-file audit? → A: Per-file audit — inspect each, scrub confidential ones from working tree and git history, keep safe-to-publish ones with `DATASETS/README.md` description.
- Q: Quantify "measurably different" in SC-006 (US5 algorithm-family acceptance threshold)? → A: ≥5 percentage points utilisation delta OR ≥1 bin delta from First-Fit Decreasing on the same instance.
- Q: CI benchmark reference instance scale — what executes per push vs. release? → A: One canonical Bischoff & Ratcliff 1995 BR instance (BR1) on every push; full BR1–BR8 suite as a release gate on `v*.*.*` tags.

## Requirements *(mandatory)*

### Functional Requirements

<!--
  Requirements are grouped by user story for traceability.
  Every requirement is testable and unambiguous.
-->

**Contract Integrity (US1)**

- **FR-001**: The set of accepted strategy values on the packer configuration MUST equal, exactly, the set of algorithms registered and executable at runtime.
- **FR-002**: The system MUST expose a single algorithm-registration mechanism; adding a new algorithm MUST automatically update the accepted configuration values, the CLI `info` listing, and any typed schemas exposed by the library.
- **FR-003**: The CLI `info` command MUST list every registered algorithm with its complexity class and a short description, sourced from the registry rather than hardcoded.
- **FR-004**: The CSV and Excel loader MUST consume a declared column mapping that covers every column the loader reads (including identifier, description, and weight columns).
- **FR-005**: The box model MUST distinguish the state "weight unknown" from the state "weight is zero".
- **FR-006**: An automated test MUST assert that the accepted configuration strategy values match the algorithm registry's keys on every CI run.

**Continuous Integration Pipeline (US2)**

- **FR-010**: CI MUST run on every push to `main` and every pull request.
- **FR-011**: CI MUST execute linting, format checking, strict static type checking, unit tests, and integration tests across Python 3.11, 3.12, 3.13, and 3.14 on Linux.
- **FR-012**: CI MUST produce a coverage report, upload it to a publicly visible free-tier coverage service, and block pull requests whose coverage falls below the project-wide threshold (90% line coverage on `src/bin_packer_3d/`).
- **FR-013**: A pre-commit configuration MUST be available that mirrors the CI lint, format, and type-check stages so contributors fail fast locally.
- **FR-014**: The repository MUST run dependency scanning (vulnerability alerts on new and existing dependencies) and at least one static-analysis sweep on every pull request.
- **FR-015**: The README MUST display live badges for build status, coverage, supported Python versions, license, and (after US8) PyPI version.
- **FR-016**: A `.github/` directory MUST include issue templates for bug reports and feature requests, and a pull-request template.
- **FR-017**: A pull request introducing a lint violation, a type error, or a failing test MUST be blocked from merging.

**Public-Facing Documentation (US3)**

- **FR-020**: A documentation site MUST be generated from the repository and published on every push to `main`.
- **FR-021**: The documentation site MUST contain the following sections: problem introduction (including a coordinate convention diagram), quickstart for the CLI and Python API, algorithm reference (one page per algorithm with complexity, pseudo-code, literature citation, and guidance), constraints reference, benchmark results page, auto-generated API reference, and contributing guide.
- **FR-022**: The README MUST contain: a hero visualisation, a two-sentence problem statement, headline benchmark numbers, at most five canonical use cases, and a single link to the documentation site.
- **FR-023**: The repository root MUST include `CONTRIBUTING.md`, `CHANGELOG.md` (Keep-a-Changelog format with an `Unreleased` section), `CODE_OF_CONDUCT.md`, and `SECURITY.md`, each linked from the README.
- **FR-024**: A `docs/adr/` directory MUST contain at least three accepted Architecture Decision Records.
- **FR-025**: An `examples/` directory MUST contain at least one runnable example that loads sample data, runs multiple algorithms, and produces comparison plots.
- **FR-026**: Every public class and function MUST carry a docstring; missing docstrings MUST fail CI.
- **FR-027**: `CONTRIBUTING.md` MUST describe a setup path such that a new contributor goes from `git clone` to a passing test run in five commands or fewer.

**Repository Hygiene (US4)**

- **FR-030**: No tracked file in the repository MAY contain hardcoded, user-specific, absolute paths.
- **FR-031**: The legacy `CODE/` directory MUST be either removed from the main branch or relocated to a clearly-marked `legacy/` (or `archive/`) folder with a README explaining its preservation as historical reference only.
- **FR-032**: Every file in `DATASETS/` MUST have an accompanying description (in `DATASETS/README.md`) covering origin, schema, and intended use.
- **FR-033**: Every `.xlsx` file in `DATASETS/` MUST be individually audited (per-file audit) for business-confidential or personally identifiable content. Files classified as confidential MUST be anonymised or removed from the working tree AND scrubbed from git history. Files classified as safe-to-publish MAY remain in the working tree and MUST be documented per FR-032.
- **FR-034**: Package metadata MUST be internally consistent — the author field in package source MUST match `pyproject.toml`.
- **FR-035**: The distributed wheel MUST include a `py.typed` marker per PEP 561.
- **FR-036**: An `.editorconfig` file at the repository root MUST codify whitespace expectations.

**Algorithm Portfolio & Benchmarks (US5)**

- **FR-040**: The library MUST implement Best-Fit Decreasing, the Extreme Point heuristic (per Crainic, Perboli, Tadei, 2008), and at least one additional heuristic from another family (e.g. Maximal Rectangles, Skyline, or Layer-building).
- **FR-041**: A `bin-packer benchmark` CLI command MUST run every registered algorithm against one or more benchmark instances and emit a comparison table containing utilisation %, bins used, success rate, and runtime.
- **FR-042**: The benchmark command MUST support plain text, JSON, and Markdown output formats.
- **FR-043**: The benchmark command MUST accept a `--seed` option; running with the same seed, input, and configuration MUST produce bit-identical results.
- **FR-044**: The Bischoff & Ratcliff 1995 BR1–BR8 reference instance set MUST be available to the benchmark runner — bundled inside the repository when the instance license permits, or provided via a shipped downloader script otherwise.
- **FR-045**: A per-push CI benchmark MUST execute the BR1 canonical instance against every registered algorithm and produce a JSON artefact attached to the build. The full BR1–BR8 suite MUST execute as a release gate on any Git tag matching `v*.*.*` and produce a consolidated JSON artefact attached to the release.
- **FR-046**: Property-based tests MUST cover every registered algorithm for these invariants: no in-bin overlap; in-bounds placements; volume conservation (input volume = placed + unpacked volume); box-count conservation (input count = placed + unpacked count).
- **FR-047**: When an unknown algorithm name is passed to the benchmark or any CLI command, the error message MUST list every registered algorithm name.

**Observability & Diagnostics (US6)**

- **FR-050**: The library MUST use module-level loggers with a `NullHandler` attached; the library MUST NOT call `logging.basicConfig` or otherwise impose global logging configuration on consumers.
- **FR-051**: The CLI MUST support `--verbose` and `--quiet` flags that adjust the effective log level.
- **FR-052**: The library source tree MUST NOT contain `print(` calls outside the CLI module's Rich-backed user-facing output.
- **FR-053**: The data loader MUST surface malformed rows via both WARNING-level logs and a structured load report returned to the caller.
- **FR-054**: A `bin-packer pack --explain` mode MUST emit a per-box placement trace at DEBUG level, documenting which bin and shelf was tried and why a placement failed if it did.

**Constraint Framework (US7)**

- **FR-060**: The library MUST provide a constraint abstraction that every algorithm consults before accepting a candidate placement.
- **FR-061**: The library MUST support per-box allowed orientations as a declarative subset of the six possible orientations (generalising the current all-or-none rotation flag).
- **FR-062**: The library MUST support a per-box maximum supported weight; any placement above a box whose cumulative overhead weight would exceed the limit MUST be rejected by every algorithm.
- **FR-063**: The constraint architecture MUST be extensible such that a future stability constraint (boxes must rest on the bin floor or on top of another box) can be added without changing the base algorithm interface.
- **FR-064**: An ADR MUST document the constraint architecture and enumerate constraints implemented now versus deferred.

**Distribution & Release Engineering (US8)**

- **FR-070**: On a Git tag matching `v*.*.*`, a release workflow MUST build the wheel and sdist, run the full test suite, and publish to PyPI via a trusted publisher (OIDC) — no long-lived API tokens in repository secrets.
- **FR-071**: The package MUST be installable via `pipx install <name>` (where `<name>` is `bin-packer-3d` or a documented alternative if the primary name is unavailable on PyPI) and expose the `bin-packer` command.
- **FR-072**: A `Dockerfile` MUST use a multi-stage build and run as a non-root user; the image MUST be published to a public free-tier container registry.
- **FR-073**: The versioning policy MUST be documented, explicitly defining the project-specific meaning of MAJOR, MINOR, and PATCH.
- **FR-074**: For every release, the Git tag, the `CHANGELOG.md` entry, and the PyPI artefact version MUST match.

**Interactive Demo (US9)**

- **FR-080**: The README MUST contain exactly one link to a zero-install demo that renders a 3D packing visualisation within 30 seconds of clicking.
- **FR-081**: If the demo is interactive (Streamlit, Gradio, or equivalent on a free-tier host), a visitor MUST be able to upload a CSV, select an algorithm, and trigger a packing run.

### Key Entities

- **Box** — a rectangular object to pack. Attributes: dimensions (length, width, height), weight (with distinguishable "unknown" state), allowed orientations (a subset of the six possible orientations), maximum supported weight, identifier and description.
- **Bin** — a rectangular container. Attributes: dimensions, maximum weight capacity (optionally "unlimited"), current utilisation.
- **Placement** — a box positioned and oriented inside a bin. Attributes: coordinates, orientation, parent bin identifier. Invariants: no two placements in the same bin overlap; all placements lie within the bin's bounds.
- **PackingResult** — the outcome of running an algorithm. Attributes: placements, unpacked boxes, per-bin utilisation, overall success rate, algorithm name, runtime.
- **Algorithm** — a registered packing strategy. Attributes: registered name, complexity class, short description, execution callable. Discovered via a single algorithm registry.
- **Constraint** — a rule consulted by every algorithm before accepting a candidate placement. Examples: allowed-orientations constraint, supported-weight constraint, (future) stability constraint.
- **LoadReport** — the outcome of loading data from CSV or Excel. Attributes: boxes parsed, rows skipped with structured reasons, warnings and errors.
- **BenchmarkResult** — per-(algorithm, instance) metrics. Attributes: utilisation %, bins used, success rate, runtime, algorithm name, instance name, seed.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A first-time visitor opening the GitHub repository sees, within 30 seconds: a green CI badge, a coverage badge showing ≥ 90%, a PyPI version badge, a hero visualisation, a two-sentence problem statement, and an install command.
- **SC-002**: On a clean machine with a modern Python toolchain, `pipx install bin-packer-3d && bin-packer pack DATASETS/sample_boxes.csv` produces a result and an HTML visualisation without additional setup.
- **SC-003**: Test coverage on `src/bin_packer_3d/` is ≥ 90% line coverage and is enforced in CI on every push.
- **SC-004**: The documentation site is reachable from a link in the README, builds in CI, and contains a reference page for every algorithm the runtime can execute.
- **SC-005**: The accepted strategy values on the packer configuration equal, exactly, the set of runtime-registered algorithm keys — asserted by an automated test.
- **SC-006**: A reference benchmark instance is executed in CI on every push and produces a JSON artefact; at least one newly-implemented algorithm demonstrates a utilisation delta of ≥5 percentage points OR a bin-count delta of ≥1 bin from First-Fit Decreasing on that instance.
- **SC-007**: A new contributor following `CONTRIBUTING.md` goes from `git clone` to a passing test run in five commands or fewer.
- **SC-008**: No file in `src/bin_packer_3d/` contains a `print(` call that should be a log call (search is automated in CI or via pre-commit).
- **SC-009**: The legacy `CODE/` directory no longer exists on `main` (history may be preserved via a tag).
- **SC-010**: No `.xlsx` file in `DATASETS/` contains business-confidential data; any that once did has been either anonymised or removed AND scrubbed from git history.
- **SC-011**: `CHANGELOG.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, and at least three ADRs exist on `main` and are linked from the README.
- **SC-012**: The most recent commit on `main` associated with a Git tag matches a published PyPI release of the same version.
- **SC-013**: A reviewer can open every directory at the repository root and identify its purpose from its top-level `README.md` within 10 seconds.
- **SC-014**: A representative pull request that deliberately introduces a lint violation, a type error, or a failing test is blocked from merging — verified during the acceptance process.

## Out of Scope

The following are explicitly NOT in scope for this phase and MUST NOT be addressed by the implementation:

- A web UI or SaaS deployment of the packing service.
- Distributed or parallelised packing across multiple machines.
- Exact / branch-and-bound ILP solvers (Gurobi, CPLEX, OR-Tools CP-SAT).
- Reinforcement-learning or machine-learning-based packers.
- Multi-bin-type heterogeneous fleet optimisation (a single bin geometry per run is acceptable; the framework must not preclude adding this later).
- A REST API (deferred to a later spec phase).
- Internationalisation of the CLI or documentation (English only).
- Backwards compatibility with the legacy `CODE/` scripts — they will be removed or relocated.
- **Native extensions (Rust, C, Cython-compiled units, or any component whose distribution requires platform-specific wheels).** Deferred until a concrete, measured profile identifies a hot path that neither NumPy vectorisation nor Numba/Cython JIT can resolve. The preferred path when that threshold is crossed is Rust via PyO3 + maturin, introduced via a separate spec phase and backed by an Architecture Decision Record that documents the flamegraph, the measured speedup against cheaper alternatives, and the wheel-distribution plan.

## Dependencies

- **US5 → US1**: Algorithm portfolio expansion depends on the algorithm registry established by the contract-integrity story.
- **US8 → US5**: The first PyPI release should ship with the full algorithm portfolio so that the public version represents the credible final state, not an interim one.
- **P1 bundle (US1, US2, US3, US4)**: These four stories are individually testable but SHOULD ship together to meet the first-impression credibility goal. Releasing any one in isolation does not move the project's CV-grade credibility meaningfully.
- **Existing codebase**: The spec assumes the current `src/bin_packer_3d/` implementation (FFD algorithm, shelf-based packer, Pydantic-based configuration, Click CLI, Plotly visualisation, pandas-based CSV/Excel loading, and pytest suite) is the baseline and will be extended — not rewritten.

## Assumptions

- **Language and runtime**: Python 3.11+ is the public floor (implementation MUST update `requires-python` from the current `>=3.10` to `>=3.11`); the maintainer develops on Python 3.14.3; the CI matrix covers 3.11, 3.12, 3.13, and 3.14 on Linux at minimum.
- **License**: MIT (unchanged from the current `LICENSE` and `pyproject.toml`).
- **Maintainer capacity**: a single solo maintainer (Bruno Ghiberto). Tooling MUST NOT require paid services beyond the GitHub free tier, PyPI free tier, and free-tier coverage / demo hosts.
- **Build system**: `hatchling` is preserved unless an implementation-phase finding identifies a compelling reason to switch.
- **Type safety**: strict static type checking (as currently configured with `mypy`) continues to pass.
- **Style**: existing `ruff` + `black` configuration (line length 100) continues to be enforced.
- **Public API stability**: the project is pre-1.0; breaking changes to existing public symbols in `bin_packer_3d` are permitted but MUST be documented in `CHANGELOG.md`.
- **Dataset privacy**: any `.xlsx` files in `DATASETS/` that contain identifiable real-world business data are treated as a privacy incident, not a hygiene issue — they MUST be anonymised or removed from both the working tree and git history before the phase closes.
- **No vendor lock-in**: no proprietary CI, no closed-source observability, no paid coverage service.
- **Documentation tool choice**: either MkDocs Material or Sphinx is acceptable; the spec does not prescribe. The plan phase selects based on ergonomics.
- **PyPI name availability**: if `bin-packer-3d` is unavailable on PyPI, a near-equivalent alternative MUST be chosen and documented in the release notes and README badges.
- **Benchmark instance redistribution**: bundled benchmark instances ship inside the repository when the instance license allows; otherwise a downloader script is provided.
- **Demo hosting choice**: the US9 demo can be either a static GitHub Pages page with a pre-rendered visualisation or a small app on a free-tier host; the spec does not prescribe.
- **Target audiences**: four primary audiences evaluate different artefacts — recruiters (README, repo summary), senior engineers (ADRs, CI workflows, test suite), operations-research practitioners (benchmark results, algorithm docs), open-source contributors (CONTRIBUTING, issue templates). Every artefact in this spec serves at least one of these audiences.
