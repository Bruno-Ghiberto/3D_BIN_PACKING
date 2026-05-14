# Feature Specification: Portfolio Polish of `bin-packer-3d`

**Feature Branch**: `002-portfolio-polish`
**Created**: 2026-05-13
**Status**: Draft
**Input**: User description: "Read @Speckit-context-prompts/spec-02-portfolio-polish/01-specify.md" — take `bin-packer-3d` in its current v0.2.0.dev0 state and deliver a presentation-layer polish pass that makes the repository, the artefacts it generates, and the running experience credible as a portfolio piece worth citing on a CV. The algorithm roadmap (US5 Parts 2-3) continues on its own track in parallel; this phase ships the project as-is, polished.

## Clarifications

### Session 2026-05-13

- Q: Should the TUI (User Story 6 / FR-035) ship in spec-02? → A: Defer to a future phase (v0.4 candidate).
- Q: How should the headline dataset (FR-034) be acquired? → A: Procedurally generated — commit generator script + seed + resulting CSV; byte-identical regeneration from the committed seed.
- Q: What version-tagging behaviour applies at end of phase (FR-033)? → A: Bump `pyproject.toml` to `0.3.0rc1` (PEP 440) and cut git tag `v0.3.0-rc1` on the polish PR merge commit. `v0.3.0` stable reserved for when US5 Parts 2-3 also land. PyPI publication remains out of scope.
- Q: What screenshot strategy applies to the README algorithm gallery (FR-005)? → A: Same dataset across all algorithms — comparison gallery using the headline procedural dataset (FR-034) packed by each registered algorithm, side-by-side, to make the visual difference between algorithms directly readable.
- Q: What accessibility baseline applies to spec-02 artefacts? → A: Baseline a11y — alt text on every README image (FR-036), colourblind-safe visualisation palette verified against deuteranopia/protanopia simulation (FR-037), and mkdocs-material WCAG-AA defaults preserved in the docs site (FR-038). Full automated WCAG-AA audit (Pa11y / axe-core CI gate) deferred to a future phase.

---

## User Scenarios & Testing *(mandatory)*

<!--
  Priorities: P1 = first-impression credibility (must ship together to make
  the CV pitch land), P2 = depth and demonstrability, P3 = reach and
  professional finish. Each story is independently testable.
-->

### User Story 1 - First-Screen README Credibility (Priority: P1)

As a recruiter, hiring manager, or peer reviewer landing on the public repository, I want the README's first screen to communicate — within 30 seconds and without running anything — what problem the project solves, what it offers, how to install and run it, and what a packed bin looks like, so that I can decide within a single scan whether to dig deeper or move on.

**Why this priority**: The README is the highest-leverage surface in the polish phase. A recruiter who closes the tab after 10 seconds never sees the CI badges, the property tests, or the 103-passing suite. Everything downstream in this phase only pays off if the first-screen scan succeeds.

**Independent Test**: Render the repository README on GitHub in a desktop browser. A first-time reader who has never seen the project should, within 60 seconds of scrolling only the first viewport-and-a-half, identify: (a) the problem domain (3D bin packing), (b) the three available algorithms by name, (c) the install command, (d) the run command, (e) at least one visual of a packed bin.

**Acceptance Scenarios**:

1. **Given** a fresh visitor opens the GitHub README, **When** they scan the first screen (above any fold), **Then** they see a hero asset (image, animated GIF, or embedded video) showing a packed bin, a one-paragraph problem statement explaining the 3D-BPP to a non-OR reader, and a one-paragraph "what this is" framing.
2. **Given** a visitor scrolls past the hero, **When** they reach the "Quick Start" section, **Then** they find a one-line install command and a one-line run command, both executable as-shown against the bundled sample dataset.
3. **Given** a visitor reads the README mid-scroll, **When** they reach the "Algorithms" section, **Then** they find a comparison table listing every algorithm registered in `ALGORITHMS` with its complexity, typical use case, and a one-line description; the table is sourced from the registry (not hardcoded).
4. **Given** a visitor wants to see results without running anything, **When** they scroll to the "Gallery" or "Showcase" section, **Then** they find a side-by-side comparison gallery: one static screenshot per registered algorithm, all packing the same headline dataset (FR-034), each labelled with the algorithm name; the visual difference between screenshots directly reflects the algorithmic difference.
5. **Given** the visitor is a CV reader specifically, **When** they reach the closing sections, **Then** they find a "Why I built this" paragraph framed for a non-technical reader and a "Highlights" section enumerating the project's technical credentials (test count, CI checks, property-based testing, strict typing, supported Python versions, algorithm count).

---

### User Story 2 - Polished, Reproducible Visualisations (Priority: P1)

As a reviewer who runs the `bin-packer pack` command on a sample dataset, I want the generated visualisation to look professionally branded, present the run metrics inside the artefact itself, and produce reproducible colours and a static export, so that what I see locally matches what is in the README screenshots and what I can embed in a slide deck or share by link.

**Why this priority**: The Plotly HTML is the project's signature artefact. Today it works but renders with stock Plotly defaults, has no embedded run summary, uses non-deterministic colours, and emits a 4.8 MB standalone file with no static fallback. A reviewer who screenshots the current visualisation gets something that looks like a Plotly tutorial output — not a finished product.

**Independent Test**: Run `bin-packer pack <headline-dataset> --strategy bfd --visualize -o /tmp/test-viz` on a fresh checkout. The resulting `bin_1.html` must open in a browser, display the project's branded theme, show a stats overlay panel with the run's metrics, and assign each box a colour that is identical to the colour the same box gets on a re-run with the same input. A static PNG (or SVG) snapshot of the bin must also exist in the output directory.

**Acceptance Scenarios**:

1. **Given** the same input dataset is packed twice with the same strategy, **When** both visualisations are compared, **Then** every box has the same colour in both runs (deterministic colour assignment).
2. **Given** a visualisation HTML is opened in a browser, **When** the visitor inspects the figure, **Then** the title carries the project name and the algorithm used, axes are labelled in millimetres, the colour palette is the project's branded palette (not Plotly default), and the hover template reports box ID + dimensions + position.
3. **Given** a visualisation HTML is opened, **When** the visitor looks at the figure overlay, **Then** an embedded panel reports: algorithm name, boxes placed (`N/total`), overall utilisation %, runtime in milliseconds.
4. **Given** the `--visualize` flag is set, **When** the run completes, **Then** the output directory contains at least one static-format file (PNG or SVG) per bin, suitable for inclusion in a slide deck or printed report without requiring a browser.
5. **Given** the visualisation file sizes are inspected, **When** the standalone HTML is generated, **Then** the file is no larger than today's baseline (4.8 MB) — and a documented mode exists to emit a slimmer CDN-loaded variant for online viewing.

---

### User Story 3 - Published Public Documentation Site (Priority: P1)

As a senior engineer, peer reviewer, or operations-research practitioner who wants to evaluate the project beyond the README, I want a published documentation site at a public URL that explains the problem domain, the architecture, each algorithm, the configuration surface, the visualisation system, and how to use the library, so that I can browse the project's depth without cloning the repository.

**Why this priority**: An open-source project without published documentation reads as half-finished, regardless of how good the code is. The repo already declares `mkdocs-material` as an optional dependency — publishing is the missing step.

**Independent Test**: Visit the published documentation URL in a browser. Confirm the site loads in under 3 seconds, lists sections for Quickstart, Algorithms (one page per registered packer), API reference, Configuration, Visualisation gallery, and Troubleshooting. Verify each algorithm page's content metadata (name, complexity, description) matches the runtime `ALGORITHMS` registry. Verify the README links to the site.

**Acceptance Scenarios**:

1. **Given** any commit lands on the `main` branch, **When** CI completes, **Then** the documentation site is rebuilt and published to a public URL, with the build status surfaced as a README badge.
2. **Given** a visitor opens the published documentation site, **When** the homepage loads, **Then** they see Quickstart, Algorithms, API reference, Configuration, Visualisation gallery, and Troubleshooting in the navigation.
3. **Given** a visitor opens an algorithm page (e.g. "Best-Fit Decreasing"), **When** they read it, **Then** the page contains the algorithm's complexity, a prose description, a worked example or visualisation, and a literature citation; the complexity and description fields agree with the runtime `ALGORITHMS` registry entry for that key.
4. **Given** a new algorithm is registered in the codebase, **When** the documentation site rebuilds, **Then** a page for the new algorithm appears automatically (or fails the build with a clear "missing docs page" error).
5. **Given** the README is rendered on GitHub, **When** a reader scans it, **Then** a "Documentation" link prominently points to the published site.

---

### User Story 4 - Tidy Repository Surface and Structure Map (Priority: P1)

As a reviewer who clones or browses the repository file tree, I want every visible top-level item to either earn its place at the top level with a clear purpose or be relocated below a clearly labelled subdirectory, so that I am never confused by the presence of a folder or file I cannot explain, and so that the project communicates its own architecture without me having to ask.

**Why this priority**: Currently `legacy/`, `Speckit-context-prompts/`, `LINKEDIN.txt`, and `DATASETS/` all appear at the root. Each has a reason to exist, but a first-time reviewer sees an inscrutable directory list and quietly downgrades their assessment. The fix is cheap (rename/move/document) and the payoff per second invested is enormous.

**Independent Test**: A first-time reader who clones the repo and runs `ls` (or browses the GitHub file tree) should, within 30 seconds, be able to map each visible top-level entry to one of: (a) "library source", (b) "tests", (c) "documentation", (d) "examples / sample data", (e) "build / CI configuration", (f) "project governance / metadata". The README's "Project Structure" section should explain the mapping without lying about what exists.

**Acceptance Scenarios**:

1. **Given** the repository root is listed, **When** a reviewer scans the top-level entries, **Then** every entry is accounted for in the README's "Project Structure" section (no orphan folders, no mystery files).
2. **Given** a top-level item exists today that no longer earns root-level visibility (e.g., `LINKEDIN.txt`, `Speckit-context-prompts/`), **When** the polish phase ships, **Then** the item is either relocated below a clearly named parent directory (`internal/`, `docs/`, etc.), excluded from the wheel build, removed entirely, or kept at root with an explicit README explanation.
3. **Given** the README "Project Structure" section is rendered, **When** a reviewer reads it, **Then** the tree shown matches the actual filesystem (no drift between docs and reality), and every shown entry has a one-line description.
4. **Given** a visitor opens the GitHub repository "About" block (sidebar), **When** they read it, **Then** they find a tagline summarising the project, at least three relevant topics (e.g., `bin-packing`, `operations-research`, `python`), and a link to the published documentation site.

---

### User Story 5 - Curated Examples and One-Command Demo (Priority: P2)

As a reviewer who wants to feel the project without reading documentation first, I want a curated `examples/` directory and a one-command demo that exercises every registered algorithm against a representative dataset, so that I can experience the project's full output in a single invocation and trust that the README screenshots are reproducible.

**Why this priority**: A live demo is the difference between "I read about it" and "I tried it". One scripted command lowers the bar to evaluation from "read 200 lines of docs first" to "type one thing and watch".

**Independent Test**: Run the demo command (e.g. `bin-packer demo`, `make demo`, or equivalent) on a fresh checkout. The command should pack at least one curated dataset through all three registered algorithms, emit a comparison report, save visualisations for each strategy, and complete in under 60 seconds on a typical developer laptop. Every screenshot and animated demo asset referenced in the README must be regeneratable by running this command.

**Acceptance Scenarios**:

1. **Given** the repository is cloned fresh and dependencies are installed, **When** the demo command is invoked with no arguments, **Then** it packs a documented curated dataset through every registered algorithm and emits a comparison report in the terminal.
2. **Given** the demo command completes, **When** the output directory is inspected, **Then** it contains: per-algorithm visualisations (HTML + static), a `placements.csv` per run, and a comparison summary file (text, JSON, or Markdown).
3. **Given** the `examples/` directory is inspected, **When** a visitor browses it, **Then** they find at least 3 distinct curated datasets ranging in difficulty (e.g., a "small / friendly" set, a "realistic mixed" set, and a "stress" set), each with a README describing its dimensions, intent, and expected qualitative result.
4. **Given** the README embeds an animated demo asset (GIF or short MP4), **When** a reader inspects how it was produced, **Then** a documented command exists in the repository that regenerates the asset from source.
5. **Given** any visualisation screenshot is shown in the README, **When** a reader wants to verify it, **Then** they can run the demo command (or a referenced sub-command) and reproduce the same screenshot up to deterministic colour-and-layout differences.

---

### User Story 6 - Interactive Terminal Experience (Deferred to v0.4)

**Status**: Deferred (clarified 2026-05-13). This user story is removed from spec-02 scope.

A polished TUI wrapper (`bin-packer tui` or equivalent) is a strong portfolio differentiator but was deferred to a future phase (v0.4 candidate) to keep the polish pass focused. Rationale: spec-02 ships best with fewer artefacts done excellently; adding Textual as a dependency, building cross-terminal compatibility (Linux / macOS / Windows Terminal), and authoring TUI-specific tests would dilute test budget and scope for the four P1 stories. The TUI option remains open for v0.4 once the CV submission cycle is in motion.

---

### User Story 7 - CV-Facing Project Identity (Priority: P3)

As the sole maintainer preparing to cite this project on a CV and on LinkedIn, I want a canonical short paragraph describing the project (≤80 words), a one-line tagline, and a curated set of "highlight" credentials stored in the repository, so that I can copy them into a CV bullet, a LinkedIn About section, or a job application without re-deriving them each time.

**Why this priority**: This polish item is owner-facing rather than visitor-facing, but it closes the loop on the CV intent that motivates the whole phase. Without committed canonical copy, the user re-improvises the project pitch each time it's referenced — risking inconsistency between LinkedIn, CV, GitHub About, and README.

**Independent Test**: Open the repository's `docs/about.md` (or equivalent). Within 30 seconds, the user can copy: a ≤80-word project paragraph, a one-line tagline (≤120 characters), and a bullet list of technical highlights ready for a CV.

**Acceptance Scenarios**:

1. **Given** the maintainer needs to update a CV bullet, **When** they open the canonical project identity file, **Then** they find a ≤80-word paragraph that names the project, the problem, the algorithms, and the engineering practices that distinguish it.
2. **Given** the GitHub repository "About" block is populated, **When** a visitor reads it, **Then** the tagline matches the canonical tagline in the repository's identity file (no drift).
3. **Given** the README's "Highlights" section is rendered, **When** a visitor scans it, **Then** they see at minimum: registered algorithms count, test count, property-based testing presence, strict-mypy presence, supported Python versions, CI check count, license.
4. **Given** the maintainer wants suggested LinkedIn promo copy, **When** they look in the repository, **Then** a `docs/promo/` directory (or equivalent) exists with at least one suggested post draft.

---

### Edge Cases

- **A new algorithm is registered after the polish phase ships.** The README's algorithm comparison table, the docs site, and the demo command must all pick it up automatically by sourcing from the `ALGORITHMS` registry. No hardcoded algorithm names anywhere in polish-phase artefacts.
- **A reviewer clones on Windows or macOS.** The demo command, the visualisation generation, and the documentation build must all succeed on the supported matrix (Python 3.11–3.14). (TUI cross-platform concern is deferred — see Clarifications.)
- **A reviewer is offline.** The standalone HTML visualisation must continue to work without internet (no CDN-only mode without an offline fallback). The published docs site is naturally online-only; this is acceptable as long as the README's own content is self-contained.
- **The docs site build fails on push to `main`.** CI must surface the failure as a blocking check (consistent with constitution §III); a broken build must never leave the published site in an inconsistent state.
- **A visual claim in the README cannot be reproduced.** This is a Contract Honesty (constitution §I) violation. Every screenshot, GIF, and benchmark number in the README must be regeneratable by a documented command. If a number drifts (e.g., utilisation % on the headline dataset changes after an algorithm tweak), CI must catch the drift or the screenshot must be regenerated and recommitted.
- **The headline sample dataset is too dense and exposes a packer bug.** If a polish-phase dataset choice surfaces a latent bug in BFD / FFD / Shelf, the bug is logged but NOT fixed in this phase (algorithm work is out-of-scope per §5 of the context prompt); the polish phase either picks a different dataset or documents the known limitation.
- **A new dependency (Kaleido, Textual) is unavailable on one of the supported Python versions.** The new dependency MUST be declared under `[project.optional-dependencies]` with a clear opt-in installation path, so that the core install footprint remains unchanged.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The README MUST present a hero visual (animated GIF, embedded video, or static image showing a packed bin) within the first viewport on a desktop browser.
- **FR-002**: The README MUST contain a one-paragraph (≤200 words) plain-language explanation of the 3D Bin Packing Problem, comprehensible to a reader with no operations-research background.
- **FR-003**: The README MUST contain an install one-liner and a run one-liner, each executable verbatim against the bundled headline sample dataset.
- **FR-004**: The README MUST contain an algorithm comparison table whose row count equals `len(ALGORITHMS)` and whose per-row content (strategy key, complexity class, description) is sourced from the runtime registry (no hardcoded values).
- **FR-005**: The README MUST display a comparison gallery: one static screenshot per registered algorithm in `ALGORITHMS`, each packing the same headline procedural dataset (FR-034), labelled with the algorithm name. All gallery screenshots MUST share the same dataset so that the visual difference between images directly reflects algorithmic difference (clarified 2026-05-13).
- **FR-006**: The visualisation system MUST assign colours to boxes deterministically — identical input + identical strategy yields identical per-box colour assignments across runs and across machines.
- **FR-007**: The visualisation system MUST apply a branded theme to every emitted figure: axis labels in millimetres, project-specific colour palette, hover template reporting box ID + dimensions + position, and a title naming the algorithm.
- **FR-008**: Each emitted visualisation MUST embed a stats overlay reporting algorithm name, boxes placed (`N/total`), overall utilisation %, and runtime in milliseconds.
- **FR-009**: For every interactive HTML emitted, the visualisation system MUST also emit a static-format file (PNG or SVG) suitable for embedding in README, slide decks, or printed reports.
- **FR-010**: The interactive HTML file size MUST NOT exceed the current baseline (4.8 MB) on equivalent input; a documented mode (e.g., `--cdn`) MAY emit a slimmer variant for online viewing only.
- **FR-011**: A documentation site MUST be published at a public URL automatically on every merge to `main`.
- **FR-012**: The published documentation site MUST contain, at minimum, the following sections: Quickstart, Algorithms (one page per registered packer), API Reference, Configuration, Visualisation Gallery, Troubleshooting.
- **FR-013**: Each algorithm page on the documentation site MUST source its metadata (name, complexity, description) from the `ALGORITHMS` registry, identical to the README comparison table's source.
- **FR-014**: The documentation site's build MUST fail (and block the merge or rollback the deploy) if any registered algorithm lacks a corresponding documentation page.
- **FR-015**: The README MUST contain a prominently placed link to the published documentation site.
- **FR-016**: The README MUST contain a "Project Structure" section whose tree representation exactly matches the actual top-level filesystem layout; every entry MUST have a one-line description.
- **FR-017**: Every current top-level repository entry (including `legacy/`, `Speckit-context-prompts/`, `LINKEDIN.txt`, `DATASETS/`) MUST be either relocated below an explanatory subdirectory, excluded from the wheel build, removed entirely, or kept at root with an explicit README explanation. No unexplained top-level entries.
- **FR-018**: The GitHub repository's "About" block MUST contain a tagline, at least three relevant topics, and a link to the published documentation site.
- **FR-019**: The repository MUST contain an `examples/` directory with at least three curated datasets, each accompanied by a README describing the dataset's dimensions, intent, and expected qualitative result.
- **FR-020**: The repository MUST expose a one-command demo (e.g., `bin-packer demo`, `make demo`, or equivalent) that packs a documented headline dataset through every registered algorithm and emits per-algorithm visualisations and a comparison summary.
- **FR-021**: The demo command MUST complete in under 60 seconds on a typical developer laptop (consumer-grade CPU, no GPU, no special hardware).
- **FR-022**: The README MUST embed an animated demo asset (GIF or short MP4) demonstrating the demo command's output.
- **FR-023**: Every visual claim in the README (screenshots, GIF, MP4, quoted numbers) MUST be regeneratable by a documented command in the repository.
- **FR-024**: The repository MUST contain a canonical project-identity file (e.g., `docs/about.md`) holding a ≤80-word project paragraph and a one-line tagline (≤120 characters).
- **FR-025**: The README MUST contain a "Highlights" section listing the project's technical credentials, including at minimum: registered-algorithm count, test count, property-based testing presence, strict-mypy presence, supported Python versions, CI-check count, and license.
- **FR-026**: The README's "Highlights" section MUST agree numerically with the live state of the repository (CI checks count, test count, supported Python versions) at the time of merge; drift is a Contract Honesty (§I) violation.
- **FR-027**: Every new dependency introduced by this phase (e.g., Kaleido, Textual) MUST be declared under `[project.optional-dependencies]` with a documented installation path; the default `pip install bin-packer-3d` install footprint MUST NOT grow by more than 5%.
- **FR-028**: Every new dependency introduced by this phase MUST be installable on Python 3.11, 3.12, 3.13, and 3.14 on Linux.
- **FR-029**: Every new public symbol introduced by this phase MUST carry a Google-format pydocstyle-compliant docstring, consistent with `001-public-release-hardening` FR-026.
- **FR-030**: The CHANGELOG MUST be updated under `[Unreleased]` with one entry per shipped polish item.
- **FR-031**: The test suite MUST grow by at least N tests, where N is the count of new public symbols, commands, or surfaces introduced by this phase; the existing 103-passing suite MUST NOT regress.
- **FR-032**: Coverage on `src/bin_packer_3d/` MUST remain ≥90% (constitution §III floor).
- **FR-033**: At end of phase, the project version MUST bump from `0.2.0.dev0` to `0.3.0rc1` (PEP 440 release-candidate suffix) in `pyproject.toml`, and a git tag `v0.3.0-rc1` MUST be cut on the polish PR merge commit (clarified 2026-05-13). Tag-release notes MUST cite the spec-02 user stories shipped and link to the published documentation site.
- **FR-034**: The headline sample dataset for README screenshots and demos MUST produce a BFD pack of utilisation ≥60% on the default bin dimensions; the current `DATASETS/sample_boxes.csv` (11% utilisation) MUST NOT be the headline asset. The headline dataset MUST be procedurally generated (clarified 2026-05-13): the repository MUST commit (a) a generator script (e.g., `scripts/generate_headline_dataset.py`), (b) the seed and parameters used, and (c) the resulting CSV file. Regenerating the CSV from the committed seed MUST produce a byte-identical result.
- **FR-035**: TUI requirements are deferred to a future phase (v0.4 candidate, clarified 2026-05-13). No TUI-related requirements apply to spec-02.
- **FR-036**: Every embedded image in the README (hero asset, screenshots, comparison gallery, GIF/MP4) MUST carry alt text describing the visualisation content for screen-reader accessibility (clarified 2026-05-13).
- **FR-037**: The visualisation colour palette (per `VisualisationStyle`) MUST be colourblind-safe: no per-box differentiation that relies solely on red-vs-green hue contrast. The palette MUST be verified against deuteranopia and protanopia simulation tooling at theme-design time, and the verification MUST be documented in the repository (clarified 2026-05-13).
- **FR-038**: The published documentation site (US3) MUST preserve mkdocs-material's WCAG-AA-conformant defaults. Any custom theme override (palette, typography, contrast) MUST be verified against the same contrast ratios before merge; the verification MUST be documented (clarified 2026-05-13).

### Key Entities

- **AlgorithmCardEntry**: One row in the README comparison table and one page on the documentation site per registered algorithm. Sourced from `ALGORITHMS` registry. Attributes: strategy key, complexity class, short description, typical-use-case prose, literature citation, link to docs page.
- **VisualisationStyle**: The branded Plotly theme applied to every emitted figure. Attributes: colour palette (deterministic, hash-derived, colourblind-safe per FR-037), axis-label format, hover template, title format, stats-overlay layout.
- **DemoArtifact**: Captured output of a single documented demo run. Attributes: source dataset path, algorithm strategy key, captured-screenshot path, generated-HTML path, generated-static-export path, run metrics (utilisation, runtime, boxes placed).
- **DocumentationPage**: One markdown page in the published docs site. Attributes: section (Quickstart / Algorithms / API Reference / etc.), slug, source path under `docs/`, last-build status, links to related entries.
- **ProjectIdentityArtefact**: The canonical identity surface used across README, GitHub About block, and external promotional material. Attributes: ≤80-word paragraph, ≤120-character tagline, technical-highlight bullet list, suggested LinkedIn copy.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A first-time reviewer can identify the project's problem domain, available algorithms, install command, run command, and a visual of a packed bin in under 60 seconds of scrolling only the README's first viewport-and-a-half. Validated by a 3-person user-test panel; at least 2 of 3 must succeed within the time budget.
- **SC-002**: The headline demo command, run on a fresh checkout, completes in under 60 seconds on a consumer-grade laptop and emits a BFD pack of overall utilisation ≥60% on the curated headline dataset.
- **SC-003**: The published documentation site loads its homepage in under 3 seconds (measured with a cold browser cache on a 50 Mb/s connection) and is reachable at a public URL linked from the README.
- **SC-004**: 100% of registered algorithms (i.e., every key in `ALGORITHMS`) appear in the README comparison table and have a corresponding page on the documentation site at merge time.
- **SC-005**: 100% of visual claims (screenshots, GIF, MP4, quoted numbers) in the README are reproducible by running a documented command in the repository.
- **SC-006**: Every emitted visualisation HTML on identical input produces identical per-box colour assignments across two consecutive runs (deterministic colour test, automatable).
- **SC-007**: The default `pip install bin-packer-3d` install footprint grows by no more than 5% relative to the v0.2.0.dev0 baseline.
- **SC-008**: The test suite contains at least 103 + N tests at merge time, where N counts new public symbols/surfaces introduced by this phase. Coverage on `src/bin_packer_3d/` remains ≥90%.
- **SC-009**: CI passes 10/10 checks on the polish PR at merge time, including any new docs-site-build check introduced by this phase.
- **SC-010**: The GitHub repository "About" block contains a tagline, ≥3 topics, and a link to the documentation site at merge time.
- **SC-011**: Every top-level repository entry visible in the rendered GitHub tree is accounted for in the README's "Project Structure" section, with no orphan entries and no drift between docs and reality.
- **SC-012**: A `docs/about.md` (or equivalent) file containing a ≤80-word canonical project paragraph and a ≤120-character tagline exists in the repository at merge time.
- **SC-013**: A git tag `v0.3.0-rc1` is cut on the polish PR merge commit; `pyproject.toml` declares `version = "0.3.0rc1"` at merge time; the tag's release notes cite the spec-02 user stories shipped and link to the published documentation site.
- **SC-014**: 100% of README-embedded images carry alt text (FR-036); the visualisation palette passes a documented colourblind-simulation check covering deuteranopia and protanopia (FR-037); the docs site retains mkdocs-material WCAG-AA contrast levels at merge time (FR-038).

## Assumptions

- **Audience.** The CV audience for this polish phase is English-speaking. No Spanish or multilingual artefacts are produced. The project remains bilingual-friendly at the maintainer-prose level, but committed documentation is English.
- **Reading path.** Reviewers primarily reach the repository via a direct link from a CV, a LinkedIn post, or a peer recommendation — not via search engines. README-first design is therefore optimal; SEO is not a priority.
- **Documentation framework.** The published documentation site uses `mkdocs-material`, the framework already declared as an optional dependency. Switching frameworks is out of scope.
- **Visualisation framework.** The visualisation continues to be built on Plotly. Switching to a different rendering engine (matplotlib, three.js, etc.) is out of scope. Static export of Plotly figures uses Kaleido unless clarification specifies otherwise (Kaleido is the canonical Plotly static-export backend).
- **Demo-asset generation toolchain.** Animated terminal demos are recorded with `vhs` (Charmbracelet's Tape-based, scriptable recorder) unless clarification specifies otherwise. `vhs` supports macOS and Linux natively; on Windows the maintainer records via WSL or Linux CI.
- **Documentation deployment trigger.** The documentation site rebuilds and publishes on every push to `main` only. PR-preview deployments are out of scope for this phase.
- **README hero asset.** The hero is an animated GIF by default (broadest GitHub-renderer compatibility). MP4 may be substituted if file-size or quality concerns dominate.
- **Headline sample dataset.** The headline dataset for README screenshots and demo runs produces a BFD pack of utilisation ≥60% on the default bin dimensions (860×890×1040 mm). Procedurally generated via a committed script and seed (FR-034, clarified 2026-05-13); regenerating from the committed seed yields byte-identical CSV output. Literature-cited datasets (e.g., Berkey-Wang BR-class instances) are deferred to US5 Part 2-3, where they serve as benchmark instances for algorithm comparison.
- **Parallel track.** US5 Parts 2-3 work (Extreme Point, Maximal Rectangles, benchmark CI gate) continues on `feature/us5-extreme-point-benchmark` in parallel. The polish phase MUST NOT block, depend on, or anticipate that work. If US5 Part 2 lands during this phase, the polish phase incorporates the new algorithm into the README comparison table and docs site via the registry-driven sourcing requirement (FR-004, FR-013), without requiring spec amendments.
- **Versioning.** The phase ends with a `v0.3.0-rc1` git tag (FR-033, clarified 2026-05-13) and a `pyproject.toml` bump to `0.3.0rc1` (PEP 440). PyPI publication remains out of scope; the tag is a GitHub-only signal of release-candidate readiness. The stable `v0.3.0` tag is reserved for the point at which US5 Parts 2-3 (Extreme Point + Maximal Rectangles + benchmark CI gate) also land.
- **Constitution compliance.** All 8 Core Principles apply. In particular: Contract Honesty (visualisation, README, and docs site never assert anything the runtime cannot verify), Test-First Discipline (every new public symbol carries tests authored first), Automated Quality Gates (CI stays green), Documentation (every new public surface documented).
- **Test budget.** Polish artefacts are tested at the level appropriate to their kind — visual assets via reproducibility-of-command tests (`SC-005`), interactive surfaces via integration tests, documentation completeness via "every algorithm has a page" assertion (`FR-014`).
- **Accessibility baseline.** Polish phase commits to a minimal a11y bar (alt text on all README images, colourblind-safe visualisation palette, mkdocs-material WCAG-AA preserved in the docs site — FR-036 through FR-038, clarified 2026-05-13). Full automated WCAG-AA audit (Pa11y / axe-core CI gate) is out of scope for this phase and a valid future-work item.
- **Branch and integration.** Work proceeds on branch `002-portfolio-polish`, integrated via the Git-Flow-lite `develop → main` flow already established by `001-public-release-hardening`.
