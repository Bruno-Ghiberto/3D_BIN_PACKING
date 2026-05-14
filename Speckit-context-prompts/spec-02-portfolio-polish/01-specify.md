# Specify Context — Phase 02: Portfolio Polish of `bin-packer-3d`

> Hand this entire document to `speckit.specify` as the feature description.
> SpecKit will turn it into a rigorous spec.md with user stories, functional
> requirements, success criteria, and edge cases. The goal of THIS document is
> to give SpecKit enough signal that it never emits `[NEEDS CLARIFICATION]`.

---

## 0. Brief

Take `bin-packer-3d` in its current state — a working Python 3D Bin Packing
library with three algorithms (`bfd`, `ffd`, `shelf`), 103 passing tests, full
CI, MIT license, and v0.2.0.dev0 — and produce a **presentation-layer polish
pass** that makes the repository, the artefacts it generates, and the running
experience credible as a portfolio piece worth citing on a CV.

This is **NOT** a feature-completion pass. The algorithm roadmap (US5 Parts 2-3:
Extreme Point + Maximal Rectangles + benchmark CI gate) stays untouched and
continues on its own track. This phase ships the project as-is, but polished
to the point where a recruiter or peer reviewer reading the repo cannot tell
that any work is "missing" without explicitly looking for what was deferred.

The user is the sole maintainer (Bruno Ghiberto). The audience for this phase
is: (a) recruiters and hiring managers who will skim for ≤30s and decide if
they'll dig deeper, (b) senior engineers who will spend 5–10 minutes reading
README + a code sample, and (c) the user himself, who will link to this
repository from a CV and a LinkedIn profile.

---

## 1. Why Now

- The previous phase (`001-public-release-hardening`, merged in PR #1) shipped
  the engineering substrate: CI, type discipline, test suite, contract
  honesty, documentation seeds. The repo is now technically credible.
- What it is **not yet** is *presentationally* credible. The README is
  functional but generic. Visualisations are emitted as 4.8 MB Plotly HTML
  files that cannot preview on GitHub. There is no animated demo, no
  screenshot gallery, no quickstart that fits on one screen.
- The user wants this repository to anchor a CV submission cycle. The polish
  pass is therefore time-bounded and presentation-focused: every change must
  improve what a non-running reader sees and what a 30-second runner
  experiences.
- US5 Parts 2-3 will continue on `feature/us5-extreme-point-benchmark` after
  this phase merges. The polish pass MUST NOT block, depend on, or anticipate
  that work.

---

## 2. Current State Snapshot

A faithful baseline. The spec must NOT assume anything beyond what is described
here.

### 2.1 What works today

- **Three registered algorithms**: `bfd` (Best-Fit Decreasing), `ffd`
  (First-Fit Decreasing), `shelf` (shelf-based). All discoverable via
  `bin-packer info`. All `O(n log n)`.
- **End-to-end CLI flow** verified: `bin-packer pack <file> --strategy
  <name>` ingests CSV/XLSX, packs, emits `placements.csv` and (with
  `--visualize`) a `bin_N.html` per bin used.
- **103 passing tests, 1 skipped** across unit / integration / property tiers.
  Hypothesis property suite checks 4 invariants × 3 algorithms.
- **CI** running on Python 3.11–3.14, Linux. Ten checks gating PRs to `main`.
  Ruff, ruff-format, mypy strict, pytest, pre-commit parity, coverage upload.
- **Documentation seed**: `docs/maintainers.md`, `docs/quickstart.md`,
  `CHANGELOG.md` (Keep-a-Changelog), `LICENSE` (MIT), `CONTRIBUTING.md` stub.
- **Live badges** in README: build, coverage, license, Python versions.
- **Constitution** at `.specify/memory/constitution.md` with 8 Core Principles
  (4 non-negotiable). All work in this phase MUST comply.

### 2.2 What is rough on the surface

- **README** is competent but reads like a template. No hero image, no demo
  GIF, no problem-statement paragraph that explains the 3D-BPP to a
  non-OR-literate reader, no "why this exists" framing, no comparison table
  between the three algorithms.
- **Visualisation output** is a 4.8 MB standalone Plotly HTML per bin. It
  works locally but: (a) cannot be previewed inline on GitHub, (b) has no
  static fallback (PNG/SVG), (c) uses default Plotly theme (gridlines,
  background, hover style all stock), (d) lacks an overlay panel summarising
  the packing run, (e) box colours are arbitrary (Plotly default palette,
  non-deterministic across runs).
- **CLI ergonomics**: `bin-packer pack` works but is dry — no progress bar
  for large datasets, no summary card with the "wow" metrics (utilisation,
  algorithm, time) presented graphically. The Rich table is good; the rest
  is plain print.
- **No TUI**. A reviewer who clones the repo runs the CLI once with
  `--help` and the sample CSV, gets a 100% pack at 11% utilisation (because
  sample boxes are tiny vs the default bin), and moves on. There is no
  interactive surface that invites exploration: try a different strategy,
  swap a dataset, tweak bin dimensions, watch the pack evolve.
- **Repository top-level** still surfaces artifacts that may confuse a
  reviewer: `legacy/` (preserved Alpha scripts), `Speckit-context-prompts/`
  (planning-phase prompts), `LINKEDIN.txt`. Each has a reason to exist but
  none is communicated at the README level.
- **Documentation site not published.** `mkdocs-material` is a declared
  optional dependency under `[docs]`, but no GitHub Pages deployment exists.
  Anyone wanting to read docs has to clone and `mkdocs serve` locally.
- **No screenshots, no GIFs, no recorded demo** anywhere in the repo. A
  reviewer who does not run the code never sees the visualisation that is
  arguably the project's best demo asset.
- **The `DATASETS/sample_boxes.csv` packs to 11% utilisation** with the
  default bin. That number on a screenshot or README example would
  *undersell* the algorithm. The sample dataset chosen for demos must
  produce a result that looks impressive at a glance.

### 2.3 Explicit non-issues (do not address)

- Algorithm quality / approximation ratios — owned by US5 Parts 2-3.
- Test coverage gaps — the test suite is the work of `001-public-release-
  hardening` and any new code in this phase carries its own tests but
  does not retrofit coverage on existing code.
- Performance optimisation — `O(n log n)` is documented; that's the bar.
- Multi-language documentation — English only for now. The user is
  bilingual but the audience for the CV pass is English-speaking.

---

## 3. Audience and Reading Budget

The spec MUST design every artefact against these three reader profiles:

| Reader | Where they land | Time budget | What they need to leave with |
|---|---|---|---|
| Recruiter / HR | README header on GitHub | 30 seconds | "This person ships polished work. Worth a recruiter screen." |
| Senior engineer / peer reviewer | README → `src/` → tests | 5–10 minutes | "Architecture is clean. Tests are real. They know OR concepts. I'd hire/work-with them." |
| Operations-research practitioner | README → docs/algorithms | 10–20 minutes | "Algorithms match the literature. Benchmarks (when they land) will be honest. Worth following." |

Anything in the polished surface that requires explanation beyond these budgets
is over-scoped. Anything that fails to land within these budgets is under-baked.

---

## 4. In-Scope Themes

The spec should organise these into prioritised user stories. The user
expects ≥4 P1 stories, with P2/P3 for stretch.

### 4.1 README rewrite — first-screen hero

- Replace the current README opening with: hero image or animated GIF of a
  pack running, one-paragraph problem statement, one-paragraph "what this
  is", install one-liner, run one-liner, link to docs.
- Algorithm comparison table (FFD vs BFD vs Shelf) with complexity, typical
  use case, citation. Sourced from the `ALGORITHMS` registry where possible.
- Sectioned table-of-contents anchored navigation.
- Screenshot gallery: at least one static PNG of a packed bin per algorithm.
- "Why I built this" paragraph framed for the CV reader.

### 4.2 Visualisation polish

- Deterministic colour assignment per box across runs (hash from ID, or
  stable per-strategy palette).
- Custom Plotly theme: dark/light variants, branded title, axis labels in mm
  with units, hover template that reports box ID + dimensions + position.
- Stats overlay panel embedded in the HTML: algorithm name, boxes placed,
  utilisation %, runtime ms.
- Static export pathway: PNG or SVG snapshot of each bin, suitable for
  README and slide decks. (Plotly supports static export via Kaleido.)
- File-size sanity: investigate whether the 4.8 MB HTML can be slimmed
  (CDN-loaded Plotly vs inline, minified config). Acceptable to ship both
  modes (`--standalone` vs `--cdn`).
- Optional: a step-by-step animation across the packing sequence (slider or
  frame-by-frame) so a viewer can watch the algorithm work. This is the
  "demo-able" asset.

### 4.3 TUI for interactive exploration

**Status: candidate scope, to be confirmed during specification.**

A textual / urwid-based TUI that wraps `bin-packer pack`: lets the reviewer
pick a dataset, pick an algorithm, watch results render in-terminal, swap
parameters without retyping the CLI. Why this matters: turns the project
from "a CLI you read about" into "a tool you instinctively want to try".
Spec must decide: build now, defer, or drop.

If built, MUST: (a) reuse existing packer / loader / metrics code without
duplication, (b) work on Linux + macOS + Windows terminals, (c) not become
a new test-coverage debt centre, (d) be invokable from the same `bin-packer`
entry point (e.g. `bin-packer tui`).

### 4.4 Documentation site publication

- Publish `mkdocs-material` site to GitHub Pages on every merge to `main`.
- Site sections: Quickstart, Algorithms (one page per registered packer
  with prose + complexity + reference), API reference (`mkdocstrings`),
  Configuration, Visualisation gallery, Troubleshooting.
- README links prominently to the published site.
- Each algorithm page anchors back to its registry metadata, so docs and
  runtime stay aligned (FR-026 lineage from spec-01).

### 4.5 Demo assets and reproducibility

- A curated `examples/` directory with 3–4 datasets of increasing
  interest: tiny (current sample), realistic mixed, dense, edge-case
  (impossible-to-fit). Each with expected output captured.
- A scripted "demo run" command (`make demo` or `bin-packer demo`) that
  runs the chosen dataset through all three algorithms and emits a
  comparison report + screenshots.
- Animated GIF of the demo run, committed to `docs/assets/` and embedded
  in README. Tool: `vhs`, `asciinema-agg`, or equivalent — spec to decide.

### 4.6 Repository surface cleanup

- Decide for each top-level item whether it stays surfaced, moves under
  `docs/` or `internal/`, or is excluded from the rendered README tree:
  `legacy/`, `Speckit-context-prompts/`, `LINKEDIN.txt`, `DATASETS/`.
- README "Project Structure" section that maps the actual top level so a
  reviewer is never disoriented.
- Pinned issues / Discussions enabled with welcome posts (optional).
- Repository `About` block on GitHub: tagline, topics, website link to
  docs site.

### 4.7 CV-facing artefacts

- One canonical paragraph (≤80 words) describing the project, suitable for
  copying into a CV bullet or LinkedIn About section. Stored in
  `docs/about.md` or equivalent.
- Suggested LinkedIn post copy and one image asset, stored in `docs/promo/`
  (optional, not committed if user prefers private).
- A "Highlights" section in the README enumerating: "Property-based testing
  with Hypothesis · Strict mypy · 10-check CI · Three registered O(n log n)
  algorithms · Interactive 3D visualisation".

---

## 5. Out of Scope (Explicit)

- US5 Parts 2-3 work: Extreme Point and Maximal Rectangles algorithms,
  benchmark engine, SC-006 CI gate. These continue on their own track.
- Streamlit / web frontend / hosted live demo beyond static GitHub Pages.
- Translating any artefact into Spanish.
- Adding new test categories beyond what new polish features need.
- Performance optimisation of existing packers.
- Onboarding multiple maintainers, code-of-conduct enforcement automation,
  governance docs beyond what already exists.
- PyPI publication — that is reserved for a later release-engineering phase
  once US5 closes. This polish phase tags a `v0.3.0-rc` at most.

---

## 6. Constraints

- **Constitution compliance is non-negotiable.** All 8 Core Principles apply.
  In particular: Contract Honesty (visualisation must not lie about what was
  packed), Test-First Discipline (any new code carries tests), Automated
  Quality Gates (CI stays green), Documentation (every new public surface
  documented).
- **No regression of the 103-passing test suite.** Polish work that breaks a
  test is rolled back, not merged with the broken test deleted.
- **Wheel slimness preserved.** `tool.hatch.build.exclude` already keeps
  `legacy/`, `DATASETS/*.xlsx`, `docs/`, `tests/`, `Speckit-context-prompts/`
  out of the distributable. New polish artefacts must declare their build
  status: shipped in wheel, source-only, or repo-only.
- **Cross-Python compatibility preserved**: 3.11, 3.12, 3.13, 3.14 Linux.
  Any new dependency (e.g. Kaleido for Plotly static export, Textual for
  TUI) must support that matrix or be marked optional.
- **Reproducibility for the reviewer.** Anything claimed in the README
  (screenshot, GIF, benchmark number) must be reproducible by running a
  documented command. No screenshots whose data nobody can regenerate.

---

## 7. Open Questions for SpecKit to Resolve

These are the decisions the user expects the specification phase to surface
explicitly rather than silently:

1. **TUI in or out?** Build now (as P2 user story), defer to a later phase,
   or drop entirely. Tradeoff: real differentiator vs additional test
   surface and dependency.
2. **Plotly export strategy.** Kaleido (heavyweight, all-platform) vs server-
   side rendering vs hand-curated screenshots only. Affects whether static
   PNG/SVG is automated.
3. **Demo-asset generation toolchain.** `vhs` (Tape-based, scriptable) vs
   `asciinema + agg` (manual record + convert) vs manual screencapture.
   Decision drives reproducibility claim in §6.
4. **Sample dataset for headline demos.** The current `sample_boxes.csv`
   produces 11% utilisation — undersells. Either curate a better headline
   dataset or generate one with documented parameters.
5. **Versioning at end of phase.** Bump to `v0.3.0-rc` and tag, or keep
   `v0.2.0.dev0` until US5 also lands. Affects release-notes work in this
   phase.
6. **README hero asset type.** Animated GIF (broad compat, larger file) vs
   embedded MP4 (smaller, GitHub renders both) vs static-PNG-with-link-to-
   demo. Determines the recording tool choice.
7. **Documentation deployment trigger.** Deploy on push to `main` only, or
   also on PR previews. Affects GitHub Pages configuration complexity.

---

## 8. Success Criteria

The phase is shippable when:

- A reviewer who has never seen the repo can, in one minute on the README
  alone, identify: what problem the project solves, what algorithms are
  available, how to install, how to run, and what a packed bin looks like.
- A reviewer who runs `bin-packer pack examples/<headline>.csv --visualize`
  sees a polished interactive visualisation, a Rich summary card, and a
  utilisation number that does not undersell the algorithm.
- The published documentation site is live, builds on every merge, and
  has at minimum: Quickstart, Algorithms (3 pages), API reference,
  Visualisation gallery.
- The README contains: hero asset, problem statement, install/run, badges
  (already there), algorithm comparison table, screenshot gallery, "highlights"
  section, "why I built this" paragraph, link to docs site.
- CI is green. Test count is ≥103 + N (where N is the count of new tests
  introduced by this phase's new code).
- A CV-ready paragraph (≤80 words) and a one-line tagline exist in a
  committed file.
- (If TUI shipped) `bin-packer tui` launches an interactive terminal app
  on a fresh install and lets a reviewer pack the sample dataset without
  reading documentation.
- The repository's GitHub `About` block is populated: tagline, topics,
  documentation link.
- A `v0.3.0-rc` git tag exists if §7.5 resolves in favour of tagging.

---

## 9. Branch and Workflow Hints

- Suggested branch name: `002-portfolio-polish` (matches existing spec
  naming convention `001-public-release-hardening`).
- The plan / tasks phase will follow this spec via `speckit.plan` and
  `speckit.tasks` as before.
- Work proceeds on the new branch in parallel with
  `feature/us5-extreme-point-benchmark`; integration via `develop` per the
  Git-Flow-lite strategy already in place.

---

**End of specify context.**
