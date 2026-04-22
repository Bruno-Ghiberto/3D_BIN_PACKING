# Constitution Context — `bin-packer-3d`

> Hand this entire document to `/speckit.constitution` as the free-text input.
> SpecKit will project this material onto the template at
> `.specify/memory/constitution.md` (currently unfilled — this is the first
> ratification) and emit a complete, versioned constitution plus a Sync Impact
> Report.
>
> Anything the command needs to fill a placeholder is in this document. It
> should not have to invent principles, ask for a project name, or guess dates.

---

## 0. Purpose of this document

The bin-packer-3d project is in its public-release hardening phase (see
`specs/001-public-release-hardening/spec.md`). The spec describes **what** will
ship. This constitution describes **the rules every spec, plan, task, and
implementation must respect** — the non-negotiable laws of the codebase.

Why now? Without a ratified constitution, `/speckit.plan` cannot run its
"Constitution Check" gate meaningfully, and there is no written record of the
quality bars the project claims to enforce. A constitution turns implicit
expectations into enforceable invariants.

The intended tone is **operational and testable**. Each principle must be:

- **Declarative** — states what is, not what we hope.
- **Testable** — has at least one mechanical or review-based enforcement.
- **Justified** — includes a rationale so future amendments can debate the
  reason, not re-litigate the conclusion.
- **Stable** — violated only through a deliberate amendment with version bump.

---

## 1. Project Identity

| Field | Value |
|---|---|
| `PROJECT_NAME` | `bin-packer-3d` |
| Human-readable name | 3D Bin Packing |
| Repository | `https://github.com/Bruno-Ghiberto/3D_BIN_PACKING` |
| Current version | 0.1.0 (Alpha) |
| License | MIT |
| Language | Python 3.11+ (public floor); maintainer develops on Python 3.14.3 |
| Maintainer | Bruno Ghiberto (solo) |
| Domain | Operations research — 3D Bin Packing Problem (NP-hard) |

Use `bin-packer-3d` as the `PROJECT_NAME` placeholder value — that is the
package identity on PyPI and the `bin-packer` CLI's binary name. "3D Bin
Packing" is an acceptable display alternative inside the constitution narrative.

---

## 2. Why a constitution, why now

Three reasons force a constitution at this exact moment:

1. **Public release is imminent.** The project is on a CV. The first
   impression must reflect engineering discipline, and discipline must be
   written down to be credible.
2. **The spec introduces structural invariants.** Spec-01 (US1) requires that
   the public API equal the runtime registry — that rule must exist
   somewhere durable, not only inside one spec.
3. **Amendments are coming.** The project will gain algorithms, constraints,
   and a release pipeline. The constitution defines how those amendments are
   themselves governed, so changes do not silently erode quality.

The constitution is **not** a style guide (see `pyproject.toml` + ruff/black
config for that) and **not** a spec (specs describe features; the constitution
describes immutable properties of the codebase). The distinction matters when
`/speckit.constitution` runs: it should write general principles and governance,
not feature-specific requirements.

---

## 3. Source material for derivation

`/speckit.constitution` should draw on the following when filling placeholders.
Every claim below is already documented elsewhere in the repo — the constitution
formalises it.

- `specs/001-public-release-hardening/spec.md` §Out of Scope, §Assumptions,
  §Dependencies — constraints and non-goals already accepted.
- `Speckit-context-prompts/spec-01-enhancing/01-specify.md` §5
  (Constraints & Non-Negotiables), §8 (Cross-Cutting Quality Bars), §10
  (Success Criteria — Project-Level Definition of Done).
- `pyproject.toml` — current build system (`hatchling`), strict mypy config,
  ruff/black line length 100, pytest markers, declared dependencies.
- `LICENSE` — MIT (authoritative; the constitution restates, does not redefine).
- `src/bin_packer_3d/` — current module layout; the constitution ratifies
  `models/` / `algorithms/` / `data/` / `visualization/` / `utils/` / `cli.py`
  / `config.py` as the accepted top-level separation.
- Project README + `.specify/memory/` — any pre-existing policy wording
  worth preserving verbatim (currently none; the file is a blank template).

---

## 4. Core Principles (proposed — ratify as written or override)

Seven principles. Each has: **Name** (Roman numeral + concise title with
NON-NEGOTIABLE tag where applicable), **Rule** (what MUST/MUST NOT hold),
**Rationale** (why), **Enforcement** (how a reviewer or CI verifies compliance).

Use exactly these seven unless the maintainer explicitly asks to drop, merge,
or reorder. The template's "less or more principles than the default" clause
accommodates seven cleanly.

### I. Contract Honesty (NON-NEGOTIABLE)

**Rule**
Every symbol the public API, CLI, configuration schema, or documentation
exposes MUST be backed by a working implementation verified at runtime. The
set of accepted configuration values, the set of CLI-discoverable algorithms,
and the set of documented strategies MUST be the same set — discoverable from
a single registry. No phantom entries. No silently-swallowed errors that mask
contract violations.

**Rationale**
A public API that lies (e.g., a `Literal` that includes `"bfd"` when no BFD
implementation exists) destroys user trust faster than a missing feature. A
registry-driven single source of truth prevents drift between declaration and
implementation. This is the most fundamental credibility defect the project
has today; the constitution ratifies the fix so it cannot regress.

**Enforcement**
- An automated test asserts the accepted `strategy` values equal the algorithm
  registry's keys on every CI run.
- CLI `info` output is generated from the registry — never hardcoded.
- Adding a new algorithm requires a single registration call; scattered wiring
  across config/CLI/tests/docs is a merge blocker.
- Error handling for loader row failures MUST produce both a log and a
  structured report — silent `continue` is forbidden.

### II. Test-First Discipline (NON-NEGOTIABLE)

**Rule**
Every new or materially-changed public symbol (class, function, CLI command,
configuration field) MUST ship with tests authored *before* the
implementation. Red → Green → Refactor applies to public contract changes.
Pure refactors (no behaviour change, no new public symbol) are exempt but
MUST retain a green suite at every commit boundary.

**Rationale**
Test-first is the cheapest mechanism to keep the contract and implementation
aligned (Principle I) and to avoid implementing the wrong thing. For a
single-maintainer open-source project, tests *are* the spec — a potential
contributor reads them to understand intent.

**Enforcement**
- Code review (self-review in the commit message for solo maintainer) notes
  "tests authored first" on every public-surface-touching commit.
- Property-based tests via Hypothesis cover algorithm invariants for every
  registered algorithm via parametrisation.
- Integration tests cover end-to-end CLI flows.
- `pyproject.toml` pytest markers separate unit / integration / slow; CI runs
  all three.

### III. Automated Quality Gates (NON-NEGOTIABLE)

**Rule**
Continuous Integration is the ground truth for "green." CI MUST run on every
push to `main` and every pull request, executing: lint (ruff), format check
(black), strict static type checking (mypy strict), unit tests, integration
tests, coverage measurement, and dependency vulnerability scanning. A pull
request that fails any gate — or drops below 90 % line coverage on
`src/bin_packer_3d/` — is blocked from merging.

**Rationale**
README claims ("all tests pass", "90 % coverage") are unverifiable without
automation. CI converts every claim into a live measurement that any reviewer
can inspect. Locally-passing checks are not sufficient — the environment must
be reproducible.

**Enforcement**
- `.github/workflows/` contains a pipeline that implements the above stages
  across Python 3.11, 3.12, 3.13, 3.14 on Linux at minimum.
- Branch protection on `main` requires all status checks to pass.
- A pre-commit configuration mirrors the CI lint/format/type-check stages so
  contributors fail fast locally.
- Coverage is uploaded to a public, free-tier service and reflected by a live
  badge in the README.

### IV. Reproducibility & Determinism

**Rule**
Any algorithm, test, or benchmark that uses randomness MUST accept a seed and
produce bit-identical results given `(seed, input, configuration)`. Benchmarks
MUST emit dated artefacts (JSON + Markdown) stored with the build. Results
that cannot be reproduced from a committed seed and input are treated as
anecdotes, not evidence.

**Rationale**
Benchmark claims on an NP-hard problem are only meaningful if reproducible.
Determinism is also the debugger's best friend: a failing test with a seed
reproduces on demand. This principle unblocks credible comparison with
published reference instances (Bischoff & Ratcliff, Loh & Nee, etc.).

**Enforcement**
- CLI `bin-packer benchmark` accepts `--seed`; the same seed yields the same
  result.
- CI benchmark runs produce a JSON artefact attached to the build.
- Property-based tests use a deterministic Hypothesis profile in CI (seed
  fixed per run, printed on failure).
- Algorithm implementations that introduce randomness MUST document the seed
  surface and expose it through configuration.

### V. Library Citizenship

**Rule**
Library code MUST be a polite guest in its consumer's process:

- No `print()` in `src/bin_packer_3d/` except inside the CLI module's
  Rich-backed user-facing output.
- No `logging.basicConfig()`, no root-logger configuration, no global
  side-effects on import. Module-level loggers with `NullHandler` attached.
- No `os.chdir()`, no mutation of `sys.path`, no environment-variable writes
  at import time.
- No network I/O on import. No blocking I/O on module load.
- Errors are raised through typed exceptions, never via `sys.exit()` from
  inside library code; CLI handlers translate exceptions to exit codes at the
  boundary.

**Rationale**
A library that silently reconfigures its consumer's logging, working
directory, or exit behaviour cannot be embedded in larger applications. Users
of `bin-packer-3d` include people writing their own Python tools on top of
it; the library must behave well for them.

**Enforcement**
- A pre-commit / CI grep check forbids `print(` outside the CLI module.
- A test asserts no root-logger handlers are installed on `import
  bin_packer_3d`.
- Code review flags any global state capture.

### VI. Documentation as Artefact (NON-NEGOTIABLE)

**Rule**
Documentation is a shipped deliverable, not an afterthought. Every public
class and function MUST carry a docstring. A documentation site MUST build in
CI and publish on every push to `main`. Non-trivial architectural decisions
MUST be recorded as Architecture Decision Records under `docs/adr/`. The
README MUST pass the 30-second credibility test: a first-time visitor sees a
hero visualisation, a two-sentence problem statement, headline benchmark
numbers, an install command, and live quality badges — within 30 seconds of
landing.

**Rationale**
For a CV-mentioned project, documentation *is* the product a recruiter or
reviewer reads. Tests prove the code works; docs prove the author cares.
ADRs specifically protect the project from revisiting the same decision every
time a new contributor arrives.

**Enforcement**
- Missing docstrings on public symbols fail CI (via ruff `D` ruleset or
  pydocstyle).
- The documentation site is linked from the README and its build status is a
  CI gate.
- Every algorithm and constraint has a reference page that is regenerated
  from source.
- ADRs follow a template (context → decision → consequences) and live at
  `docs/adr/NNN-<slug>.md`.
- README rendering is part of the release checklist.

### VII. Privacy by Default

**Rule**
No identifiable real-world business, personal, or customer data may exist in
the repository working tree OR git history. Datasets bundled with the project
are either published reference instances (with license carried forward) or
synthetic / fully-anonymised samples. Any such data that once existed MUST
be scrubbed from history using `git filter-repo` (or equivalent) before the
release that surfaces it.

**Rationale**
The project's origin story includes real-world business artefacts (Spanish
`.xlsx` files with packing lists). Leaking those through a public push would
cause harm to a third party and destroy the project's credibility. This is a
security concern, not a hygiene concern — treat accordingly.

**Enforcement**
- `DATASETS/README.md` enumerates every file with origin and license.
- A pre-release audit step scans the working tree AND the git history for
  files matching common business-artefact patterns (packing lists, personal
  names, internal project codes).
- Any violation blocks the release; history rewrite is the remediation.

### VIII. Performance Discipline

**Rule**
Optimizations MUST be justified by measurement, not intuition. The optimization
ladder — in mandatory order of consideration — is:

1. **Better algorithm** (change the complexity class).
2. **Vectorization** via NumPy (rewrite the hot path in array operations).
3. **JIT compilation** via Numba or Cython (decorator or typed extension).
4. **Native extension** via Rust (preferred; PyO3 + maturin) or C.

Skipping rungs requires an ADR (see Enforcement). "Because Rust is faster" is
never a sufficient reason on its own to skip rungs 2 or 3.

**Rationale**
For a library maintained by a single person, the cost of a native extension
— new build backend (`hatchling` → `maturin`), ~20 wheels per release
(platforms × Python versions), `cibuildwheel` in CI, two languages to
maintain fluency in — is disproportionately high. A measured 30× speedup on
a genuine hot path justifies that cost; a hypothetical one does not.
Performance claims that cannot be backed by flamegraphs and benchmark
numbers are folklore, not engineering. This principle also protects the
project's contract of "install in one command" from silently degrading into
"install, wait for cargo, pray the toolchain is right."

**Enforcement**
- Introducing a native extension (Rust, C, C++, Cython compiled units, or
  any component whose distribution requires platform-specific wheels)
  requires an Architecture Decision Record at `docs/adr/` that documents:
  - (a) The profile of the hot path, with a flamegraph (`py-spy`,
    `scalene`, or equivalent) committed under `docs/benchmarks/profiles/`.
  - (b) The attempts and measured results for NumPy vectorization and
    Numba/Cython on the same hot path.
  - (c) The measured speedup of the native extension vs. those
    alternatives on a committed benchmark instance.
  - (d) The distribution plan: wheel matrix, fallback behaviour when no
    wheel is available for the user's platform, impact on
    `requires-python`, impact on the install experience.
- PRs adding Rust/C source or changing the build backend to `maturin` /
  `setuptools-rust` without a corresponding accepted ADR are rejected.
- Benchmarks introduced to justify an optimization MUST be reproducible
  (Principle IV applies) and become part of the CI benchmark suite so that
  future regressions are caught.
- Premature optimization — an optimization shipped without a profile proving
  it targets a real hot path — is a merge blocker.

---

## 5. Additional Constraints Section — Technology Baseline & Non-Negotiables

This section fills the `[SECTION_2_NAME]` / `[SECTION_2_CONTENT]` template
placeholders. Title it exactly **"Technology Baseline & Non-Negotiables"**.

Ratified baseline (all changes to these require an amendment):

- **Language & runtime**: Python 3.11+ as the public floor (requires an
  update to `requires-python` from `>=3.10` to `>=3.11` during
  implementation). Supported versions in CI matrix: 3.11, 3.12, 3.13, 3.14.
  The maintainer develops on Python 3.14.3; developing against the latest
  stable is encouraged so new-idiom adoption (PEP 695 type syntax, better
  exception messages, `ExceptionGroup`, `TaskGroup`) is ratified as it
  becomes widely available. Dropping a supported version is a MAJOR
  amendment. Adding a newly-released stable version is MINOR.
- **License**: MIT (SPDX: `MIT`). License header policy: source files do NOT
  need per-file headers; `LICENSE` at the root governs the whole tree.
- **Build system**: `hatchling` via `pyproject.toml`. `setup.py` is
  forbidden. Switching build backend requires a MAJOR amendment.
- **Packaging**: single distribution (`bin-packer-3d`) with `bin-packer`
  console script and `py.typed` marker shipped in the wheel (PEP 561).
- **Type safety**: `mypy --strict` on `src/bin_packer_3d/` is a gate. New
  public symbols carry type hints. `# type: ignore` requires an adjacent
  comment explaining why.
- **Style**: `ruff` + `black`, line length 100. No local style exceptions.
- **Logging**: standard library `logging` only; `loguru`, `structlog`, or
  custom logging frameworks are forbidden in the library module. The CLI may
  layer Rich on top for presentation.
- **Dependency policy**: a runtime dependency is added only if it appears in
  at least two distinct modules, removes ≥ 50 lines of hand-written code, or
  is a peer of an already-present dependency. Every addition includes a
  one-line rationale in the commit message or an ADR for non-obvious cases.
  Dependabot (or equivalent) scans for updates weekly.
- **Vendor lock-in prohibition**: no paid CI, no closed-source observability
  or coverage provider, no SaaS-only tool in the critical path. Free-tier
  GitHub / PyPI / Codecov-class / HF Spaces / Streamlit Community Cloud are
  acceptable.
- **Internationalisation**: English only for CLI output, documentation,
  comments, commit messages, and identifiers. Existing mixed-language
  artefacts (Spanish `.xlsx` names, early commit messages) are tolerated in
  history but not re-introduced.
- **Coordinate convention**: `X = length`, `Y = width`, `Z = height`. This is
  non-obvious and easy to misuse; it is ratified so that future algorithms
  cannot redefine it without a MAJOR amendment and a migration plan.
- **Public API stability**: the project is pre-1.0. Breaking changes to any
  symbol exported from `bin_packer_3d` are permitted, but MUST be documented
  in `CHANGELOG.md` under the `Unreleased` section with a migration note.
- **Coverage floor**: 90 % line coverage on `src/bin_packer_3d/`. Dropping
  below blocks the PR. Raising the floor is encouraged; lowering it requires
  an amendment.

---

## 6. Development Workflow & Quality Gates Section

This section fills the `[SECTION_3_NAME]` / `[SECTION_3_CONTENT]` template
placeholders. Title it exactly **"Development Workflow & Quality Gates"**.

### 6.1 Branching and commit conventions

- Default branch: `main`. Always shippable — no in-progress work merged.
- Feature work lives on SpecKit-named branches (`NNN-<short-name>`,
  sequential). Created by the `speckit.git.feature` hook.
- Commit style: **Conventional Commits** — `type(scope): subject`. Types:
  `feat` | `fix` | `docs` | `refactor` | `test` | `chore` | `perf` | `build`
  | `ci` | `revert`. `scope` is a top-level module (`algorithms`, `config`,
  `cli`, `docs`, `ci`, etc.).
- Commit messages MUST NOT contain AI-attribution lines (no
  `Co-Authored-By: Claude …`, no `🤖 Generated` trailers).
- Breaking changes use `!` (`feat(config)!: …`) AND a `BREAKING CHANGE:`
  footer explaining the migration path.

### 6.2 Pull-request workflow (applies even for solo maintainer)

- Every non-trivial change ships as a PR even if self-merged. PRs are the
  durable audit trail, not the approval gate.
- A PR MUST:
  1. Reference the spec (or issue) it implements.
  2. Pass every CI gate (Principle III).
  3. Update `CHANGELOG.md` under `Unreleased` for any user-visible change.
  4. Update or add tests (Principle II).
  5. Update docstrings / docs for any changed public symbol (Principle VI).
  6. Cite which constitutional principles it touches in a
     "Constitution Impact" line, and note if any principle is tested against
     a new boundary case.
- Merge strategy: squash-merge for features, rebase for hotfixes. No merge
  commits on `main`.

### 6.3 Release workflow

- Versioning: **SemVer** with this project's specific meaning:
  - **MAJOR**: breaking change to any exported public API in `bin_packer_3d`;
    dropping a supported Python version; amending this constitution in a
    backward-incompatible way; changing the coordinate convention.
  - **MINOR**: adding a new algorithm, constraint, output format, CLI
    command, or materially expanded guidance in the constitution.
  - **PATCH**: bug fix, clarification, wording/typo fix, internal refactor
    with no observable change.
- Release trigger: Git tag `v<MAJOR>.<MINOR>.<PATCH>` pushed to `main`.
- Release artefacts: wheel + sdist, published to PyPI via OIDC trusted
  publisher (no long-lived API tokens in repo secrets).
- The Git tag, `CHANGELOG.md` entry, constitution version (if amended), and
  PyPI artefact version MUST match exactly. Discrepancies block the release
  until reconciled.
- Docker image (multi-stage, non-root runtime) published to a public free-tier
  container registry alongside every release.

### 6.4 Benchmark & reproducibility workflow

- `bin-packer benchmark` runs in CI on every push to `main` and on every PR
  that touches `src/bin_packer_3d/algorithms/`. A JSON artefact is attached
  to the build.
- The `docs/benchmarks/` page is regenerated from the CI artefact and
  committed alongside each release.
- A regression beyond 25 % utilisation drop or 25 % runtime increase on a
  fixed reference instance surfaces a CI warning that must be acknowledged
  in the PR description.

### 6.5 Security & privacy workflow

- Dependabot (or equivalent) opens PRs for dependency updates; minor/patch
  updates are auto-mergeable after CI.
- `SECURITY.md` at the root documents the disclosure channel.
- Before every release, a pre-release audit script scans the working tree
  AND git history for patterns matching business artefacts
  (packing-list-like `.xlsx` names, identifiable personal names, internal
  project codes). A hit blocks the release.
- `.env` files, credentials, and tokens are never committed. `.env.example`
  lists required variables without values.

### 6.6 Documentation workflow

- Docs site builds in CI; a failing build blocks the release.
- Every algorithm ships with a reference page (complexity, pseudo-code,
  citation, when-to-use).
- ADRs for architectural decisions that cross module boundaries or change a
  principle interpretation. Status field: `Proposed` | `Accepted` |
  `Superseded`.
- The README is reviewed at every MAJOR / MINOR release for the 30-second
  credibility test.

---

## 7. Governance

This section fills the `[GOVERNANCE_RULES]` placeholder.

### 7.1 Supremacy

- The constitution supersedes every other project practice, README, or ad-hoc
  convention. A disagreement between this document and any other artefact is
  resolved in favour of this document until the constitution is amended.
- Every spec's `/speckit.plan` output MUST include a "Constitution Check"
  gate that explicitly lists which principles the plan touches and how they
  are satisfied. A plan that cannot satisfy a principle MUST propose an
  amendment before proceeding.

### 7.2 Amendment procedure

Amendments are treated as first-class changes with the same rigor as code.

1. Draft the amendment as a PR modifying `.specify/memory/constitution.md`.
2. Include in the PR description:
   - The version bump (MAJOR / MINOR / PATCH) and rationale.
   - The principles or sections added, removed, or modified.
   - The Sync Impact Report covering dependent templates and guidance docs.
   - A migration note if any existing spec, plan, or code must be updated.
3. `/speckit.constitution` produces the Sync Impact Report automatically;
   include it as an HTML comment at the top of the file.
4. Self-review (solo maintainer) using the same PR checklist as features.
5. Merge with a commit message of the form:
   `docs(constitution): amend to v<NEW_VERSION> — <short description>`.
6. Tag the repo with `constitution-v<NEW_VERSION>` to make the amendment
   independently referenceable.

Amendment authorities:

- **Maintainer** (currently Bruno Ghiberto): may propose, review, and merge
  amendments.
- **Co-maintainers** (future): once added, the first action is to amend this
  procedure to require a second reviewer for MAJOR amendments. Adding a
  co-maintainer is itself a MINOR amendment.
- **Contributors**: may propose amendments via PR; merge remains with the
  maintainer until the procedure is updated.

### 7.3 Versioning policy for the constitution

Independent of the package version. Use SemVer:

- **MAJOR**: backward-incompatible governance or principle removal /
  redefinition (e.g. dropping "Privacy by Default", lowering coverage floor
  below 90 %, removing the test-first requirement).
- **MINOR**: adding a new principle, adding a new section, materially
  expanding guidance within an existing principle.
- **PATCH**: wording clarifications, typo fixes, non-semantic refinements.

Initial ratification: **v1.0.0**.

### 7.4 Compliance review

- Every PR's "Constitution Impact" line is checked for plausibility during
  self-review.
- Every release performs a full constitution compliance audit: walk every
  principle, produce a pass/fail line, and block the release on any fail
  that is not accompanied by an amendment.
- Specs reference the constitution by version (`This spec was drafted against
  constitution v1.0.0.`). A constitution amendment that conflicts with a
  pending spec forces the spec to be re-reviewed.

### 7.5 Runtime guidance file

The agent-specific guidance file for the project is `CLAUDE.md` at the
repository root. `CLAUDE.md` explains how AI agents (Claude Code in
particular) should interact with this repository; it defers to this
constitution for principles and only adds agent-specific operational notes.
Any conflict between `CLAUDE.md` and this constitution is resolved in favour
of the constitution.

---

## 8. Version Metadata — values for the template

Fill the template footer line exactly as follows:

| Placeholder | Value |
|---|---|
| `CONSTITUTION_VERSION` | `1.0.0` |
| `RATIFICATION_DATE` | `2026-04-22` |
| `LAST_AMENDED_DATE` | `2026-04-22` |

Use ISO-8601 `YYYY-MM-DD` exactly; no natural-language dates.

---

## 9. Propagation targets — Sync Impact Report scope

When `/speckit.constitution` produces its Sync Impact Report, verify and note
the status of these dependent artefacts:

| Artefact | Expected state after v1.0.0 ratification |
|---|---|
| `.specify/templates/plan-template.md` | Must contain a "Constitution Check" section that instantiates the seven principles. ⚠ Likely needs update. |
| `.specify/templates/spec-template.md` | Spec need not cite principles by default but MUST accommodate the "Constitution Impact" rubric in its Acceptance Scenarios or a new optional section. ⚠ Likely needs update. |
| `.specify/templates/tasks-template.md` | Task types must include "Documentation", "Test-first scaffolding", "Benchmark reproducibility", and "Privacy audit" to reflect the principles. ⚠ Likely needs update. |
| `.specify/templates/commands/*.md` | No agent-specific names; generic guidance. ✅ Verify. |
| `CLAUDE.md` (root) | Add a pointer to this constitution as the authoritative rules source. ⚠ Update. |
| `README.md` | Link to `.specify/memory/constitution.md` from the "Project Policy" or "Contributing" section. ⚠ Update once US3 rewrites the README. |
| `docs/adr/` | Create if absent; will accrue ADRs per Principle VI. ⚠ Create in spec-01. |

`/speckit.constitution` should list each artefact with ✅ / ⚠ in the Sync
Impact Report and flag any follow-up TODOs.

---

## 10. What the generated constitution MUST look like — output contract

The file emitted at `.specify/memory/constitution.md` by `/speckit.constitution`
MUST satisfy all of the following:

1. **No remaining bracket placeholders** (`[ALL_CAPS_IDENTIFIER]`). A TODO
   marker is acceptable only if a value is genuinely unknown; this document
   supplies all values, so no TODO is expected.
2. **Heading hierarchy preserved** from `.specify/templates/constitution-template.md`:
   - `# bin-packer-3d Constitution` (H1)
   - `## Core Principles` (H2)
     - `### I. Contract Honesty (NON-NEGOTIABLE)` (H3) and seven more H3
       subsections through `### VIII. Performance Discipline`
   - `## Technology Baseline & Non-Negotiables` (H2)
   - `## Development Workflow & Quality Gates` (H2)
   - `## Governance` (H2)
3. **Version footer line** present and formatted exactly:
   `**Version**: 1.0.0 | **Ratified**: 2026-04-22 | **Last Amended**: 2026-04-22`.
4. **Sync Impact Report** prepended as an HTML comment (`<!-- ... -->`) at
   the very top of the file, above the H1. Content: version change
   (`<none> → 1.0.0`), principles added (all seven), sections added (all
   three), templates to update (list with ✅ / ⚠), deferred TODOs (none
   expected).
5. **Declarative, testable language** throughout. `MUST` / `MUST NOT` /
   `SHOULD` used deliberately. No vague "would be nice", no
   aspirational-only language.
6. **Rationale paragraphs** accompany each principle. A principle with no
   rationale is harder to defend in an amendment discussion.
7. **Runtime guidance pointer** in the Governance section names `CLAUDE.md`
   as the agent guidance file.
8. **ISO dates** `YYYY-MM-DD`.
9. **ASCII / UTF-8 only**. No smart quotes introduced by word-processor
   autoformatting. No emoji decoration.
10. **≤ 500 lines total** (current draft fits comfortably). A constitution
    that doesn't fit on a single scrollable view has already lost the reader.

---

## 11. Edge cases for `/speckit.constitution` to resolve

- **Existing constitution file is the blank template.** Treat as first
  ratification: `RATIFICATION_DATE` = `LAST_AMENDED_DATE` = today.
- **Eight principles vs. five-slot template.** Respect the user's count —
  add three more H3 subsections after the template's fifth. The skill's
  "less or more principles than the template" clause explicitly permits this.
- **Performance Discipline (Principle VIII) and the Rust question.** The
  maintainer has both Python 3.14.3 and Rust 1.93.1 installed locally. The
  decision ratified here is: Rust is NOT in scope for the v1.0.0 release
  because no bottleneck has been measured. The project implements its
  algorithms in Python first, profiles, applies the cheapest adequate
  remedy (NumPy / Numba / Cython) before escalating to Rust. A future
  spec-phase may introduce Rust via PyO3/maturin, but ONLY after the
  ADR-backed evidence trail specified by Principle VIII. This keeps the
  door open without encoding speculation in v1.0.0.
- **Conflicting wording between this context and an existing repo artefact.**
  Prefer this context, but flag the conflict in the Sync Impact Report.
- **Tooling reality today.** Some gates (e.g., CodeQL, docs site) do not yet
  exist at the moment of ratification — they are introduced by spec-01.
  The constitution ratifies the *rule*, not the current tooling state.
  That is acceptable and intended: the constitution is aspirational about
  tooling only insofar as spec-01 is scheduled to deliver it.
- **Coordinate convention as principle vs. constraint.** It lives in
  §5 (Technology Baseline) rather than in the principles, because it is a
  model decision, not a behavioural rule. Do not promote it into Core
  Principles without maintainer direction.
- **Compliance of spec-01 itself.** Spec-01 was drafted before the
  constitution existed. Retrofit compliance: the spec's acceptance criteria
  already satisfy Principles I, III, V, VI, VII. Principles II and IV are
  imposed going forward; the spec's test strategy and benchmark reproducibility
  lines already conform.
- **Amendment velocity.** Early in the project (solo maintainer, pre-1.0),
  amendments may be frequent. The versioning policy accommodates this — PATCH
  amendments for wording fixes are cheap. Don't hesitate to amend if a
  principle proves unworkable.

---

## 12. Narrative the final constitution must convey

A second-level summary the constitution SHOULD deliver in its opening
paragraphs (can be inlined into the Core Principles intro or the first
principle's rationale):

> *"`bin-packer-3d` is a public open-source Python library for the 3D Bin
> Packing Problem. Its constitution is organised around one conviction: the
> code, the documentation, and the tests must tell the same story. Every
> principle below — contract honesty, test-first discipline, automated
> quality gates, reproducibility, library citizenship, documentation as
> artefact, privacy by default, and performance discipline — is an
> enforcement mechanism for that conviction. Amendments follow the same
> rigor as code changes; the
> constitution evolves only through explicit, versioned decisions."*

If the generated constitution does not convey that conviction in substance,
the context prompt has failed and should be revised.

---

## 13. Inputs available to `/speckit.constitution`

- This document (`Speckit-context-prompts/Context-Constitution.md`) — the
  primary source.
- `.specify/templates/constitution-template.md` — the shape to fill.
- `.specify/memory/constitution.md` — the destination (currently unfilled).
- `specs/001-public-release-hardening/spec.md` — active spec whose
  acceptance criteria must remain compatible with the ratified principles.
- `Speckit-context-prompts/spec-01-enhancing/01-specify.md` — the original
  context prompt that framed spec-01.
- `pyproject.toml`, `LICENSE`, `README.md`, `CLAUDE.md` — existing
  authoritative artefacts for version, license, identity, and agent guidance.

---

*End of context. Hand this entire document to `/speckit.constitution`.*
