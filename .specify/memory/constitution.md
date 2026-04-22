<!--
Sync Impact Report — Constitution v1.0.0 (Initial Ratification)
===============================================================

Version change: <none> → 1.0.0 (initial ratification)

Principles added (all 8):
  I.    Contract Honesty (NON-NEGOTIABLE)
  II.   Test-First Discipline (NON-NEGOTIABLE)
  III.  Automated Quality Gates (NON-NEGOTIABLE)
  IV.   Reproducibility & Determinism
  V.    Library Citizenship
  VI.   Documentation as Artefact (NON-NEGOTIABLE)
  VII.  Privacy by Default
  VIII. Performance Discipline

Sections added:
  - Technology Baseline & Non-Negotiables
  - Development Workflow & Quality Gates
  - Governance (with Supremacy, Amendment procedure, Versioning policy,
    Compliance review, Runtime guidance file subsections)

Principles removed / modified: none (first ratification)

Templates / dependent artefacts:
  ⚠  .specify/templates/plan-template.md
       "Constitution Check" section is a placeholder. Update to instantiate
       the 8 principles as explicit pass/fail gates during the next
       /speckit.plan run (spec-01's plan phase will exercise and finalise
       this).
  ✅ .specify/templates/spec-template.md
       No direct constitution reference; existing structure compatible.
       specs/001-public-release-hardening/spec.md already aligns.
  ✅ .specify/templates/tasks-template.md
       No direct constitution reference; existing phase structure
       (Setup → Foundational → User Stories → Polish) compatible.
       Principle-driven task types (test-first scaffolding, privacy audit,
       benchmark reproducibility, ADR creation) are to be injected by
       /speckit.tasks per-spec, not hard-coded in the template.
  N/A .specify/templates/commands/
       Directory does not exist in spec-kit 0.7.4; skills under
       .claude/skills/speckit-*/ have replaced templated commands.
  ⚠  CLAUDE.md (repository root)
       Currently untracked. Should include a pointer to this constitution
       as the authoritative rules source when first committed.
  ⚠  README.md
       Link from a "Project Policy" or "Contributing" section to
       .specify/memory/constitution.md. Deferred until US3 (Public-Facing
       Documentation) rewrites the README under spec-01.
  ⚠  docs/adr/
       Does not yet exist. Will be created during spec-01 implementation
       per Principle VI.

Deferred items / TODOs: none

Consistent with:
  - specs/001-public-release-hardening/spec.md (active spec)
  - Speckit-context-prompts/spec-01-enhancing/01-specify.md
  - Speckit-context-prompts/Context-Constitution.md (source context prompt)

Suggested commit message:
  docs(constitution): ratify v1.0.0 with 8 principles and governance
-->

# bin-packer-3d Constitution

`bin-packer-3d` is a public open-source Python library for the 3D Bin
Packing Problem (NP-hard). Its constitution is organised around one
conviction: the code, the documentation, and the tests must tell the same
story. Every principle below — contract honesty, test-first discipline,
automated quality gates, reproducibility, library citizenship, documentation
as artefact, privacy by default, and performance discipline — is an
enforcement mechanism for that conviction. Amendments follow the same rigor
as code changes; the constitution evolves only through explicit, versioned
decisions.

## Core Principles

### I. Contract Honesty (NON-NEGOTIABLE)

Every symbol the public API, CLI, configuration schema, or documentation
exposes MUST be backed by a working implementation verified at runtime. The
set of accepted configuration strategy values, the set of CLI-discoverable
algorithms, and the set of documented strategies MUST be the same set —
discoverable from a single registry. No phantom entries. No silently
swallowed errors that mask contract violations.

**Rationale**: A public API that declares more than it implements destroys
user trust faster than a missing feature. A registry-driven single source
of truth prevents drift between declaration and implementation.

**Enforcement**: An automated test asserts the accepted strategy values on
`PackerConfig` equal the algorithm registry's keys. CLI `info` output is
generated from the registry — never hardcoded. Adding a new algorithm is a
single registration call; scattered wiring across config, CLI, tests, and
docs is a merge blocker. Loader row-level failures produce both a WARNING
log AND a structured report — silent `continue` is forbidden.

### II. Test-First Discipline (NON-NEGOTIABLE)

Every new or materially-changed public symbol (class, function, CLI
command, configuration field) MUST ship with tests authored before the
implementation. Red → Green → Refactor applies to public-contract changes.
Pure refactors (no behavioural change, no new public symbol) are exempt
but MUST retain a green suite at every commit boundary.

**Rationale**: Test-first is the cheapest mechanism to keep the contract
and implementation aligned (Principle I) and to avoid implementing the
wrong thing. For a single-maintainer open-source project, tests are the
living specification a contributor reads to understand intent.

**Enforcement**: Commit messages on public-surface-touching commits note
"tests authored first". Property-based tests via Hypothesis cover algorithm
invariants for every registered algorithm via parametrisation. Integration
tests cover end-to-end CLI flows. The pytest marker matrix
(unit / integration / slow) is enforced in CI.

### III. Automated Quality Gates (NON-NEGOTIABLE)

Continuous Integration is the ground truth for "green". CI MUST run on
every push to `main` and every pull request, executing: lint (ruff),
format check (black), strict static type checking (`mypy --strict`), unit
tests, integration tests, coverage measurement, and dependency
vulnerability scanning. A pull request that fails any gate — or drops
below 90 % line coverage on `src/bin_packer_3d/` — is blocked from merging.

**Rationale**: Locally-passing claims are unverifiable. CI converts every
quality claim into a live measurement any reviewer can inspect.
Reproducible environments prevent the "works on my machine" pattern.

**Enforcement**: `.github/workflows/` contains the pipeline across Python
3.11, 3.12, 3.13, and 3.14 on Linux at minimum. Branch protection on `main`
requires all status checks to pass. A pre-commit configuration mirrors the
CI lint, format, and type-check stages so contributors fail fast locally.
Coverage is uploaded to a publicly visible free-tier service and reflected
by a live README badge.

### IV. Reproducibility & Determinism

Any algorithm, test, or benchmark that uses randomness MUST accept a seed
and produce bit-identical results given `(seed, input, configuration)`.
Benchmarks MUST emit dated artefacts (JSON + Markdown) stored with the
build. Results that cannot be reproduced from a committed seed and input
are anecdotes, not evidence.

**Rationale**: Performance and utilisation claims on an NP-hard problem
are only meaningful if reproducible. Determinism also makes debugging
tractable: a failing test with a seed reproduces on demand.

**Enforcement**: `bin-packer benchmark` accepts `--seed`; the same seed
yields the same result. CI benchmark runs attach a JSON artefact to the
build. Property-based tests use a deterministic Hypothesis profile in CI
(fixed seed per run, printed on failure). Algorithms with randomness
document the seed surface and expose it through configuration.

### V. Library Citizenship

Library code MUST be a polite guest in its consumer's process:

- No `print()` calls in `src/bin_packer_3d/` except inside the CLI
  module's Rich-backed user-facing output.
- No `logging.basicConfig()`, no root-logger configuration, no global
  side-effects on import. Module-level loggers with `NullHandler`
  attached.
- No `os.chdir()`, no mutation of `sys.path`, no environment-variable
  writes at import time.
- No network I/O and no blocking I/O on module import.
- Errors are raised through typed exceptions; `sys.exit()` is never called
  from library code. The CLI translates exceptions to exit codes at the
  boundary.

**Rationale**: A library that silently reconfigures its consumer's
logging, working directory, or exit behaviour cannot be embedded in
larger applications. Users of `bin-packer-3d` build their own Python
tools on top; it must behave well for them.

**Enforcement**: A pre-commit / CI grep check forbids `print(` outside
the CLI module. An automated test asserts that no root-logger handlers
are installed after `import bin_packer_3d`. Code review flags any global
state capture.

### VI. Documentation as Artefact (NON-NEGOTIABLE)

Documentation is a shipped deliverable, not an afterthought. Every public
class and function MUST carry a docstring. A documentation site MUST
build in CI and publish on every push to `main`. Non-trivial
architectural decisions MUST be recorded as Architecture Decision Records
(ADRs) at `docs/adr/`. The README MUST pass the 30-second credibility
test: a first-time visitor sees a hero visualisation, a two-sentence
problem statement, headline benchmark numbers, an install command, and
live quality badges within 30 seconds.

**Rationale**: For a CV-mentioned project, documentation is the product a
reviewer reads. Tests prove the code works; docs prove the author cares.
ADRs specifically protect the project from re-litigating the same
decision every time a new contributor arrives.

**Enforcement**: Missing docstrings on public symbols fail CI (via ruff's
`D` ruleset or pydocstyle). The documentation site build is a CI gate;
a failing build blocks the release. Every algorithm and constraint has a
reference page regenerated from source. ADRs follow
`docs/adr/NNN-<slug>.md` with a Status field (Proposed / Accepted /
Superseded). The README is reviewed at every MAJOR or MINOR release.

### VII. Privacy by Default

No identifiable real-world business, personal, or customer data MAY exist
in the repository's working tree OR git history. Datasets bundled with
the project are either published reference instances (with their license
carried forward) or synthetic / fully-anonymised samples. Any such data
that once existed MUST be scrubbed from history using `git filter-repo`
(or equivalent) before the release that surfaces it.

**Rationale**: The project's origin includes real-world business
artefacts (Spanish `.xlsx` files with packing lists). A public push of
those would harm a third party and destroy the project's credibility.
This is a security concern, not a hygiene concern — treat accordingly.

**Enforcement**: `DATASETS/README.md` enumerates every file with origin
and license. A pre-release audit script scans the working tree AND git
history for files matching business-artefact patterns (packing-list-like
`.xlsx` names, identifiable personal names, internal project codes). Any
violation blocks the release; history rewrite is the remediation.

### VIII. Performance Discipline

Optimizations MUST be justified by measurement, not intuition. The
optimization ladder — in mandatory order of consideration — is:

1. **Better algorithm** (change the complexity class).
2. **Vectorization** via NumPy (rewrite the hot path in array
   operations).
3. **JIT compilation** via Numba or Cython.
4. **Native extension** via Rust (preferred; PyO3 + maturin) or C.

Skipping rungs requires an ADR (see Enforcement). "Because Rust is
faster" is never a sufficient reason on its own to skip rungs 2 or 3.

**Rationale**: For a library maintained by a single person, the cost of
a native extension — new build backend (`hatchling` → `maturin`), ~20
wheels per release (platforms × Python versions), `cibuildwheel` in CI,
dual-language fluency — is disproportionately high. A measured 30×
speedup on a genuine hot path justifies that cost; a hypothetical one
does not. Performance claims unbacked by flamegraphs and benchmark
numbers are folklore, not engineering. This principle also protects the
project's "install in one command" contract from silently degrading into
"install, wait for cargo, pray the toolchain is right."

**Enforcement**: Introducing a native extension (Rust, C, C++, Cython-
compiled units, or any component whose distribution requires
platform-specific wheels) requires an ADR at `docs/adr/` documenting:

- (a) The profile of the hot path, with a flamegraph (`py-spy`,
  `scalene`, or equivalent) committed under `docs/benchmarks/profiles/`.
- (b) The attempted and measured results for NumPy vectorization and
  Numba / Cython JIT on the same hot path.
- (c) The measured speedup of the native extension vs. those
  alternatives on a committed benchmark instance.
- (d) The distribution plan: wheel matrix, fallback behaviour when no
  wheel is available for the user's platform, impact on
  `requires-python`, and impact on the install experience.

PRs adding Rust/C source or switching the build backend to `maturin` or
`setuptools-rust` without a corresponding accepted ADR are rejected.
Benchmarks introduced to justify an optimization MUST be reproducible
(Principle IV applies) and become part of the CI benchmark suite so
future regressions are caught. Premature optimization — an optimization
shipped without a profile proving it targets a real hot path — is a
merge blocker.

## Technology Baseline & Non-Negotiables

- **Language & runtime**: Python 3.11+ is the public floor
  (`requires-python = ">=3.11"`). Supported versions in the CI matrix:
  3.11, 3.12, 3.13, 3.14. The maintainer develops on Python 3.14.3 so
  new-idiom adoption (PEP 695 type parameter syntax, improved exception
  messages, `ExceptionGroup`, `TaskGroup`) is ratified as it becomes
  widely available. Dropping a supported version is a MAJOR amendment;
  adding a newly-released stable version is MINOR.
- **License**: MIT (SPDX: `MIT`). Per-file license headers are NOT
  required; the root `LICENSE` governs the whole tree.
- **Build system**: `hatchling` via `pyproject.toml`. `setup.py` is
  forbidden. Switching build backend (e.g. to `maturin` for a Rust
  extension) requires a MAJOR amendment AND satisfies Principle VIII.
- **Packaging**: a single distribution (`bin-packer-3d`) with the
  `bin-packer` console script and a `py.typed` marker per PEP 561.
- **Type safety**: `mypy --strict` on `src/bin_packer_3d/` is a CI gate.
  New public symbols carry type hints. `# type: ignore` requires an
  adjacent comment explaining why.
- **Style**: `ruff` + `black`, line length 100. No local style
  exceptions.
- **Logging**: standard library `logging` only inside the library module;
  `loguru`, `structlog`, or custom frameworks are forbidden. The CLI may
  layer Rich on top for presentation.
- **Dependency policy**: a runtime dependency is added only if it appears
  in at least two distinct modules, removes ≥ 50 lines of hand-written
  code, or is a peer of an already-present dependency. Every addition
  includes a one-line rationale in the commit message or an ADR for
  non-obvious cases. Dependabot scans for updates weekly.
- **Vendor lock-in prohibition**: no paid CI, no closed-source
  observability or coverage provider, no SaaS-only tool in the critical
  path. Free-tier GitHub, PyPI, Codecov-class, HF Spaces, and Streamlit
  Community Cloud are acceptable.
- **Internationalisation**: English only for CLI output, documentation,
  comments, commit messages, and identifiers. Mixed-language artefacts
  in existing history are tolerated but not re-introduced.
- **Coordinate convention**: `X = length`, `Y = width`, `Z = height`.
  Non-obvious and easy to misuse; ratified so that future algorithms
  cannot redefine it without a MAJOR amendment and a migration plan.
- **Public API stability**: the project is pre-1.0. Breaking changes to
  any symbol exported from `bin_packer_3d` are permitted but MUST be
  documented in `CHANGELOG.md` under `Unreleased` with a migration note.
- **Coverage floor**: 90 % line coverage on `src/bin_packer_3d/`.
  Dropping below blocks the PR. Raising the floor is encouraged;
  lowering it requires an amendment.

## Development Workflow & Quality Gates

### Branching and commit conventions

- Default branch: `main`. Always shippable — no in-progress work merged.
- Feature work lives on SpecKit-named branches (`NNN-<short-name>`,
  sequential), created by the `speckit.git.feature` hook.
- Commit style: **Conventional Commits** — `type(scope): subject`. Types:
  `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`, `build`,
  `ci`, `revert`. `scope` is a top-level module (`algorithms`, `config`,
  `cli`, `docs`, `ci`, etc.).
- Commit messages MUST NOT contain AI-attribution lines (no
  `Co-Authored-By: Claude …`, no `🤖 Generated` trailers).
- Breaking changes use `!` (`feat(config)!: …`) AND a `BREAKING CHANGE:`
  footer explaining the migration path.

### Pull-request workflow

PRs apply even for a solo maintainer — they are the durable audit trail,
not the approval gate.

Every PR MUST:

1. Reference the spec (or issue) it implements.
2. Pass every CI gate (Principle III).
3. Update `CHANGELOG.md` under `Unreleased` for any user-visible change.
4. Update or add tests (Principle II).
5. Update docstrings / docs for any changed public symbol
   (Principle VI).
6. Cite a "Constitution Impact" line listing which principles the PR
   touches and noting any new boundary case tested against a principle.

Merge strategy: squash-merge for features, rebase for hotfixes. No merge
commits on `main`.

### Release workflow

- Versioning: **SemVer** with project-specific meaning:
  - **MAJOR**: breaking change to any exported public API; dropping a
    supported Python version; backward-incompatible constitution
    amendment; changing the coordinate convention.
  - **MINOR**: new algorithm, constraint, output format, or CLI command;
    materially expanded guidance in the constitution.
  - **PATCH**: bug fix, clarification, wording / typo fix, internal
    refactor with no observable change.
- Release trigger: Git tag `v<MAJOR>.<MINOR>.<PATCH>` pushed to `main`.
- Release artefacts: wheel + sdist published to PyPI via an OIDC trusted
  publisher. No long-lived API tokens in repository secrets.
- The Git tag, the `CHANGELOG.md` entry, the constitution version (if
  amended in this release), and the PyPI artefact version MUST match
  exactly. Discrepancies block the release.
- A Docker image (multi-stage build, non-root runtime) is published to a
  public free-tier container registry alongside every release.

### Benchmark & reproducibility workflow

- `bin-packer benchmark` runs in CI on every push to `main` and on every
  PR that touches `src/bin_packer_3d/algorithms/`. A JSON artefact is
  attached to the build.
- The `docs/benchmarks/` page is regenerated from the CI artefact and
  committed alongside each release.
- A regression beyond 25 % utilisation drop or 25 % runtime increase on a
  fixed reference instance surfaces a CI warning that must be
  acknowledged in the PR description.

### Security & privacy workflow

- Dependabot (or equivalent) opens PRs for dependency updates. Minor and
  patch updates are auto-mergeable after CI passes.
- `SECURITY.md` at the root documents the disclosure channel.
- Before every release, a pre-release audit script scans the working
  tree AND git history for patterns matching business artefacts
  (packing-list-like `.xlsx` names, identifiable personal names,
  internal project codes). Any hit blocks the release.
- `.env` files, credentials, and tokens are never committed.
  `.env.example` lists required variables without values.

### Documentation workflow

- The documentation site build is a CI gate; a failing build blocks the
  release.
- Every algorithm ships with a reference page (complexity, pseudo-code,
  citation, guidance on when to use it).
- ADRs are required for architectural decisions that cross module
  boundaries or change a principle interpretation. Status field:
  Proposed / Accepted / Superseded.
- The README is reviewed at every MAJOR or MINOR release against the
  30-second credibility test.

## Governance

### Supremacy

The constitution supersedes every other project practice, README, or
ad-hoc convention. A disagreement between this document and any other
artefact is resolved in favour of this document until the constitution
is amended.

Every spec's `/speckit.plan` output MUST include a "Constitution Check"
gate that explicitly lists which principles the plan touches and how
they are satisfied. A plan that cannot satisfy a principle MUST propose
an amendment before proceeding.

### Amendment procedure

Amendments are first-class changes with the same rigor as code.

1. Draft the amendment as a PR modifying
   `.specify/memory/constitution.md`.
2. The PR description MUST include:
   - The version bump (MAJOR / MINOR / PATCH) and the rationale.
   - The principles or sections added, removed, or modified.
   - The Sync Impact Report covering dependent templates and guidance
     docs.
   - A migration note if any existing spec, plan, or code must update.
3. `/speckit.constitution` produces the Sync Impact Report automatically
   and prepends it as an HTML comment at the top of the file.
4. Self-review (solo maintainer) using the same PR checklist as
   features.
5. Merge with a commit message of the form:
   `docs(constitution): amend to v<NEW_VERSION> — <short description>`.
6. Tag the repository with `constitution-v<NEW_VERSION>` to make the
   amendment independently referenceable.

Amendment authorities:

- **Maintainer** (currently Bruno Ghiberto): may propose, review, and
  merge amendments.
- **Co-maintainers** (future): once added, the first action is to amend
  this procedure to require a second reviewer for MAJOR amendments.
  Adding a co-maintainer is itself a MINOR amendment.
- **Contributors**: may propose amendments via PR; merge authority
  remains with the maintainer until the procedure is updated.

### Versioning policy for the constitution

Independent of the package version. SemVer on the constitution itself:

- **MAJOR**: backward-incompatible governance or principle removal /
  redefinition (e.g. dropping "Privacy by Default", lowering the
  coverage floor below 90 %, removing the test-first requirement).
- **MINOR**: adding a new principle, adding a new section, materially
  expanding guidance within an existing principle.
- **PATCH**: wording clarifications, typo fixes, non-semantic
  refinements.

### Compliance review

- Every PR's "Constitution Impact" line is verified during self-review.
- Every release performs a full constitution compliance audit: walk
  every principle, produce a pass / fail line, and block the release on
  any fail that is not accompanied by an amendment.
- Specs reference the constitution by version
  (`Drafted against constitution v1.0.0.`). An amendment that conflicts
  with a pending spec forces the spec to be re-reviewed.

### Runtime guidance file

`CLAUDE.md` at the repository root is the agent-specific guidance file
for AI interaction with this repository. It defers to this constitution
for principles and adds only agent-specific operational notes. Any
conflict between `CLAUDE.md` and this constitution is resolved in
favour of the constitution.

**Version**: 1.0.0 | **Ratified**: 2026-04-22 | **Last Amended**: 2026-04-22
