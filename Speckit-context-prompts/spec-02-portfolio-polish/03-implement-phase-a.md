# Implement Context — Phase A: Scaffolding for spec-02 Portfolio Polish of `bin-packer-3d`

> Hand this document to `/speckit-implement` as the operational prompt
> for the FIRST invocation of spec-02. It does NOT say WHAT to do —
> that's `tasks.md` — and it does NOT say WHY — that's the
> constitution + plan + spec. It says **HOW** to execute the 24 Phase A
> tasks safely, reproducibly, and within the constitution's
> NON-NEGOTIABLE principles.
>
> **Input spec**: `specs/002-portfolio-polish/spec.md`
> (7 USs, 38 FRs, 14 SCs; US6 Deferred).
> **Plan**: `specs/002-portfolio-polish/plan.md`
> (13 sections, 8 constitutional gates PASS).
> **Tasks**: `specs/002-portfolio-polish/tasks.md`
> (92 tasks T001..T092, 9 phases).
> **This invocation's scope**: Phase A — Setup + Foundational
> (T001..T024, 24 tasks). Scaffolding only — no user-story work.
> **Constitution**: `.specify/memory/constitution.md` v1.0.1
> (8 principles, 4 NON-NEGOTIABLE).

---

## 0. Brief

This document is the execution policy that `/speckit-implement` MUST
follow when processing the Phase A range of `tasks.md`. It codifies:

- how Phase A's 24 tasks are sequenced and parallelised,
- how test-first discipline is enforced mechanically for the five
  test tasks in Phase A (T010, T011, T015, T017, T023),
- how commits are structured,
- what blocks a task from being marked complete,
- what halts the run and hands control back to the maintainer,
- the cross-track hygiene policy that keeps Phase A from colliding
  with US5 algorithm work happening in parallel on
  `feature/us5-extreme-point-benchmark`.

Without this prompt, `/speckit-implement` would invent execution policy
on the fly every run. That is unacceptable under a NON-NEGOTIABLE
test-first regime. Everything below is a RULE, not a suggestion.
Violation is a merge blocker for the implementing agent's work.

**Scope boundary**: this prompt applies to spec-02 **Phase A only**
(T001..T024). User-story bundles B'-1 through B''-2 and the Phase C
release ceremony each get their own implement context
(`04-implement-phase-b-prime-1.md` and onwards). Phase A is 100%
scaffolding — no user stories ship in this invocation, no features
become visible to the end user. The invocation closes only when the
"Foundation Ready" checkpoint (§5.2) is demonstrated.

---

## 1. Primary Input References

The `/speckit-implement` invocation MUST read these inputs and keep
pointers to them for the duration of the run.

| Artefact | Path | Role |
|---|---|---|
| Task list | `specs/002-portfolio-polish/tasks.md` | **Source of truth for WHAT**. Task IDs, `[P]` markers, `[USn]` labels, file paths. Phase A executes T001..T024 strictly in dependency order. |
| Implementation plan | `specs/002-portfolio-polish/plan.md` | Technical context (Python 3.11, deps, structure), 8 constitutional gates, three-phase delivery strategy. Consulted during ambiguity resolution (§7). |
| Spec | `specs/002-portfolio-polish/spec.md` | 7 user stories, 38 FRs, 14 SCs, Independent-Test blocks. Phase A has no user story, so spec is consulted mainly for FR cross-reference when implementing scaffolding. |
| Constitution | `.specify/memory/constitution.md` (v1.0.1) | The 8 principles, 4 NON-NEGOTIABLE. Overrides any conflicting guidance. Consulted at every gate check (§5) and every ambiguity (§7 step 2). |
| Research / ADRs | `specs/002-portfolio-polish/research.md` | 12 ACCEPTED ADRs. ADR-001 (Plotly pin + Kaleido), ADR-006 (theme), ADR-007 (palette), ADR-009 (dataset generator), ADR-012 (colourblind verification) are Phase A-critical. Never overridden by the agent. |
| Data model | `specs/002-portfolio-polish/data-model.md` | Three new entities (`VisualisationStyle`, `DemoArtifact`, `HeadlineDatasetSeed`) plus augmented `Placement.colour`. Consulted when implementing T015..T023. |
| Contracts | `specs/002-portfolio-polish/contracts/*.md` | API surface for the polish phase. Phase A consumers: `visualisation-theme.md` (T015..T021), `headline-dataset.md` (T011..T014). |
| Quickstart | `specs/002-portfolio-polish/quickstart.md` | 5-command reviewer flow. Phase A doesn't ship anything user-facing, but the quickstart is the test oracle for what the reviewer experience will look like end-to-end. |
| US5 cross-track contract (READ-ONLY) | `specs/001-public-release-hardening/spec.md § US5` + `specs/001-public-release-hardening/tasks.md § T085..T160` | Reference only — describes the parallel US5 algorithm track. Consulted to enforce the no-touch list (§9). |
| Plan context prompt | `Speckit-context-prompts/spec-02-portfolio-polish/02-plan.md` § 11 | Authoritative source for the US5 cross-track hygiene rules referenced in §9. |
| CLAUDE.md | repo root | Project-level agent guidance. Should not be edited by `/speckit-implement` except via explicit task (none in Phase A). |

On invocation start, load them all into context. There is only one
Phase A invocation, so no reload cycle to worry about.

---

## 2. Execution Scope per Invocation (DR-1 LOCKED)

**Rule**: one invocation = one **phase/bundle**, not one task, not
one release-milestone, not the whole feature.

This is the DR-1 decision for spec-02. It differs from spec-01's DR-1
(which scoped invocations to single user stories) because spec-02
groups user stories into bundles per the plan's three-phase delivery
strategy (A scaffolding / B' delivery / B'' P2 / C release ceremony).

### Why per-phase/bundle

- Phase A has no user-story boundary — it is pure scaffolding that
  unblocks all downstream stories. There is no Independent-Test from
  `spec.md` to use as a phase-close gate. Instead the gate is the
  multi-criteria "Foundation Ready" checkpoint defined in §5.2.
- Each later bundle (B'-1, B'-2, B'-3, B''-1, B''-2) ends with one or
  more user-story Independent-Tests from `spec.md`. The bundle is the
  natural review unit because cross-story dependencies (US1 depends on
  US2 Kaleido work) make per-US invocations impractical.
- Phase C is the release ceremony — version bump, tag prep, docs
  deploy verification, post-merge maintainer steps.

### Invocation sequence for spec-02

| Order | Phase / Bundle | Task range | Milestone |
|---|---|---|---|
| 1 | **Phase A — Scaffolding (this invocation)** | T001..T024 (24 tasks) | unblocks all USs |
| 2 | Phase B'-1 — US2 + US1 (visualisation + README) | T025..T041 (17 tasks) | v0.3.0-rc1 P1 |
| 3 | Phase B'-2 — US3 (docs site) | T042..T058 (17 tasks) | v0.3.0-rc1 P1 |
| 4 | Phase B'-3 — US4 (repo surface) | T059..T066 (8 tasks) | v0.3.0-rc1 P1 |
| 5 | Phase B''-1 — US5 (demo command) | T067..T074 (8 tasks) | v0.3.0-rc1 P2 |
| 6 | Phase B''-2 — US7 (CV identity) | T075..T079 (5 tasks) | v0.3.0-rc1 P2 |
| 7 | Phase C — Polish & Release | T080..T092 (13 tasks) | v0.3.0-rc1 tag |

Seven invocations total. Maintainer triggers each via a new
implement-context prompt; the agent does NOT chain phases on its own.

### What "one invocation" means mechanically for Phase A

- Agent reads `tasks.md` T001..T024 and the inputs in §1.
- Runs tasks in dependency order (§3.3) with `[P]`-parallelism (§3.2).
- Commits per DR-2 policy (§4).
- At phase end: runs the "Foundation Ready" multi-criteria checkpoint
  (§5.2), reports the result to the maintainer, exits.
- Does NOT auto-proceed to Phase B'-1 even if context budget remains.
- Does NOT open a PR — that is a separate explicit maintainer ask
  after the agent reports green.

---

## 3. Execution Policy

### 3.1 Test-first enforcement (NON-NEGOTIABLE — Constitution II)

Strict TDD mode is enabled (per project `CLAUDE.md`). For every task
labelled as a **test task** in Phase A:

| Test task | Test file | Implementation pair |
|---|---|---|
| T010 | `tests/integration/test_install_footprint.py` | T024 (CI job activates the test) |
| T011 | `tests/unit/test_dataset_generator.py` | T012 (`scripts/generate_headline_dataset.py`) |
| T015 | `tests/unit/test_visualization_theme.py` | T016 (`src/bin_packer_3d/visualization/theme.py`) |
| T017 | `tests/unit/test_palette_colourblind.py` | T019 (`src/bin_packer_3d/visualization/palette.py`) |
| T023 | `tests/unit/test_placement_colour.py` | T022 (`Placement.colour` augmentation) |

For each pair:

1. Author the test code per the task description.
2. Run `pytest <specific_test_file>` and **CONFIRM it FAILS** (red).
   If the test passes on first run, STOP — either the test is
   insufficiently strict, the implementation already exists (redundant
   task), or the import path is wrong. Surface to the maintainer;
   do not commit a "test that never fails".
3. Commit the failing test with message `test(scope): add <what> (T###)`
   and body line: `verified red: pytest <file> exits 1 with N expected
   failures before this commit`.
4. Proceed to the paired implementation task(s).
5. After implementation: run the same test and **CONFIRM it now
   PASSES**.
6. Commit the implementation per DR-2 with body line `tests authored
   first (<test-commit-SHA> fails green at HEAD~1)`.

No implementation task may be committed without a preceding failing-test
commit earlier in this Phase A commit log.

**Note on T023 ↔ T022 ordering**: T023 (test) must commit red before
T022 (implementation). However, T022 also requires T016 (theme module)
and T019 (palette module) to be committed first — palette's
`colour_for_box` is consumed by `Placement.colour`. The correct
sequence is:

1. T015 red → T016 impl (theme module green)
2. T017 red → T019 impl (palette module green) → T020 (commit
   colourblind-check PNG)
3. T023 red → T022 impl (Placement.colour green)

### 3.2 Parallelism for `[P]` tasks

`[P]`-tagged tasks touch disjoint files and have no dependency on
incomplete tasks. Phase A `[P]` islands:

- **Island 1 (Setup config)**: T001, T002, T003, T004, T005 all touch
  the same `pyproject.toml` or `.gitignore`. The `[P]` marker assumes
  the agent applies all edits in one tool call (parallel Write/Edit)
  and commits as a composite: `chore(deps): pin Plotly, add viz extra,
  add colorspacious dev dep, exclude examples (T001..T005)`.
- **Island 2 (Foundational placeholders)**: T007, T008 — two empty
  filesystem operations, independent. Single composite commit
  acceptable: `chore: scaffold examples/README and snapshot fixture
  dir (T007, T008)`.
- **Island 3 (Foundational tests)**: T010, T011, T015, T017 are
  parallel-safe red-commits (different test files, different topics).
  May be authored + committed in any order so long as each is verified
  red on its own before commit.
- **Island 4 (Foundational scripts + impl)**: T018 (verify_palette
  script), T021 (`__init__.py` re-exports), T022 (placement augment),
  T023 (placement test) — all parallel-safe per `tasks.md` markers,
  modulo the test-first ordering in §3.1.

Non-`[P]` tasks (T006, T009, T012, T013, T014, T016, T019, T020, T024)
are strictly sequential and must respect §3.3 dependency edges.

### 3.3 Dependency respect

`tasks.md` Phase 1 + Phase 2 section is authoritative. Intra-phase
dependency edges:

```
T001..T005 (Setup [P])
      ↓
T006 (CHANGELOG placeholder, blocks nothing downstream)
      ↓
T007, T008 (Foundational placeholders [P])
      ↓
T001..T004 must be MERGED before T009 (install footprint baseline
must reflect the new pyproject configuration)
      ↓
T009 → T010 (test that consumes the baseline)
      ↓
T011 (red) → T012 (impl: scripts/generate_headline_dataset.py)
      ↓
T012 + T013 (seed file) → T014 (run generator, commit headline.csv)
      ↓
T015 (red) → T016 (impl: theme.py)
      ↓
T017 (red) → T019 (impl: palette.py)
      ↓
T018 + T019 → T020 (run verify_palette_colourblind script, commit PNG)
      ↓
T021 (__init__ re-exports — depends on T016 + T019)
      ↓
T016 + T019 → T022 (Placement.colour augmentation)
      ↓
T023 (red) → T022 commit sequence per §3.1 note
      ↓
T024 (_ci-core.yml install-footprint job — depends on T010 having
       been authored; can be merged before T010 turns green)
```

If the agent spots an ordering error in `tasks.md` (e.g., a `[P]`
marker on a task that actually has a dependency), it does NOT silently
reorder — see DR-7 (§7).

### 3.4 Progress-report format

At phase end, agent emits a compact report:

```
Phase: A — Scaffolding (invocation 1 of 7)
Tasks: T001..T024 (24 tasks, 0 skipped, 0 deferred)
Commits: M on feature/002-portfolio-polish (see git log -M)
Tests: 103 baseline + N new = total / 0 fail, 1 skip (pre-existing)
Coverage: X.X % on src/bin_packer_3d/ (Y % lowest module: <module>)
Mypy: 0 errors (strict)
CHANGELOG: [Unreleased] updated with K bullets under Added
Foundation Ready (§5.2): PASS — all 6 sub-criteria green
  - pip install -e .: OK
  - pip install -e .[viz]: OK
  - Public re-exports importable: OK
  - examples/headline.csv exists + byte-identical regen: OK
  - docs/assets/palette_colourblind_check.png exists + dE >= 15: OK
  - test_install_footprint.py green on clean venv: OK
Next recommended invocation: Phase B'-1 (US2 + US1, T025..T041)
```

No prose outside this shape unless the maintainer asks for it.

---

## 4. Commit Policy (DR-2 + DR-3 LOCKED)

### 4.1 Conventional Commits format

`type(scope): subject ≤ 72 chars`

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`,
`build`, `ci`, `revert` (Constitution §Commit style).

**Phase A scope vocabulary** (use these or surface a new scope to the
maintainer):

| Scope | Used for | Phase A tasks |
|---|---|---|
| `deps` | pyproject.toml dependency edits | T001, T002, T003 |
| `chore` | non-functional config + placeholder dirs | T004, T005, T006, T007, T008 |
| `ci` | `_ci-core.yml`, CI fixtures | T009, T010, T024 |
| `data` | dataset generator, seed, CSV | T011, T012, T013, T014 |
| `viz` | theme, palette, palette-check artefacts | T015..T020 |
| `models` | placement augmentation | T021, T022, T023 |
| `tests` | test-only commits (red authoring) | any test-only commit |

Imperative mood: "add X", "fix Y", "remove Z" — never "added", "fixes",
"removing".

### 4.2 Grouping rule (DR-2)

- **One commit per task**, OR
- **One commit per tight (test + implementation) pair** — e.g.,
  T015 + T016 as `feat(viz): add Plotly theme module (T015, T016)`
  with the test-first body line referencing the red-commit SHA.
- **Composite commits allowed for tight `[P]` islands** that share a
  single file or concern (e.g., T001..T005 as one
  `chore(deps): scaffold spec-02 pyproject + ignore rules (T001..T005)`
  is acceptable).
- **Never** batch the entire Phase A into one commit — loses
  bisectability and breaks Constitution §Compliance review.

Target: 12–18 commits across Phase A, depending on how many test+impl
pairs vs single-task commits the agent chooses.

### 4.3 Body conventions

- Blank line after subject.
- Wrap body at 72 columns.
- Reference task IDs this commit implements in the subject as `(T###)`
  OR in the body as `Implements: T###, T###`.
- For test-first implementation commits: include the line
  `tests authored first (<test-commit-SHA> fails green at HEAD~1)`
  once the paired test commit exists.
- For test red-commits: include the line
  `verified red: pytest <test-file> exits 1 with N expected failures
  before this commit`.
- For T020 (commits a generated PNG): include `generated:
  deterministic via scripts/verify_palette_colourblind.py @ <script-SHA>`
  for auditability — the next regeneration must produce a byte-identical
  PNG.
- For T014 (commits a generated CSV): include `generated: deterministic
  via scripts/generate_headline_dataset.py @ <script-SHA> with seed
  examples/headline.seed @ <seed-SHA>`.
- Breaking changes: NOT expected in Phase A (scaffolding only, no
  public-API removals). If a breaking change is unavoidable, STOP and
  surface — the plan does not authorise breaking changes in Phase A.

### 4.4 NO AI attribution (DR-3)

Constitution §Commit style forbids:

- `Co-Authored-By: Claude …`
- `🤖 Generated with Claude Code`
- any `Generated-By:`, `Assisted-By:`, `AI-` trailer.

This is hard-enforced. The agent MUST NOT add any such line. A commit
with an AI-attribution trailer is a violation the agent MUST fix via
`git commit --amend` (or, if already pushed, via a follow-up
correction commit — never rewrite published history without maintainer
approval).

### 4.5 Per-phase CHANGELOG update

At Phase A close, agent confirms `CHANGELOG.md` `## [Unreleased]`
section contains at least three bullets under `### Added` covering:

- viz: branded Plotly theme + deterministic colourblind-safe palette.
- data: procedurally generated headline dataset + seed.
- ci: install-footprint regression guard.

Bullets are added inline by the impl commits (T016, T019, T014, T024)
— not in a separate "update CHANGELOG" commit. T006 only seeds the
placeholder; per-task commits flesh it out.

These bullets migrate to the versioned section `[0.3.0-rc1]` at
release-tag time — NOT by the agent, by the maintainer in Phase C.

---

## 5. Gate Verification

Gates map to Constitution principles. Each gate has a run-condition
(when it fires) and a fail-action (what happens if red).

### 5.1 Post-task gates

After EVERY implementation task (non-test), before commit:

| Gate | Command | Fail action |
|---|---|---|
| G-Ruff-Lint | `ruff check <touched files>` | Auto-fix via `ruff check --fix`; re-run; if still red, STOP |
| G-Ruff-Format | `ruff format --check <touched files>` | Auto-fix via `ruff format`; re-run; if still red, STOP |
| G-Mypy | `mypy --strict src/bin_packer_3d/` | See §6.3 — narrow ignore + TODO + CHANGELOG tech debt |
| G-Targeted-Test | `pytest <test file paired with this task> -x` | See §6.2 |

Green on all four = task is committable.

### 5.2 Post-phase gates — Foundation Ready Checkpoint

After T024 commits and before the progress report (§3.4) emits:

| Gate | Command | Fail action |
|---|---|---|
| G-Full-Pytest | `pytest tests/ -m "not slow"` | STOP — all-suite failure is a show-stopper |
| G-Baseline | `pytest tests/` produces ≥103 passing (no spec-01 regression) | STOP |
| G-Coverage-Overall | `pytest --cov=src/bin_packer_3d --cov-fail-under=90` (DR-5) | STOP — below-floor blocks phase completion |
| G-Coverage-Module | parse `coverage.xml`, flag any module < 70 % | WARNING — record in progress report, don't block |
| G-Mypy-Full | `mypy --strict src/bin_packer_3d/` | STOP |
| G-Changelog | `## [Unreleased]` has ≥3 new bullets under `### Added` | STOP |
| **G-Foundation-Ready** | Six sub-criteria below — ALL must pass | STOP if any fail |

**Foundation Ready sub-criteria** (replaces spec-01's per-US
Independent-Test for Phase A):

1. **Install (no extras)**: from a clean
   `python -m venv .venv-phaseA-check` +
   `.venv-phaseA-check/bin/pip install -e .` exits 0; pinned Plotly
   resolves to `>=5.18,<6.0`.
2. **Install (viz extra)**: `pip install -e .[viz]` exits 0;
   `python -c "import kaleido"` exits 0.
3. **Public re-exports**: `python -c "from bin_packer_3d import
   VisualisationStyle, BIN_PACKER_3D_DARK, BIN_PACKER_3D_LIGHT,
   apply_theme, colour_for_box"` exits 0.
4. **Headline dataset deterministic**: `python
   scripts/generate_headline_dataset.py --seed examples/headline.seed
   --out /tmp/headline-redo.csv && cmp examples/headline.csv
   /tmp/headline-redo.csv` exits 0.
5. **Colourblind artefact**: `docs/assets/palette_colourblind_check.png`
   exists, is non-empty, and `python
   scripts/verify_palette_colourblind.py --check-only` (or equivalent
   ΔE re-verification path) exits 0 with the asserted ≥15 threshold.
6. **Install-footprint regression guard active**: `pytest
   tests/integration/test_install_footprint.py` exits 0 on a clean
   venv (consumed by the new CI job from T024).

The agent runs each sub-criterion as a shell check, captures pass/fail,
and reports in the progress report (§3.4). Any FAIL blocks the
invocation from being marked complete.

### 5.3 Pre-release gates

**N/A for Phase A.** Pre-release gates (constitution audit, SC-006
benchmark, privacy audit, docs build, quickstart smoke, tag/CHANGELOG
match) fire only at the Phase C release ceremony. See the future
`07-implement-phase-c.md` prompt.

### 5.4 Gate failure philosophy

- **FAIL LOUD**. The agent never uses `pytest ... || true`, never sets
  `continue-on-error: true` in a CI job, never suppresses mypy via
  bare `# type: ignore`. Every suppression is targeted AND justified.
- Gates that are flagged as WARNINGs (per-module coverage) surface in
  the progress report and accrue in CHANGELOG `[Unreleased] / ### Debt`
  if any are introduced (Phase A is not expected to introduce debt).

---

## 6. Failure & Recovery Policy

### 6.1 Implementation-time test failure (red after code written)

1. Re-read the failing assertion and the task description.
2. Attempt a direct fix. Commit-amending the WIP is fine — nothing is
   pushed yet.
3. If two fix attempts don't pass: STOP, surface to maintainer with:
   - task ID,
   - test name + failure output (last 30 lines),
   - the hypothesis being attempted,
   - what the agent tried and why it didn't work.
4. NEVER `pytest --no-header` / `pytest -k "not failing"` to hide it.
   NEVER mark the test `@pytest.mark.skip` without an ADR justifying
   the skip.

### 6.2 Lint / format failure

Auto-fixable: run `ruff check --fix` + `ruff format`; re-verify;
continue. Not auto-fixable (rare): read the rule, apply manually,
verify.

### 6.3 `mypy --strict` failure

Preferred: fix the type issue. Strict mode catches real bugs — treat
it as such, not as a nuisance.

Escape hatch (only when the alternative is a large unrelated
refactor):

1. Add `# type: ignore[<exact-rule-name>]` — never bare
   `# type: ignore`.
2. Add adjacent TODO comment referencing the task ID:
   `# TODO(T###): <why> — revisit in <future spec / phase>`
3. Add CHANGELOG.md `[Unreleased] / ### Debt` entry noting the tech
   debt: `- [debt] <module>:<line> narrow type-ignore pending <reason>`.

This trifecta is the ONLY way a `# type: ignore` enters the tree. Phase
A is not expected to require any — the new modules are small and
type-clean by design.

### 6.4 External failures

- GitHub API rate limit / transient 5xx: exponential backoff (2 s,
  4 s, 8 s); give up after 3 attempts; surface to maintainer.
- `pip install` failure during Foundation Ready sub-criteria (§5.2):
  inspect the error; if missing system packages, surface the exact OS
  command needed. Common cause for Kaleido: missing Chromium
  dependencies on minimal containers (Phase A assumes the maintainer's
  dev machine has them).

### 6.5 Unexpected repo state

Cases requiring STOP + maintainer handoff (no auto-resolution):

- Merge conflict on any file the current task touches.
- Uncommitted changes from a prior invocation that don't match the
  expected WIP pattern.
- Stash entries older than the session.
- Branch diverged from `develop` in an unexpected way.
- Any reference to a spec ID other than `002-portfolio-polish` in a
  file the agent didn't author.
- Modifications to any file in the US5 no-touch list (§9) — even a
  whitespace edit triggers immediate STOP.

The agent describes the state, lists the three most-likely causes,
and asks the maintainer before touching anything.

---

## 7. Ambiguity Resolution

A task description might be ambiguous at execution time (e.g.,
T015 "axis-label format follows `{name} (mm)` template" — does "name"
mean the axis identifier or a custom display label?). Decision tree:

**Step 1 — Consult design artefacts** (2-min budget):

- `plan.md` § Project Structure: file path authoritative.
- `research.md` ADR on the topic: decision authoritative. For Phase A
  the relevant ADRs are 001 (deps), 006 (theme), 007 (palette), 009
  (dataset), 012 (colourblind).
- `data-model.md`: entity shape authoritative (e.g., the
  `VisualisationStyle` frozen dataclass fields).
- `contracts/*.md`: public-surface authoritative
  (`visualisation-theme.md` and `headline-dataset.md` are Phase A's
  binding contracts).

If a clear answer emerges, proceed with an inline one-line comment in
the commit body: `Resolved via ADR-006 (research.md)`.

**Step 1.5 — Cross-track hygiene check** (NEW for spec-02, 30-sec
budget):

Before touching any file under `src/bin_packer_3d/`, consult the §9
no-touch list AND
`Speckit-context-prompts/spec-02-portfolio-polish/02-plan.md` § 11.
If the target file is in US5 territory, STOP immediately — this is not
an ambiguity, this is a boundary violation.

**Step 2 — Consult constitution** (30-sec budget):

Does a principle prescribe a default? Examples:

- Principle I (Contract Honesty) — Phase A's re-exports (T021) must
  source from the same module that owns the symbol; no aliasing.
- Principle IV (Reproducibility & Determinism) — any randomised
  process (dataset gen, palette assignment) must produce
  bit-identical output for the same `(seed, input)`.
- Principle V (Library Citizenship) — new modules (`theme.py`,
  `palette.py`) MUST NOT use `print()`, MUST NOT configure logging,
  MUST NOT have import-time side effects.

If a principle applies, proceed with commit body citation:
`Resolved via Constitution §IV (Reproducibility & Determinism)`.

**Step 3 — STOP and surface (high-stakes)**:

If the ambiguity affects a public surface, test contract, commit
graph, or cross-track boundary, STOP. Surface with:

```
Task: T0XX — <one-line description>
Ambiguous phrase: "<exact phrase from task>"
Options:
  A) <concrete option 1> — consequences: <...>
  B) <concrete option 2> — consequences: <...>
  C) <concrete option 3> — consequences: <...>
Recommendation: <A|B|C> because <rationale from spec/plan>
```

Wait for maintainer's single-letter answer.

**Step 4 — Take a default (low-stakes only)**:

Ambiguities the agent MAY resolve silently:

- Private helper function signatures (not exported).
- Log message wording (Phase A introduces no new logging).
- Internal variable naming.
- Test-case parameter values (seeds, dimensions) if the test is
  property-based.
- File-order within a task's scope.

Document the default in the commit body: `Default: <what>`.

---

## 8. Hard Stopping Points

**Phase A has ZERO task-specific hard stops.** All 24 tasks
(T001..T024) are agent-executable end-to-end. No GitHub UI actions, no
`git filter-repo`, no PyPI publish, no tag pushes, no irreversible
history rewrites.

This is a deliberate property of Phase A: scaffolding is by definition
internal and reversible. The first task that requires a hard stop is
T030 in Phase B'-1 (vhs hero GIF recording, which needs a TTY on the
maintainer's Linux machine).

Additionally, the agent NEVER performs the following without explicit
per-action maintainer consent, regardless of task:

- Force-push to any remote branch (`git push --force[-with-lease]`).
- `git reset --hard` on a branch that has remote commits.
- `git branch -D` on any branch with unmerged work.
- `git filter-repo` anywhere.
- `git rebase -i` on published commits.
- `git commit --amend` on a commit that has been pushed.
- `rm -rf` on anything outside `dist/`, `build/`, `.pytest_cache/`,
  `__pycache__`, `.mypy_cache/`, `.ruff_cache/`, `htmlcov/`,
  `.venv-phaseA-check/` (the temporary Foundation Ready check venv).
- Any GitHub API mutation (no PR creation / merging / closing without
  maintainer say-so).

---

## 9. Cross-Track Hygiene Policy (NEW vs spec-01)

Spec-02 ships in parallel with US5 algorithm work on
`feature/us5-extreme-point-benchmark`. Both branches were cut from
`develop` at `08ca725` and will eventually merge back to `develop`. To
prevent merge conflicts and accidental scope creep, Phase A enforces a
strict no-touch list and a defensive rebase protocol.

### 9.1 No-touch list (US5 territory — DR-9 LOCKED)

The agent MUST NOT create, modify, rename, or delete any of these
files during Phase A:

- `src/bin_packer_3d/algorithms/extreme_point.py` (US5 creates this)
- `src/bin_packer_3d/algorithms/maximal_rectangles.py` (US5 creates this)
- `src/bin_packer_3d/benchmark/runner.py` (US5 implements this; the
  v0.2.0 scaffolding is in `feature/us5-extreme-point-benchmark`'s
  reserved space)
- Any new file inside `src/bin_packer_3d/benchmark/` beyond what
  spec-01 US5 Part 1 already merged.

Reading these files for context is allowed (e.g., to understand the
existing `ALGORITHMS` registry interface). Writing is not.

### 9.2 Touch-with-care list

The agent MAY touch these files but must add a `cross-track: spec-02
Phase A` body line to the commit and minimise the diff:

- `src/bin_packer_3d/__init__.py` (T021 re-exports `VisualisationStyle`,
  `apply_theme`, `colour_for_box`). US5 may later add re-exports for
  new packers; coordinate via the registry pattern — do not pre-stage
  US5 symbols here.
- `src/bin_packer_3d/algorithms/__init__.py` (Phase A does NOT modify
  this; mentioned because US5 will. If a task description in Phase A
  appears to require modifying it, STOP and surface — likely a
  cross-track boundary violation slipped in).

### 9.3 Rebase protocol

If during Phase A the maintainer reports that US5 Part 2 has merged to
`develop`, the agent:

1. STOPS at the current task boundary (does NOT continue mid-task).
2. Runs `git fetch origin develop`.
3. Runs `git rebase origin/develop` on `feature/002-portfolio-polish`.
4. If rebase clean: re-runs `pytest tests/ -m "not slow"` to confirm
   the 103-test baseline still passes alongside the new US5 tests.
   Surface a one-line confirmation: `rebased on develop @ <sha>; tests
   still green`.
5. If rebase produces conflicts: STOP and surface the conflicted file
   list. The maintainer decides resolution.

### 9.4 Drift-test note (forward-looking)

T042 (in Phase B'-2, US3 docs) introduces
`tests/integration/test_docs_build.py` asserting "every key in
`ALGORITHMS` has a corresponding `docs/algorithms/<key>.md`". If US5
Part 2 has merged before Phase B'-2, the new `extreme_point` and
`maximal_rectangles` algorithm pages will need stubs. This is **NOT**
a Phase A concern — flagged here for situational awareness so the
agent does not pre-stage docs for not-yet-merged packers.

---

## 10. Memory & Telemetry

The project uses Engram for cross-session memory. Protocol per
project `CLAUDE.md` + global Engram instructions.

### 10.1 After each tight task group

Call `mem_save` with:

- **title**: verb + object, e.g., "Authored visualization theme module
  (T015, T016)".
- **type**: `decision`, `bugfix`, `architecture`, `pattern`, `config`,
  `discovery`, `learning`.
- **topic_key**: stable key under the reserved spec-02 prefix.
- **content** in the 4-block format (`**What**`, `**Why**`, `**Where**`,
  `**Learned**`).
- **scope**: `project`.

### 10.2 At phase close (before exiting)

Call `mem_session_summary` per the §Protocol in the project
`CLAUDE.md` / global Engram instructions. Use the Goal / Instructions /
Discoveries / Accomplished / Next Steps / Relevant Files shape.

### 10.3 Topic keys for spec-02 Phase A

Reserve these prefixes — do not drift:

- `3d-bin-packing/spec-02/phase-a/setup/*` (T001..T006)
- `3d-bin-packing/spec-02/phase-a/foundational/install-footprint/*` (T009, T010, T024)
- `3d-bin-packing/spec-02/phase-a/foundational/dataset/*` (T011..T014)
- `3d-bin-packing/spec-02/phase-a/foundational/theme/*` (T015, T016, T021)
- `3d-bin-packing/spec-02/phase-a/foundational/palette/*` (T017..T020)
- `3d-bin-packing/spec-02/phase-a/foundational/placement/*` (T022, T023)

Cross-cutting findings (e.g., a gotcha that affects Phase B' too):
`3d-bin-packing/spec-02/cross-phase/*`.

### 10.4 Pre-flight memory search

Before the first task of the invocation, run
`mem_search query:"spec-02 phase-a"` to discover any prior partial
Phase A work. If hits are returned, retrieve via `mem_get_observation`
and surface to the maintainer before proceeding — this prevents
double-writes if a prior session crashed mid-Phase-A.

### 10.5 Do NOT save

- Trivial "task T0XX completed" pings (already in git log).
- Content copied from the spec / plan / constitution (already on disk).
- Code snippets the agent just wrote (already committed).

DO save: non-obvious decisions, gotchas discovered, deviations from
plan, anything that saves a future session a re-investigation.

---

## 11. Environment Setup

Before the first task of the invocation, the agent runs a pre-flight
check and reports status. If any check fails: STOP and surface.

| Check | Command | Pass condition |
|---|---|---|
| Python version | `python --version` | `3.11`, `3.12`, `3.13`, or `3.14` |
| Package install | `pip show bin_packer_3d` | editable install at repo path |
| Dev extras present | `python -c "import pytest, ruff, mypy, hypothesis"` | all importable |
| Colorspacious available | `python -c "import colorspacious"` | importable after T003 merges; acceptable to fail BEFORE T003 |
| Pre-commit active | `pre-commit --version` + `.git/hooks/pre-commit` exists | both yes |
| Current branch | `git branch --show-current` | `feature/002-portfolio-polish` |
| Active feature pointer | `jq -r .feature_directory < .specify/feature.json` | `specs/002-portfolio-polish` |
| Working-tree state | `git status --porcelain` | empty (Phase A is invocation 1; no prior WIP expected) |
| Remote configured | `git remote -v` | `origin` points at Bruno-Ghiberto/3D_BIN_PACKING |
| Baseline tests green | `pytest tests/ -m "not slow"` | 103 pass, 0 fail (spec-01 regression guard) |
| US5 territory clean | check that `src/bin_packer_3d/algorithms/extreme_point.py`, `maximal_rectangles.py`, and any new files under `benchmark/` do not exist in the working tree | absent OR untouched since `08ca725` |

If the maintainer starts the invocation from a cold start (new
machine, new clone), `CONTRIBUTING.md` contains the full setup path —
the agent may point the maintainer at it rather than running setup
itself.

---

## 12. Completion & Handoff

### 12.1 "Done" per Phase A

All of:

1. Every task in T001..T024 has its checkbox flipped to `[x]` in
   `tasks.md` (agent edits this at commit time for the relevant task).
2. All post-phase gates green (§5.2), including the six-criterion
   Foundation Ready checkpoint.
3. CHANGELOG.md `[Unreleased] / ### Added` updated with at least three
   bullets covering the viz / data / ci additions (§4.5).
4. Phase-close progress report emitted (§3.4) AND
   `mem_session_summary` saved (§10.2).

### 12.2 PR strategy (DR-6 + DR-10 LOCKED for Phase A)

- DR-6: Direct commits on `feature/002-portfolio-polish`; one PR per
  phase to `develop`; release tag from `main` post-merge in Phase C.
- DR-10: **Phase A ships as a single PR to `develop`** with all
  Phase A commits preserved (merge-commit, not squash). Estimated
  ~770 LOC across ~24 commits. If the maintainer's branch-protection
  configuration enforces a ≤400 LOC PR cap, Phase A carries a
  `size:exception` label justified by "scaffolding bundle —
  indivisible Foundation Ready unit; ~58% of LOC is tests + data +
  config".

The agent does NOT open the PR — that is a maintainer action after
the agent's handoff message reports green.

### 12.3 When to re-run `/speckit-tasks`

If during Phase A the agent identifies a task that is MISSING from
`tasks.md` (not a task-reorder issue — a genuine gap), it STOPS and
surfaces. The maintainer decides whether to:

- re-run `/speckit-tasks` to regenerate (tasks.md rewritten end-to-end),
  OR
- amend `tasks.md` inline with the new task ID (e.g., `T024a`).

The agent does NOT extend `tasks.md` on its own.

### 12.4 When to re-run `/speckit-analyze`

Optional at Phase A close. MANDATORY at each release-milestone
boundary (next one: pre-Phase-C-tag). MANDATORY if any spec/plan
amendment occurs mid-implementation.

### 12.5 Handback message shape at invocation end

```
Invocation complete — Phase A Scaffolding (1 of 7).
Result: <PASS | PARTIAL | FAIL>
Commits: M on feature/002-portfolio-polish
Tests: 103 baseline + N new = total / 0 fail
Coverage: X.X% (≥90% required)
Mypy: 0 errors (strict)
Foundation Ready: <PASS | FAIL — <which sub-criteria failed>>
Next recommended invocation: Phase B'-1 (US2 + US1, T025..T041)
Open questions for maintainer (if any): <list>
Hard stops encountered: <list with task IDs, or none>
Cross-track events: <e.g., rebased on develop @ <sha>, or none>
```

---

## 13. Open Questions

At authoring time of THIS prompt — **none**. Phase A does not depend
on the three deferred maintainer decisions tracked in spec-02 (US2
ordering, LINKEDIN.txt disposition, CV authorship) — all three land in
later bundles.

If the maintainer invokes `/speckit-implement` for Phase A with open
questions still listed in this section, the agent MUST refuse to
proceed and request resolution first. Keeping this section empty is
itself a gate.

### Future-proofing

If a later amendment to spec-02 or the constitution introduces a new
concern that this prompt doesn't cover, the concern lands here until
the prompt itself is amended via a new design cycle.

Examples of what WOULD belong here if it arose (none do today):

- "If US5 Part 2 merges to develop before Phase A completes, do we
  still ship Phase A as planned or sub-divide?"
- "If the install-footprint baseline (T009) measures higher than
  expected because of a transitive Plotly dep, do we negotiate the
  +5% budget or pin a lower-footprint Plotly variant?"

---

## Locked decisions summary

| DR | Topic | Locked value |
|---|---|---|
| DR-1 | Execution scope per invocation | **Per phase/bundle** — 7 invocations for spec-02 |
| DR-2 | Commit grouping | One commit per task OR tight test+impl pair OR `[P]`-island composite; never whole-phase |
| DR-3 | AI attribution on commits | NONE — no Co-Authored-By, no robot trailer |
| DR-4 | Interactive vs unattended | Interactive — maintainer triggers each invocation |
| DR-5 | Coverage gate | Overall ≥90% blocking; per-module ≥70% warning-only |
| DR-6 | Branch & PR workflow | Direct commits on `feature/002-portfolio-polish`; single PR per phase to `develop`; release tag from `main` in Phase C |
| DR-7 | Task re-ordering authority | None — agent surfaces ordering concerns, never reorders silently |
| DR-8 | Mid-invocation skill use | Read-only only (`/sc:explain`, `/sc:analyze`); no scope-changing skills |
| **DR-9** | **US5 cross-track no-touch list** | 3 files locked (`algorithms/extreme_point.py`, `algorithms/maximal_rectangles.py`, `benchmark/runner.py`); rebase-on-merge protocol enforced |
| **DR-10** | **PR sizing for Phase A** | Single PR to `develop`, ~24 commits preserved via merge-commit; `size:exception` label if branch protection enforces a LOC cap |

These are not suggestions. The agent executing `/speckit-implement`
against spec-02 Phase A MUST obey all ten.
