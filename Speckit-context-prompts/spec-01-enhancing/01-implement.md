# Implement Context — Phase 01: Public-Release Hardening of `bin-packer-3d`

> Hand this document to `/speckit.implement` as the operational prompt.
> It does NOT say WHAT to do — that's `tasks.md` — and it does NOT say
> WHY — that's the constitution. It says **HOW** to execute the 162
> tasks safely, reproducibly, and within the constitution's
> NON-NEGOTIABLE principles.
>
> **Input spec**: `specs/001-public-release-hardening/spec.md`
> (9 USs, 54 FRs, 14 SCs).
> **Plan**: `specs/001-public-release-hardening/plan.md`
> (13 sections, 8 constitutional gates PASS).
> **Tasks**: `specs/001-public-release-hardening/tasks.md`
> (162 tasks T001..T160 + T105a + T157a, 12 phases).
> **Constitution**: `.specify/memory/constitution.md` v1.0.1 (8 principles,
> 4 NON-NEGOTIABLE).

---

## 0. Brief

This document is the execution policy that `/speckit.implement` MUST
follow when processing `tasks.md`. It codifies:

- how tasks are batched per invocation,
- how test-first discipline is enforced mechanically,
- how commits are structured,
- what blocks a task from being marked complete,
- what halts the run and hands control back to the maintainer.

Without this prompt, `/speckit.implement` would invent execution policy
on the fly every run. That is unacceptable under a NON-NEGOTIABLE
test-first regime. Everything below is a RULE, not a suggestion.
Violation is a merge blocker for the implementing agent's work.

Scope boundary: this prompt applies to spec-01 only. Later specs will
author their own implement contexts, potentially with different policies.

---

## 1. Primary Input References

The `/speckit.implement` invocation MUST read these inputs and keep
pointers to them for the duration of the run.

| Artefact | Path | Role |
|---|---|---|
| Task list | `specs/001-public-release-hardening/tasks.md` | **Source of truth for WHAT**. Task IDs, [P] markers, [USn] labels, file paths. Executed strictly in dependency order. |
| Implementation plan | `specs/001-public-release-hardening/plan.md` | Technical context (Python 3.11, deps, structure), 8 constitutional gates, phased milestones. Consulted during ambiguity resolution (§7). |
| Spec | `specs/001-public-release-hardening/spec.md` | 9 user stories, 54 FRs, 14 SCs, Independent-Test blocks. Consulted to verify a user-story phase is truly "done" (§11). |
| Constitution | `.specify/memory/constitution.md` (v1.0.1) | The 8 principles, 4 NON-NEGOTIABLE. Overrides any conflicting guidance. Consulted at every gate check (§5) and every ambiguity (§7 step 2). |
| Research / ADRs | `specs/001-public-release-hardening/research.md` | 10 ACCEPTED ADRs. Consulted during ambiguity resolution (§7 step 1). Never overridden by the agent. |
| Data model | `specs/001-public-release-hardening/data-model.md` | 14 entities with invariants. Consulted when implementing a new model or changing an existing one. |
| Contracts | `specs/001-public-release-hardening/contracts/*.md` | API surface (api.md), CLI (cli.md), config-schema (config-schema.md), benchmark JSON (benchmark-format.md). Authoritative for public-facing shapes. |
| Quickstart | `specs/001-public-release-hardening/quickstart.md` | 5-command flows. Consulted during T155 polish (smoke validation) and when authoring `docs/quickstart/`. |
| CLAUDE.md | repo root | Agent-specific guidance; references this document. Should not be edited by `/speckit.implement` except via explicit task. |

On first invocation, load them all into context. On subsequent
invocations (different user-story batch), reload `tasks.md` fresh
(other agents may have updated it) and treat the others as cached if
they haven't changed since last read.

---

## 2. Execution Scope per Invocation (DR-1 LOCKED)

**Rule**: one invocation = one **user-story phase**, not one task, not
one release-milestone, not the whole feature.

This is the DR-1 decision. It is non-negotiable for this spec.

### Why per-user-story

- Each user-story phase in `tasks.md` ends with an Independent-Test
  checkpoint — an inspection gate the maintainer can run manually.
- A single task is too small a unit: agent spin-up cost + context
  reload dominate. A full release-milestone is too large: can exceed
  context budget and undermines the checkpoint-per-phase discipline.
- Matches the spec's "Incremental Delivery" strategy verbatim.

### Invocation sequence for spec-01

| Order | Phase | Scope (task range) | Milestone |
|---|---|---|---|
| 1 | Setup | T001–T009 | prep for v0.2.0 |
| 2 | Foundational | T010–T015 | prep for v0.2.0 |
| 3 | US1 — Contract Integrity 🎯 MVP | T016–T030 | v0.2.0 |
| 4 | US4 — Repository Hygiene | T075–T084 | v0.2.0 |
| 5 | US2 — CI Pipeline (bootstrap) | T031–T048 (T049 deferred to invocation 10) | v0.2.0 |
| 6 | US3 — Documentation (seed) | T050–T052 only (T053–T074 deferred to invocation 13) | v0.2.0 |
| 7 | US5 — Algorithms & Benchmarks | T085–T109 + T105a | v0.3.0 |
| 8 | US6 — Observability | T110–T122 | v0.3.0 |
| 9 | US7 — Constraint Framework | T123–T134 | v0.3.0 |
| 10 | US2 — matrix completion | T049 | v0.3.0 |
| 11 | US8 — Release Engineering | T135–T145 | v1.0.0 |
| 12 | US9 — Interactive Demo | T146–T150 | v1.0.0 |
| 13 | US3 — Documentation (completion) | T053–T074 | v1.0.0 |
| 14 | Polish | T151–T160 + T157a | v1.0.0 |

14 invocations total. Maintainer triggers each; agent does NOT chain.

### What "one invocation" means mechanically

- Agent reads the scope's task range from `tasks.md`.
- Runs them in dependency order (§3).
- Commits per DR-2 policy (§4).
- At phase end: runs the Independent-Test from `spec.md` for that
  user story, reports pass/fail to the maintainer, exits.
- Does NOT auto-proceed to the next phase even if time remains.

---

## 3. Execution Policy

### 3.1 Test-first enforcement (NON-NEGOTIABLE — Constitution II)

For every task labelled as a **test task** (T010, T012, T016–T020, T031,
T032, T050, T051, T075, T076, T085–T091, T110–T113, T123–T125, T135,
T136, T146, plus T105a):

1. Author the test code per the task description.
2. Run `pytest <specific_test_file_or_node>`.
3. **CONFIRM the test FAILS** (red). If the test passes on first run,
   STOP — either the test is insufficiently strict, the implementation
   already exists (redundant task), or the import path is wrong.
   Surface the situation to the maintainer; do not commit a "test that
   never fails" (this is worse than no test).
4. Commit the failing test with message `test(scope): add <what> (T###)`.
5. Proceed to the paired implementation task(s).
6. After implementation: run the same test — **CONFIRM it now PASSES**.
7. Commit the implementation per DR-2 with body line `tests authored
   first (T### fails green at HEAD~1)`.

No implementation task may be committed without a preceding failing-test
commit in the current phase's commit log.

### 3.2 Parallelism for `[P]` tasks

`[P]`-tagged tasks touch disjoint files and have no dependency on
incomplete tasks. The agent MAY:

- edit multiple `[P]` files in one response (parallel Write / Edit
  tool calls),
- batch their commits into per-file commits OR one combined commit
  with a composite subject (`chore(config): add editorconfig,
  ruff.toml, manifest (T006, T007, T005)`).

Non-`[P]` tasks are strictly sequential.

### 3.3 Dependency respect

`tasks.md` §Dependencies is authoritative. Specifically:

- Setup BLOCKS Foundational BLOCKS every user story.
- Within a story: tests BEFORE implementation, models BEFORE services,
  services BEFORE CLI, CLI BEFORE integration tests.
- Cross-story deps from the spec: US5 depends on US1 registry, US8
  ships US1–US7 together.

If the agent spots an ordering error in `tasks.md` (e.g., T024 listed
before T021 on which it depends), it does NOT silently reorder — see
DR-7 (§7).

### 3.4 Progress-report format

At phase end, agent emits a compact report:

```
Phase: US1 — Contract Integrity (invocation 3 of 14)
Tasks: T016..T030 (15 tasks, 0 skipped, 0 deferred)
Commits: 12 on 001-public-release-hardening (see git log -12)
Tests: 47 pass, 0 fail, 0 skip (pytest tests/ -m "not slow")
Coverage: 91.4 % on src/bin_packer_3d/ (87 % lowest module: data/loaders.py)
Mypy: 0 errors (strict)
CHANGELOG: [Unreleased] updated with 4 bullets under "Changed"
Independent Test (US1): PASS — bin-packer info lists 2 registered algos;
  PackerConfig(strategy="bogus") fails with enumerated list;
  malformed CSV loads without KeyError.
Next recommended invocation: US4 (T075..T084)
```

No prose outside this shape unless the maintainer asks for it.

---

## 4. Commit Policy (DR-2 + DR-3 LOCKED)

### 4.1 Conventional Commits format

`type(scope): subject ≤ 72 chars`

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`,
`build`, `ci`, `revert` (Constitution §Commit style).

Scope is a top-level module or concern:

- `algorithms`, `config`, `cli`, `models`, `data`, `observability`,
  `constraints`, `benchmark`, `visualization`, `docs`, `ci`, `deps`,
  `release`, `tests`, `chore`, `hygiene` (the latter for DATASETS /
  CODE/ moves).

Imperative mood: "add X", "fix Y", "remove Z" — never "added", "fixes",
"removing".

### 4.2 Grouping rule (DR-2)

- **One commit per task**, OR
- **One commit per tight (test + implementation) pair** — e.g.,
  T019 + T025 as `feat(models)!: make Box.weight Optional[float]
  (T019, T025)` with the test-first body line.
- **Never** batch an entire phase into one commit — loses bisectability
  and breaks Constitution §Compliance review.

### 4.3 Body conventions

- Blank line after subject.
- Wrap body at 72 columns.
- Reference the task IDs this commit implements in the subject (as `(T###)`)
  OR in the body (as `Implements: T###, T###`).
- For test-first implementation commits: include the line
  `tests authored first (<test-commit-SHA> fails green at HEAD~1)`
  once the paired test commit exists.
- Breaking changes: `feat(config)!: ...` subject marker AND a
  `BREAKING CHANGE:` footer paragraph with migration note. Both.
  Example:

  ```
  feat(models)!: make Box.weight Optional[float] (T019, T025)

  Box.weight changes from `float` (default 0.0) to `Optional[float]`
  (default None) so callers can distinguish "unknown" from "known zero".

  tests authored first (abcd1234 fails green at HEAD~1)

  BREAKING CHANGE: Box() previously set weight=0.0 implicitly. Callers
  relying on that default must now pass weight=0.0 explicitly.
  Unchanged: Box(weight=0.0) semantics. See CHANGELOG [0.2.0] for
  migration note.
  ```

### 4.4 NO AI attribution (DR-3)

Constitution §Commit style forbids:

- `Co-Authored-By: Claude …`
- `🤖 Generated with Claude Code`
- any `Generated-By:`, `Assisted-By:`, `AI-`, trailer.

This is hard-enforced. The agent MUST NOT add any such line. The
agent-specific memory record for this project also documents this
preference. A commit with an AI-attribution trailer is a violation the
agent MUST fix via `git commit --amend` (or, if already pushed, via
follow-up correction commit — never rewrite published history without
maintainer approval).

### 4.5 Per-phase CHANGELOG update

At phase end, agent appends bullets under `CHANGELOG.md` `## [Unreleased]`
section (Keep-a-Changelog subsections: Added / Changed / Fixed / Removed /
Deprecated / Security). These bullets migrate to the versioned section
(`[0.2.0]`, `[0.3.0]`, `[1.0.0]`) at release-tag time — NOT by the agent,
by the maintainer running the release.

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

### 5.2 Post-phase gates

After the last task of a user-story phase (before the progress report):

| Gate | Command | Fail action |
|---|---|---|
| G-Full-Pytest | `pytest tests/ -m "not slow"` | STOP; all-suite failure is a show-stopper |
| G-Coverage-Overall | `pytest --cov=src/bin_packer_3d --cov-fail-under=90` (DR-5) | STOP; below-floor blocks phase completion |
| G-Coverage-Module | custom script: parse `coverage.xml`, flag any module < 70 % | WARNING only (DR-5); record in progress report but don't block |
| G-Mypy-Full | `mypy --strict src/bin_packer_3d/` | STOP |
| G-Changelog | grep `## \[Unreleased\]` in CHANGELOG.md has ≥ 1 new bullet since phase start | STOP (if user-visible change) |
| G-Independent-Test | manual run of the Independent-Test block from `spec.md` for the current US | STOP; phase is NOT done without this pass |

### 5.3 Pre-release gates

Before any `v*.*.*` tag is prepared (tag push itself is manual —
§8 hard stop):

| Gate | Command / Check | Fail action |
|---|---|---|
| G-Constitution-Audit | walk Principles I–VIII; produce `docs/compliance/v{X.Y.Z}-audit.md` | STOP on any NON-NEGOTIABLE fail |
| G-SC-006 | `pytest tests/integration/test_benchmark.py::test_sc006_delta_threshold` (v0.3.0+) | STOP (H1 finding, gates v0.3.0) |
| G-Privacy-Audit | `python scripts/audit_datasets.py` | STOP on hit |
| G-Docs-Build | `mkdocs build --strict` (v1.0.0) | STOP |
| G-Quickstart-Smoke | execute quickstart.md Scenario A on a clean venv (v1.0.0) | STOP |
| G-Tag-Changelog-Match | `pyproject.toml` version == `bin_packer_3d.__version__` == latest `CHANGELOG.md` `## [x.y.z]` header | STOP |

### 5.4 Gate failure philosophy

- **FAIL LOUD**. The agent never uses `pytest ... || true`, never sets
  `continue-on-error: true` in a CI job, never suppresses mypy via bare
  `# type: ignore`. Every suppression is targeted AND justified.
- Gates that are flagged as WARNINGs (per-module coverage, per-push
  benchmark regression) surface in the progress report and accrue in
  CHANGELOG `Unreleased` as tech debt.

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

Auto-fixable: run `ruff check --fix` + `ruff format`; re-verify; continue.
Not auto-fixable (rare): read the rule, apply manually, verify.

### 6.3 `mypy --strict` failure

Preferred: fix the type issue. Strict mode catches real bugs — treat
it as such, not as a nuisance.

Escape hatch (only when the alternative is a large unrelated refactor):

1. Add `# type: ignore[<exact-rule-name>]` — never bare `# type: ignore`.
2. Add adjacent TODO comment referencing the task ID:
   `# TODO(T###): <why> — revisit in <future spec / phase>`
3. Add CHANGELOG.md `[Unreleased]` entry under `### Security/Debt` (new
   subsection if needed) noting the tech debt:
   `- [debt] <module>:<line> narrow type-ignore pending <reason>`.

This trifecta is the ONLY way a `# type: ignore` enters the tree.

### 6.4 External failures

- GitHub API rate limit / transient 5xx: exponential backoff (2 s, 4 s,
  8 s); give up after 3 attempts; surface to maintainer.
- PyPI publish failure: do NOT retry without maintainer inspection —
  partial publishes can leave artefacts.
- `pip install` failure: inspect the error; if it's a missing system
  package, surface the exact OS command needed.

### 6.5 Unexpected repo state

Cases requiring STOP + maintainer handoff (no auto-resolution):

- Merge conflict on any file the current task touches.
- Uncommitted changes from a prior invocation that don't match the
  expected WIP pattern.
- Stash entries older than the session.
- Branch diverged from `main` in an unexpected way.
- Any reference to a spec ID other than `001-public-release-hardening`
  in a file the agent didn't author.

The agent describes the state, lists the three most-likely causes,
and asks the maintainer before touching anything.

---

## 7. Ambiguity Resolution

A task description might be ambiguous at execution time (e.g.,
"register existing FFDPacker" — does "register" include constructor
signature changes?). Decision tree:

**Step 1 — Consult design artefacts** (2-min budget):

- `plan.md` §Project Structure: file path authoritative.
- `research.md` ADR on the topic: decision authoritative.
- `data-model.md`: entity shape authoritative.
- `contracts/*.md`: public surface authoritative.

If a clear answer emerges, proceed with an inline one-line comment in
the commit body: `Resolved via ADR-0001 research.md`.

**Step 2 — Consult constitution** (30-sec budget):

Does a principle prescribe a default? Examples:

- Principle I (Contract Honesty) — any choice that would let a phantom
  surface leak is rejected.
- Principle V (Library Citizenship) — any choice introducing a root
  logger side-effect is rejected.
- Principle VIII (Performance Discipline) — any optimisation without a
  baseline is rejected.

If a principle applies, proceed with commit body citation:
`Resolved via Constitution §V (Library Citizenship)`.

**Step 3 — STOP and surface (high-stakes)**:

If the ambiguity affects a public surface, test contract, or commit
graph, STOP. Surface with:

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
- Log message wording (as long as event key matches the spec).
- Internal variable naming.
- Test-case parameter values (seeds, dimensions) if the test is
  property-based.
- File-order within a task's scope.

Document the default in the commit body: `Default: <what>`.

---

## 8. Hard Stopping Points

Tasks that the agent MUST NOT auto-execute. For each, the agent
prepares everything up to the stopping point, then hands off with a
clear request.

| Task | Reason for stop | What agent delivers | What maintainer must do |
|---|---|---|---|
| T040 | Branch protection is a GitHub UI action | `docs/maintainers.md` with the required-check list | Configure via repo Settings → Branches |
| T077, T078, T079 | `git filter-repo` rewrites history irreversibly | `DATASETS/AUDIT.md` with per-file disposition; staged pre-rewrite branch | Review AUDIT.md; run `git filter-repo` manually; force-push replacement branch |
| T081 | `git mv CODE legacy` + history tag is destructive on main | Tag `legacy-code-preserved` ready on pre-move commit; move diff ready | Confirm tag; apply move |
| T137, T139 | First Docker image build + GHCR push | Dockerfile + release.yml changes ready; `.dockerignore` | Maintainer inspects before tag |
| T140 | OIDC trusted publisher dashboard action | `docs/maintainers.md` with exact config values | Log into pypi.org; configure publisher |
| T141 | PyPI name reservation (name may be taken) | Draft `pyproject.toml` with `name = "bin-packer-3d"` and fallbacks listed | Verify availability manually; pick name; bless |
| T158 | `git tag v1.0.0` is a release gate | All pre-release gates green per §5.3 | Run `git tag v1.0.0 -m "..."`; `git push origin v1.0.0` |

Additionally, the agent NEVER performs the following without explicit
per-action maintainer consent, regardless of task:

- Force-push to any remote branch (`git push --force[-with-lease]`).
- `git reset --hard` on a branch that has remote commits.
- `git branch -D` on any branch with unmerged work.
- `git filter-repo` anywhere.
- `git rebase -i` on published commits.
- `git commit --amend` on a commit that has been pushed.
- `rm -rf` on anything outside `dist/`, `build/`, `.pytest_cache/`,
  `__pycache__`, `.mypy_cache/`, `.ruff_cache/`, `htmlcov/`.
- Any GitHub API mutation outside issue comments (no PR creation /
  merging / closing without maintainer say-so).

---

## 9. Memory & Telemetry

The project uses Engram for cross-session memory. Protocol:

### 9.1 After each task batch (or tight group)

Call `mem_save` with:

- **title**: verb + object, e.g., "Registered FFD and Shelf algorithms
  (T021–T023)".
- **type**: `decision`, `bugfix`, `architecture`, `pattern`, `config`,
  `discovery`, `learning`.
- **topic_key**: stable key like
  `3d-bin-packing/spec-01/us1/algorithm-registry`.
- **content** in the 4-block format (`**What**`, `**Why**`, `**Where**`,
  `**Learned**`).
- **scope**: `project`.

### 9.2 At phase close (before exiting)

Call `mem_session_summary` per the §Protocol in the project CLAUDE.md /
global Engram instructions. Use the Goal / Instructions / Discoveries /
Accomplished / Next Steps / Relevant Files shape.

### 9.3 Topic keys for spec-01

Reserve these prefixes — do not drift:

- `3d-bin-packing/spec-01/setup/*`
- `3d-bin-packing/spec-01/foundational/*`
- `3d-bin-packing/spec-01/us1/*` through `3d-bin-packing/spec-01/us9/*`
- `3d-bin-packing/spec-01/phase-A|B|C/*` (cross-cutting per release
  milestone)
- `3d-bin-packing/spec-01/polish/*`
- `3d-bin-packing/spec-01/release/v{X.Y.Z}/*`

### 9.4 Do NOT save

- Trivial "task T0XX completed" pings (already in git log).
- Content copied from the spec / plan / constitution (already on disk).
- Code snippets the agent just wrote (already committed).

DO save: non-obvious decisions, gotchas discovered, deviations from
plan, anything that saves a future session a re-investigation.

---

## 10. Environment Setup

Before the first task of any invocation, the agent runs a pre-flight
check and reports status. If any check fails: STOP and surface.

| Check | Command | Pass condition |
|---|---|---|
| Python version | `python --version` | `3.11`, `3.12`, `3.13`, or `3.14` |
| Package install | `pip show bin_packer_3d` | editable install at repo path |
| Dev extras present | `python -c "import pytest, ruff, mypy, hypothesis"` | all importable |
| Pre-commit active | `pre-commit --version` + `.git/hooks/pre-commit` exists | both yes |
| Current branch | `git branch --show-current` | `001-public-release-hardening` |
| Working-tree state | `git status --porcelain` | empty OR matches expected WIP from prior invocation's progress report |
| Remote configured | `git remote -v` | `origin` points at Bruno-Ghiberto/3D_BIN_PACKING |
| Tags reachable | `git fetch --tags` | exit 0 |

If the maintainer starts an invocation from a cold start (new machine,
new clone), `CONTRIBUTING.md` (authored in T066) contains the full
setup path — the agent may point the maintainer at it rather than
running setup itself.

---

## 11. Completion & Handoff

### 11.1 "Done" per user-story phase

All of:

1. Every task in the phase's section has its checkbox flipped to `[x]`
   in `tasks.md` (agent edits this at commit time).
2. All post-phase gates green (§5.2).
3. Spec's Independent-Test block for that US passes on demand.
4. CHANGELOG.md `[Unreleased]` updated.
5. Any ADR referenced in the phase has its published form at
   `docs/adr/NNNN-<slug>.md` (if Phase C doc deliverables touched it).
6. Phase-close progress report emitted (§3.4) + `mem_session_summary`
   saved (§9.2).

### 11.2 "Done" per release milestone

All user-story phases mapped to that milestone green (see §2 table).
Plus:

- All pre-release gates green (§5.3).
- `/speckit.analyze` re-run with zero new findings at CRITICAL or HIGH.
- Release tag prepared (not pushed — §8).
- Release notes (`CHANGELOG.md`) migrated from `[Unreleased]` to
  `[{X.Y.Z}]` (done by maintainer at tag time, not by agent).

### 11.3 When to re-run `/speckit.tasks`

If during implementation the agent identifies a task that is MISSING
from `tasks.md` (not a task-reorder issue — a genuine gap), it STOPS
and surfaces. The maintainer decides whether to:

- re-run `/speckit.tasks` to regenerate (tasks.md rewritten end-to-end),
  OR
- amend `tasks.md` inline with the new task ID (e.g., `T105b`).

The agent does NOT extend `tasks.md` on its own.

### 11.4 When to re-run `/speckit.analyze`

Required at every release-milestone boundary (before preparing a tag).
Optional at every user-story-phase boundary. If any spec/plan
amendment occurs mid-implementation, MANDATORY.

### 11.5 Handback message shape at invocation end

```
Invocation complete — <Phase name> (N of 14).
Result: <PASS | PARTIAL | FAIL>
Commits: M on 001-public-release-hardening
Tests: passed / failed
Coverage: X %
Independent Test: <PASS | FAIL — <reason>>
Next recommended invocation: <Phase name>
Open questions for maintainer (if any): <list>
Hard stops encountered: <list with task IDs, or none>
```

---

## 12. Open Questions

At authoring time of THIS prompt — none.

If the maintainer invokes `/speckit.implement` with open questions
still listed in this section, the agent MUST refuse to proceed and
request resolution first. Keeping this section empty is itself a
gate.

### Future-proofing

If a later amendment to the spec or constitution introduces a new
concern that this prompt doesn't cover, the concern lands here until
the prompt itself is amended via a new design cycle
(`/sc:design` → review → write).

Examples of what WOULD belong here if it arose (none do today):

- "Does the release workflow auto-create GitHub Releases, or wait for
  manual `gh release create`?"
- "Should the benchmark CI job fail-hard on regression, or only warn?"
- "What happens if a non-NON-NEGOTIABLE gate fails — deferred fix
  allowed, or hard stop?"

---

## Locked decisions summary

| DR | Topic | Locked value |
|---|---|---|
| DR-1 | Execution scope per invocation | Per user-story phase (14 invocations for spec-01) |
| DR-2 | Commit grouping | One commit per task OR tight test+impl pair; never whole-phase |
| DR-3 | AI attribution on commits | NONE — no Co-Authored-By, no 🤖 trailer |
| DR-4 | Interactive vs unattended | Interactive — maintainer triggers each invocation |
| DR-5 | Coverage gate | Overall ≥ 90 % blocking; per-module ≥ 70 % warning-only |
| DR-6 | Branch & PR workflow | Direct commits on feature branch; single PR to main per release milestone |
| DR-7 | Task re-ordering authority | None — agent surfaces ordering concerns, never reorders silently |
| DR-8 | Mid-invocation skill use | Read-only only (`/sc:explain`, `/sc:analyze`); no scope-changing skills |

These are not suggestions. The agent executing `/speckit.implement`
against spec-01 MUST obey all eight.
