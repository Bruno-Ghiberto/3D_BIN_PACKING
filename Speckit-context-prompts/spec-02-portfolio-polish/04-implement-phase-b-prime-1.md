# Implement Context — Phase B'-1: US2 + US1 Bundle for spec-02 Portfolio Polish of `bin-packer-3d`

> Hand this document to `/speckit-implement` as the operational prompt
> for the SECOND invocation of spec-02. It does NOT say WHAT to do —
> that's `tasks.md` — and it does NOT say WHY — that's the
> constitution + plan + spec. It says **HOW** to execute the 17
> Phase B'-1 tasks safely, reproducibly, and within the constitution's
> NON-NEGOTIABLE principles.
>
> **Input spec**: `specs/002-portfolio-polish/spec.md`
> (7 USs, 38 FRs, 14 SCs; US6 Deferred).
> **Plan**: `specs/002-portfolio-polish/plan.md`
> (13 sections, 8 constitutional gates PASS).
> **Tasks**: `specs/002-portfolio-polish/tasks.md`
> (92 tasks T001..T092, 9 phases; T001..T024 closed in Phase A).
> **This invocation's scope**: Phase B'-1 — US2 Polished Visualisations
> + US1 First-Screen README Credibility (T025..T041, 17 tasks).
> **Constitution**: `.specify/memory/constitution.md` v1.0.1
> (8 principles, 4 NON-NEGOTIABLE).
> **Phase A retrospective**: see Engram observation
> `3d-bin-packing/spec-02/phase-a/closed` + session memory
> `.serena/memories/sessions/2026-05-14-spec-02-phase-a-implementation.md`.

---

## 0. Brief

This document is the execution policy that `/speckit-implement` MUST
follow when processing the Phase B'-1 range of `tasks.md`. It codifies:

- how Phase B'-1's 17 tasks are sequenced across **two user stories**
  with a non-obvious cross-US dependency (US2's Kaleido static-export
  branch unblocks US1's gallery PNGs),
- how test-first discipline is enforced mechanically for the four
  test tasks in Phase B'-1 (T025, T026, T027, T035),
- how commits are structured into **two chained PRs** (US2 first, US1
  second) per DR-10 below,
- what blocks a task from being marked complete,
- what halts the run and hands control back to the maintainer — with
  one task-specific hard stop at T030 (vhs hero-GIF recording requires
  a TTY on the maintainer's Linux machine),
- the cross-track hygiene policy from Phase A still in force, plus a
  forward-looking note on registry-driven gallery generation if US5
  Part 2 has merged to `develop` mid-invocation.

Without this prompt, `/speckit-implement` would invent execution policy
on the fly every run. That is unacceptable under a NON-NEGOTIABLE
test-first regime. Everything below is a RULE, not a suggestion.
Violation is a merge blocker for the implementing agent's work.

**Scope boundary**: this prompt applies to spec-02 **Phase B'-1 only**
(T025..T041 = US2 7 tasks + US1 10 tasks). Phase B'-2 (US3 docs site,
T042..T058), Phase B'-3 (US4 repo surface, T059..T066), Phase B''-1
(US5 demo, T067..T074), Phase B''-2 (US7 CV identity, T075..T079), and
Phase C (release ceremony, T080..T092) each get their own implement
context. The invocation closes only when **both** Independent-Test
gates (§5.2) are demonstrated — first US2's, then US1's.

---

## 1. Primary Input References

The `/speckit-implement` invocation MUST read these inputs and keep
pointers to them for the duration of the run.

| Artefact | Path | Role |
|---|---|---|
| Task list | `specs/002-portfolio-polish/tasks.md` | **Source of truth for WHAT**. Task IDs, `[P]` markers, `[USn]` labels, file paths. Phase B'-1 executes T025..T041 in the **execution order** of §3.3 (US2 implementation first, then US1), NOT in numeric task-ID order. |
| Implementation plan | `specs/002-portfolio-polish/plan.md` | Technical context (Python 3.11, deps, structure), 8 constitutional gates, three-phase delivery strategy. Consulted during ambiguity resolution (§7). |
| Spec | `specs/002-portfolio-polish/spec.md` | 7 user stories, 38 FRs, 14 SCs, Independent-Test blocks. Phase B'-1 binds US1 (FR-001..FR-014, FR-036) and US2 (FR-005..FR-009, FR-037, FR-038). |
| Constitution | `.specify/memory/constitution.md` (v1.0.1) | The 8 principles, 4 NON-NEGOTIABLE. Overrides any conflicting guidance. Consulted at every gate check (§5) and every ambiguity (§7 step 2). |
| Research / ADRs | `specs/002-portfolio-polish/research.md` | 12 ACCEPTED ADRs. Phase B'-1-critical: ADR-001 (Plotly+Kaleido pin), ADR-002 (mkdocs-material defaults — referenced for accessibility parity), ADR-004 (regenerated-section markers), ADR-005 (Highlights regeneration source), ADR-006 (theme), ADR-007 (palette), ADR-008 (demo command CLI shape — referenced because the `pack --static-format` flag from T039 mirrors `demo --static-format` from US5), ADR-011 (stats overlay), ADR-012 (colourblind verification). |
| Data model | `specs/002-portfolio-polish/data-model.md` | Three new entities (`VisualisationStyle`, `DemoArtifact`, `HeadlineDatasetSeed`) plus augmented `Placement.colour`. Phase B'-1 consumes `Placement.colour` in T037, references `VisualisationStyle.stats_overlay_layout` in T037, and previews `DemoArtifact` shape (full implementation lands in Phase B''-1 US5). |
| Contracts | `specs/002-portfolio-polish/contracts/*.md` | API surface for the polish phase. Phase B'-1 consumers: `visualisation-theme.md` (T037, T038, T039), `algorithm-card-source.md` (T028 regenerator's table block), `repository-structure.md` (T028 regenerator's Project Structure block — note: full enforcement lands in Phase B'-3 US4 T059). |
| Quickstart | `specs/002-portfolio-polish/quickstart.md` | 5-command reviewer flow. Phase B'-1 ships TWO of the five commands: `pack --visualize` (via US2 T037..T040) and the README hero/visual references in `bin-packer info` output is **out of scope** here. The `demo` command lands in Phase B''-1. |
| US5 cross-track contract (READ-ONLY) | `specs/001-public-release-hardening/spec.md § US5` + `specs/001-public-release-hardening/tasks.md § T085..T160` | Reference only — describes the parallel US5 algorithm track. Consulted to enforce the no-touch list (§9). Phase B'-1 has a NEW concern over Phase A: T033 invokes `bin-packer pack` against every key in `ALGORITHMS` registry; if US5 Part 2 has merged, the new packers MUST be picked up via the registry, not hardcoded (FR-004). |
| Plan context prompt | `Speckit-context-prompts/spec-02-portfolio-polish/02-plan.md` § 11 | Authoritative source for the US5 cross-track hygiene rules referenced in §9. |
| Phase A retrospective | Engram topic `3d-bin-packing/spec-02/phase-a/closed` + `.serena/memories/sessions/2026-05-14-spec-02-phase-a-implementation.md` | Phase A discoveries that bind Phase B'-1 — see §11.1 mandatory pre-flight read. Critical: discovery #7 (Kaleido 1.x ↔ Plotly 5.x incompatibility) drives DR-11 below. |
| CLAUDE.md | repo root | Project-level agent guidance. Should not be edited by `/speckit-implement` except via explicit task (none in Phase B'-1). |

On invocation start, load them all into context. Phase A retrospective
is mandatory — it contains the Kaleido failure mode that DR-11 below
neutralises before T035 ships.

---

## 2. Execution Scope per Invocation (DR-1 INHERITED FROM PHASE A)

**Rule**: one invocation = one **phase/bundle**, not one task, not
one release-milestone, not the whole feature.

This is the DR-1 decision for spec-02, inherited unchanged from
Phase A (`03-implement-phase-a.md` § 2). Phase B'-1 is the second of
seven invocations.

### Why per-phase/bundle (Phase B'-1 specifics)

- Phase B'-1 bundles **two user stories** (US2 + US1) because they
  share a hard cross-story dependency: US1's T033 (gallery PNG
  generation for the README) consumes US2's T037+T038 (theme-applied
  plotter + Kaleido static-export branch). Splitting them across two
  invocations would force a temporary-placeholder commit in US1's PR,
  which is wasteful churn.
- Each user story has its own Independent-Test from `spec.md`. The
  bundle closes only when **both** gates pass (§5.2 below).
- The chained-PR strategy (DR-10 below) gives review focus per US
  while keeping the bundle as one logical invocation.

### Invocation sequence for spec-02

| Order | Phase / Bundle | Task range | Milestone | Status |
|---|---|---|---|---|
| 1 | Phase A — Scaffolding | T001..T024 (24 tasks) | unblocks all USs | ✅ CLOSED |
| 2 | **Phase B'-1 — US2 + US1 (this invocation)** | T025..T041 (17 tasks) | v0.3.0-rc1 P1 | ▶ THIS RUN |
| 3 | Phase B'-2 — US3 (docs site) | T042..T058 (17 tasks) | v0.3.0-rc1 P1 | pending |
| 4 | Phase B'-3 — US4 (repo surface) | T059..T066 (8 tasks) | v0.3.0-rc1 P1 | pending |
| 5 | Phase B''-1 — US5 (demo command) | T067..T074 (8 tasks) | v0.3.0-rc1 P2 | pending |
| 6 | Phase B''-2 — US7 (CV identity) | T075..T079 (5 tasks) | v0.3.0-rc1 P2 | pending |
| 7 | Phase C — Polish & Release | T080..T092 (13 tasks) | v0.3.0-rc1 tag | pending |

Seven invocations total. Maintainer triggers each via a new
implement-context prompt; the agent does NOT chain phases on its own.

### What "one invocation" means mechanically for Phase B'-1

- Agent reads `tasks.md` T025..T041 and the inputs in §1.
- Reads the Phase A retrospective (mandatory; §11.1).
- Lands the DR-11 Kaleido pin precursor (one-line edit to
  `pyproject.toml` `viz` extras) as the FIRST commit of the
  invocation, BEFORE T035 ships. See §11.2.
- Runs **US2 first** (T035..T041) in dependency order with
  `[P]`-parallelism (§3.2), then US1 (T025..T034). See §3.3.
- Commits per DR-2 policy (§4).
- Halts at T030 for maintainer-side vhs recording (§8.1). Resumes at
  T031 after maintainer commits `docs/assets/hero.gif`.
- At each US end: runs the Independent-Test gate (§5.2), reports
  result, **does not auto-open the PR** for that US's chain.
- At phase end: emits the progress report (§3.4) and the handback
  message (§12.5), exits.
- Does NOT auto-proceed to Phase B'-2 even if context budget remains.

---

## 3. Execution Policy

### 3.1 Test-first enforcement (NON-NEGOTIABLE — Constitution II)

Strict TDD mode is enabled (per project `CLAUDE.md`). For every task
labelled as a **test task** in Phase B'-1:

| Test task | Test file | Implementation pair(s) | Notes |
|---|---|---|---|
| T025 | `tests/integration/test_readme_drift.py` | T028 (`scripts/regenerate_readme.py`) + T031 (README rewrite) + T032 (run regenerator) | Drift test — green only once the FULL US1 chain lands |
| T026 | `tests/integration/test_readme_alt_text.py` | T031 (README rewrite with alt text on every `![]()`) | Verifies FR-036 |
| T027 | `tests/integration/test_highlights_drift.py` | T028 (regenerator's Highlights section block) + T032 | Verifies ADR-005 — Highlights sourced from live state |
| T035 | `tests/integration/test_visualisation_e2e.py` | T037 (theme+colour+overlay) + T038 (Kaleido static export) + T040 (verify green) | Byte-identical HTML + pixel-equal PNG across two runs |

**T036 is NOT a TDD test** — it is a snapshot-fixture *capture* task
that runs the implementation against `examples/headline.csv` and
commits the canonical expected outputs. It cannot be authored red.
Sequence: T035 red → T037+T038 green → T036 captures fixtures → T040
re-verifies against the committed fixtures.

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
commit earlier in this Phase B'-1 commit log.

**Note on the US1 drift-test chain**: T025, T026, T027 are red-committed
BEFORE T028 ships. T028 (regenerator script) does NOT turn them green
on its own — T031 (the README rewrite with the regenerated-section
markers and alt text) plus T032 (running the regenerator to populate
the blocks) is what turns T025 and T027 green. T026 turns green at
T031. The implementation-commit body for T031 references the SHAs of
all three red commits.

### 3.2 Parallelism for `[P]` tasks

`[P]`-tagged tasks touch disjoint files and have no dependency on
incomplete tasks. Phase B'-1 `[P]` islands:

- **Island A (US2 red test)**: T035 is the only test in US2; not
  parallel with anything in US2's own chain.
- **Island B (US1 red tests)**: T025, T026, T027 are all `[P]` — three
  different test files, no inter-dependencies. May be authored +
  committed in any order so long as each is verified red on its own
  before commit. A single composite commit is acceptable:
  `test(docs): add README drift + alt-text + highlights drift tests
  (T025, T026, T027)`.
- **Island C (US1 curated assets)**: T029 (vhs Tape script) is `[P]`
  with respect to T028 (regenerator) — different file, no shared
  surface. They can be authored side-by-side, but T029 lands BEFORE
  the T030 hard stop.

Non-`[P]` tasks (T028, T030, T031, T032, T033, T034, T036, T037, T038,
T039, T040, T041) are strictly sequential and must respect §3.3
dependency edges.

### 3.3 Dependency respect

`tasks.md` Phase 3 + Phase 4 sections are authoritative. **Execution
order for Phase B'-1 follows the recommended ordering in spec-02
(US2 first), NOT numeric task-ID order.** This is the DR-12 locked
decision (see §13 Locked Decisions Summary).

Intra-phase dependency edges, in execution order:

```
[DR-11 precursor commit: kaleido<1.0.0 pin in pyproject.toml viz extras]
      ↓
==== US2 chain (Chain 1 — opens PR-A) ====
T035 (red: test_visualisation_e2e.py)
      ↓
T037 (impl: plotter.py — apply_theme + Placement.colour + stats overlay)
      ↓
T038 (impl: plotter.py — Kaleido static-export branch, ImportError fallback)
      ↓
T039 (impl: cli.py — --static-format option on `bin-packer pack`)
      ↓
T036 (snapshot capture: run pack, commit tests/fixtures/expected/bfd/*)
      ↓
T040 (verify: pytest test_visualisation_e2e.py green)
      ↓
T041 (CHANGELOG bullet for viz)
      ↓
[US2 Independent-Test gate §5.2.1]
      ↓
[MAINTAINER MERGES PR-A to develop / OR keeps it on feature branch
 per DR-10 — agent does not gate on this; continues to US1 chain]
      ↓
==== US1 chain (Chain 2 — opens PR-B) ====
T025, T026, T027 ([P] red tests — committed in one composite OR three
                 separate commits)
      ↓
T028 (impl: scripts/regenerate_readme.py — three generators + --check flag)
      ↓
T029 (impl: scripts/render_demo_gif.tape — vhs Tape script)
      ↓
[HARD STOP §8.1 — T030 maintainer-side vhs recording on Linux TTY]
      ↓
[MAINTAINER commits docs/assets/hero.gif from local vhs invocation]
      ↓
T031 (impl: README.md rewrite — hero, alt text, markers, gallery placeholders)
      ↓
T032 (run regenerator: python scripts/regenerate_readme.py;
      commit populated README)
      ↓
T033 (gallery PNG generation: invoke `bin-packer pack --visualize`
      against examples/headline.csv per ALGORITHMS key;
      move bin_1.png to docs/assets/gallery/<key>.png;
      depends on US2 T037+T038 being committed)
      ↓
T034 (verify: pytest test_readme_drift.py test_readme_alt_text.py
      test_highlights_drift.py — turns T025, T026, T027 green)
      ↓
[US1 Independent-Test gate §5.2.2]
```

The cross-US dependency T037+T038 → T033 is the hard edge that
mandates US2-first ordering. If for any reason the agent must run US1
first (e.g., DR-12 is overridden by the maintainer), T033 needs a
temporary placeholder PNG commit + a follow-up replacement commit
after T038 ships. This is explicitly NOT the path Phase B'-1 takes
under DR-12.

If the agent spots an ordering error in `tasks.md` (e.g., a `[P]`
marker on a task that actually has a dependency), it does NOT silently
reorder — see DR-7 (§7).

### 3.4 Progress-report format

At phase end, agent emits a compact report:

```
Phase: B'-1 — US2 + US1 (invocation 2 of 7)
Tasks: T025..T041 (17 tasks, 0 skipped, 0 deferred)
Precursor: DR-11 kaleido<1.0.0 pin landed (commit <SHA>)
Commits: M on feature/002-portfolio-polish across two chains:
  Chain 1 (US2, T035..T041): K commits, PR-A at <URL or "not opened">
  Chain 2 (US1, T025..T034): L commits, PR-B at <URL or "not opened">
Tests: 118 baseline (post-Phase-A) + N new = total / 0 fail
Coverage: X.X % on src/bin_packer_3d/ (Y % lowest module: <module>)
Mypy: 0 errors (strict)
CHANGELOG: [Unreleased] updated with K bullets under Added
US2 Independent-Test (§5.2.1): PASS — byte-identical HTML + pixel-equal PNG
US1 Independent-Test (§5.2.2): PASS — README first-screen credible
  - hero.gif renders on GitHub: OK (manual maintainer verification at T030)
  - regenerated sections drift-clean: OK
  - alt text on every image: OK
  - gallery PNGs match registry: OK
Hard stops encountered:
  - T030 vhs recording — resolved by maintainer at <commit SHA>
Cross-track events: <none | rebased on develop @ <sha> | ...>
Next recommended invocation: Phase B'-2 (US3 docs site, T042..T058)
```

No prose outside this shape unless the maintainer asks for it.

---

## 4. Commit Policy (DR-2 + DR-3 INHERITED FROM PHASE A)

### 4.1 Conventional Commits format

`type(scope): subject ≤ 72 chars`

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`,
`build`, `ci`, `revert` (Constitution §Commit style).

**Phase B'-1 scope vocabulary** (use these or surface a new scope to
the maintainer):

| Scope | Used for | Phase B'-1 tasks |
|---|---|---|
| `deps` | DR-11 kaleido pin in pyproject.toml | precursor commit |
| `viz` | plotter.py + visualisation pipeline changes | T037, T038 |
| `cli` | bin-packer CLI extensions | T039 |
| `test` | test-only commits (red authoring) | T025, T026, T027, T035 |
| `tests` | snapshot fixture capture (T036) | T036 |
| `docs` | README + vhs tape + regenerator | T028, T029, T031, T032 |
| `assets` | committed binary artefacts (hero.gif, gallery PNGs) | T030 (maintainer), T033 |
| `chore` | non-functional (CHANGELOG, etc.) | T041 |
| `verify` | wrap-up verification runs | T034, T040 (may be a no-op commit OR fold into the prior impl commit's body — DR-2 §4.2) |

Imperative mood: "add X", "fix Y", "remove Z" — never "added", "fixes",
"removing".

### 4.2 Grouping rule (DR-2)

- **One commit per task**, OR
- **One commit per tight (test + implementation) pair** — e.g.,
  T035 + T037 as `feat(viz): apply branded theme to plotter + add
  stats overlay (T035, T037)` with the test-first body line
  referencing the red-commit SHA. **Caveat for Phase B'-1**: T035's
  green state depends on BOTH T037 and T038, so the cleanest pair is
  T035-red as its own commit, then T037 and T038 as separate impl
  commits each citing T035-red.
- **Composite commits allowed for tight `[P]` islands** — e.g.,
  T025+T026+T027 as one
  `test(docs): add README drift + alt-text + highlights drift tests
  (T025, T026, T027)`.
- **Verify-only tasks (T034, T040)**: prefer to fold the verify into
  the body of the last impl commit (`Verifies: T034 — pytest <files>
  exits 0`) rather than create an empty `chore: verify ...` commit.
  A separate verify commit is acceptable only if the verify run
  produces a committed artefact (it does not, in Phase B'-1).
- **Never** batch the entire Phase B'-1 into one commit — loses
  bisectability and breaks Constitution §Compliance review. The
  chained-PR strategy (DR-10) makes this physically impossible
  anyway: two PRs means at least two merge points.

Target: 14–20 commits across Phase B'-1, split as roughly Chain 1
(US2) = 6–9 commits and Chain 2 (US1) = 8–11 commits, plus the DR-11
precursor.

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
- For T036 (snapshot fixture commits): include `generated:
  deterministic via bin-packer pack examples/headline.csv --strategy
  bfd --visualize -o tests/fixtures/expected/bfd/ @ <plotter-SHA>`
  for auditability — the next regeneration must produce byte-identical
  files. If a future intentional theme change invalidates the
  snapshot, the maintenance procedure is documented in `docs/maintainers.md`
  § Snapshot maintenance (authored in T053, Phase B'-2; if that section
  is not yet present, the agent leaves a TODO line in the commit body).
- For T033 (gallery PNG commits): include `generated: deterministic via
  bin-packer pack examples/headline.csv --strategy <key> --visualize @
  <plotter-SHA>` for each algorithm key present in `ALGORITHMS` at
  generation time.
- Breaking changes: NOT expected in Phase B'-1. The viz pipeline gains
  a new `--static-format` CLI flag (additive) and a stats-overlay
  annotation (additive). Plotter `Figure` return shape is unchanged.
  If a breaking change is unavoidable, STOP and surface — the plan
  does not authorise breaking changes in Phase B'-1.

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

At Phase B'-1 close, agent confirms `CHANGELOG.md` `## [Unreleased]`
section gains at least TWO new bullets under `### Added`:

- viz: branded plotter — apply_theme, Placement.colour binding, stats
  overlay, Kaleido static export, `--static-format` CLI flag (US2).
- docs: README rewrite — hero GIF, regenerated sections (algorithms
  table, highlights, project structure), comparison gallery, alt text
  on every image (US1).

Bullets are added inline by the impl commits (T041 for viz; a new
inline bullet from T031 or T032 for docs — Phase B'-1 does NOT
introduce a separate "docs CHANGELOG" task because the docs work in
US1 is large enough that the maintainer wants visibility on the
bullet text). If T031 / T032 do not naturally include the CHANGELOG
edit, the agent adds it as part of the T032 commit body's edits.

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
| G-Pre-commit | `pre-commit run --files <touched>` (mirrors the CI parity job from spec-01) | Fix per hook output; never `--no-verify` |

Green on all five = task is committable.

### 5.2 Post-US Independent-Test gates (REPLACES Phase A Foundation Ready)

Phase B'-1 has **two** Independent-Test gates, one per user story,
fired at the close of each chain. The bundle is complete only when
BOTH pass.

#### 5.2.1 US2 Independent-Test gate

Fires after T041 commits and before Chain-1 PR-A is opened (or
declared ready for opening). Source: tasks.md § Phase 4 Independent
Test block, paraphrased here for clarity.

| Gate | Command | Fail action |
|---|---|---|
| G-US2-IT-1 | Two consecutive `bin-packer pack examples/headline.csv --strategy bfd --visualize -o /tmp/run1` and `... -o /tmp/run2` runs produce byte-identical `bin_1.html` (`cmp /tmp/run1/bin_1.html /tmp/run2/bin_1.html` exits 0) | STOP — non-determinism is a release-blocker per FR-006 and SC-005 |
| G-US2-IT-2 | Same two runs produce pixel-equal `bin_1.png` (`sha256sum /tmp/run1/bin_1.png /tmp/run2/bin_1.png` matches) | STOP — same rationale |
| G-US2-IT-3 | `bin_1.html` opened in a headless inspection (`python -c "import bs4; print(bs4.BeautifulSoup(open('/tmp/run1/bin_1.html').read(), 'html.parser').title.text)"`) reveals the branded title format and at least one HTML element matching `class*="stats-overlay"` OR an `Annotation` block in the Plotly JSON containing algorithm/boxes/utilisation/runtime keys | STOP — overlay missing |
| G-US2-IT-4 | `python -c "from bin_packer_3d.visualization.plotter import <emit_static_function>; ..."` produces a PNG when `kaleido` is installed; ImportError with the documented message when uninstalled (verify via `pip uninstall kaleido -y` in a throwaway venv) | STOP — FR-009 unmet |
| G-US2-IT-5 | `bin-packer pack ... --static-format svg` produces a `.svg` (not `.png`) in the output directory | STOP — T039 incomplete |
| G-US2-IT-6 | `pytest tests/integration/test_visualisation_e2e.py -v` exits 0 with all assertions passing | STOP |

PASS = open PR-A (DR-10 chain 1). FAIL = STOP, fix, retry.

#### 5.2.2 US1 Independent-Test gate

Fires after T034 commits and before Chain-2 PR-B is opened. Source:
tasks.md § Phase 3 Independent Test block.

| Gate | Command | Fail action |
|---|---|---|
| G-US1-IT-1 | `pytest tests/integration/test_readme_drift.py -v` exits 0 (regenerator output matches committed README byte-for-byte) | STOP |
| G-US1-IT-2 | `pytest tests/integration/test_readme_alt_text.py -v` exits 0 (every image has non-empty alt text, FR-036) | STOP |
| G-US1-IT-3 | `pytest tests/integration/test_highlights_drift.py -v` exits 0 (regenerated Highlights matches live state per ADR-005) | STOP |
| G-US1-IT-4 | `python scripts/regenerate_readme.py --check` exits 0 (idempotent regeneration — equivalent to G-US1-IT-1 but via the script's own self-check path) | STOP |
| G-US1-IT-5 | `ls docs/assets/hero.gif docs/assets/gallery/*.png` lists every expected asset; for each algorithm key in `ALGORITHMS`, a matching `docs/assets/gallery/<key>.png` exists | STOP — gallery incomplete (cross-check with §9.4) |
| G-US1-IT-6 (manual) | Maintainer opens `https://github.com/Bruno-Ghiberto/3D_BIN_PACKING/blob/feature/002-portfolio-polish/README.md` in a desktop browser, scrolls the first viewport-and-a-half, and confirms within 60 seconds: (a) problem domain visible, (b) the three (or more, if US5 Part 2 merged) algorithm names visible, (c) install command visible, (d) run command visible, (e) at least one packed-bin visual rendered (hero.gif OR a gallery PNG) | WARN — surface to maintainer; record outcome in handback message §12.5 |

The G-US1-IT-6 manual gate is the only Phase B'-1 gate the agent
cannot self-verify. The agent renders a CHECKLIST in the handback
message and asks the maintainer to tick each item. PASS on the five
automated gates is sufficient to declare the gate "agent-green"; the
manual sub-criterion is annotated `pending maintainer verification`
in the progress report (§3.4).

PASS = open PR-B (DR-10 chain 2). FAIL on any automated gate = STOP,
fix, retry.

### 5.3 Post-phase gates (cross-cutting)

Fired ONCE at the end of Phase B'-1 (after both Independent-Test
gates pass):

| Gate | Command | Fail action |
|---|---|---|
| G-Full-Pytest | `pytest tests/ -m "not slow"` | STOP — all-suite failure is a show-stopper |
| G-Baseline | total passing ≥118 (post-Phase-A baseline) + N new = full suite | STOP |
| G-Coverage-Overall | `pytest --cov=src/bin_packer_3d --cov-fail-under=90` (DR-5) | STOP — below-floor blocks phase completion |
| G-Coverage-Module | parse `coverage.xml`, flag any module < 70 % | WARNING — record in progress report, don't block |
| G-Mypy-Full | `mypy --strict src/bin_packer_3d/` | STOP |
| G-Changelog | `## [Unreleased]` has ≥2 new bullets under `### Added` per §4.5 | STOP |
| G-Slow-Install-Footprint | `pytest tests/integration/test_install_footprint.py --no-header -q` (still passes the Phase A baseline; Phase B'-1 adds product code that may drift the budget) | STOP — re-baseline only with maintainer ratification |

### 5.4 Pre-release gates

**N/A for Phase B'-1.** Pre-release gates (constitution audit, SC-006
benchmark, privacy audit, docs build, quickstart smoke, tag/CHANGELOG
match) fire only at the Phase C release ceremony. See the future
`09-implement-phase-c.md` prompt.

### 5.5 Gate failure philosophy

- **FAIL LOUD**. The agent never uses `pytest ... || true`, never sets
  `continue-on-error: true` in a CI job, never suppresses mypy via
  bare `# type: ignore`. Every suppression is targeted AND justified.
- Gates that are flagged as WARNINGs (per-module coverage; G-US1-IT-6
  pending manual) surface in the progress report and accrue in
  CHANGELOG `[Unreleased] / ### Debt` if any are introduced.
- The install-footprint budget (G-Slow-Install-Footprint) may need
  re-baselining if US2's plotter additions exceed +5% over the
  Phase A baseline. The agent does NOT re-baseline silently —
  surfaces to maintainer with the measured delta and asks. The same
  rules from Phase A discovery #1 apply: re-baseline only after the
  full phase product code lands, not midway.

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

Phase B'-1 risk areas: Plotly's runtime types are loose (many
`go.Figure` operations are typed as `Any`). Targeted ignores around
`fig.layout.template = ...` assignments are acceptable IF the
surrounding code asserts the type via `cast()` or a structural check.

### 6.4 External failures

- GitHub API rate limit / transient 5xx: exponential backoff (2 s,
  4 s, 8 s); give up after 3 attempts; surface to maintainer.
- Kaleido subprocess failures (Phase B'-1 specific): if
  `kaleido.write_fig()` (or the static-export pathway from T038)
  fails with a Chromium-dependency error, surface the exact OS command
  needed. Common cause: missing system libs on minimal containers.
  Phase B'-1 assumes the maintainer's dev machine has them per the
  Phase A `CONTRIBUTING.md` setup; if running in a CI runner, the
  install-footprint job already exercises this path.
- vhs failures during T030: not the agent's concern — T030 is a
  maintainer-side hard stop (§8.1).

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
- A `docs/assets/hero.gif` file present BEFORE T030's maintainer
  handback — likely a stale artefact from a prior invocation. STOP
  and ask whether to keep, replace, or remove.

The agent describes the state, lists the three most-likely causes,
and asks the maintainer before touching anything.

---

## 7. Ambiguity Resolution

A task description might be ambiguous at execution time (e.g.,
T037 "add an `Annotation`-based stats overlay reporting algorithm
name, boxes placed (`N/total`), overall utilisation %, and runtime in
ms" — what placement on the figure? Top-left? Bottom-right? In the
title bar?). Decision tree:

**Step 1 — Consult design artefacts** (2-min budget):

- `plan.md` § Project Structure: file path authoritative.
- `research.md` ADR on the topic: decision authoritative. For Phase
  B'-1 the relevant ADRs are 001 (Plotly+Kaleido), 004 (regenerated
  sections), 005 (Highlights regeneration), 006 (theme), 007
  (palette), 008 (CLI shape for `--static-format` mirroring), 011
  (stats overlay layout — defines the four-key shape), 012
  (colourblind verification).
- `data-model.md`: entity shape authoritative
  (`VisualisationStyle.stats_overlay_layout` defines the overlay
  position contract — Phase B'-1 reads, does not modify).
- `contracts/*.md`: public-surface authoritative
  (`visualisation-theme.md` for T037+T038+T039;
  `algorithm-card-source.md` for T028's table generator;
  `repository-structure.md` for T028's Project Structure block).

If a clear answer emerges, proceed with an inline one-line comment in
the commit body: `Resolved via ADR-011 (research.md)`.

**Step 1.5 — Cross-track hygiene check** (30-sec budget):

Before touching any file under `src/bin_packer_3d/`, consult the §9
no-touch list AND
`Speckit-context-prompts/spec-02-portfolio-polish/02-plan.md` § 11.
If the target file is in US5 territory, STOP immediately — this is not
an ambiguity, this is a boundary violation.

**Step 2 — Consult constitution** (30-sec budget):

Does a principle prescribe a default? Examples:

- Principle I (Contract Honesty) — Phase B'-1's `--static-format`
  CLI flag (T039) must be documented in the same commit that adds
  it; the help text must match the contract in `visualisation-theme.md`.
- Principle IV (Reproducibility & Determinism) — the plotter's HTML
  output must be byte-identical across two runs on the same input.
  This is what G-US2-IT-1 enforces. Any source of non-determinism
  (timestamps in metadata, random hover-template ordering, dict
  iteration order in Python <3.7) is a fix-required bug, not a
  documented limitation.
- Principle V (Library Citizenship) — new module imports
  (`import kaleido` inside the static-export branch) MUST be local
  to the function that uses them, NOT top-level. Top-level import
  would force every plotter consumer to install the `viz` extra
  even when not exporting statically.

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
- Log message wording (Phase B'-1 introduces no new logging beyond
  what the constitution allows on the CLI surface).
- Internal variable naming.
- Test-case parameter values (seeds, dimensions) if the test is
  property-based.
- File-order within a task's scope.
- vhs Tape script timing parameters (T029) — within the 8 fps / ~30
  second envelope from tasks.md, the exact dwell times per command
  are agent-discretion. Surface only if the maintainer's GIF size
  budget appears at risk (the tasks.md envelope already targets
  ~800×500 viewport).

Document the default in the commit body: `Default: <what>`.

---

## 8. Hard Stopping Points

Phase B'-1 has **one task-specific hard stop**: T030 (vhs hero-GIF
recording).

### 8.1 T030 — vhs hero recording (DR-13 LOCKED)

T030 reads (verbatim from tasks.md): "Maintainer manual step: run
`vhs scripts/render_demo_gif.tape -o docs/assets/hero.gif` on Linux;
commit `docs/assets/hero.gif`. Documented in `docs/maintainers.md`
§ Hero asset regeneration."

Stop-and-resume protocol:

1. After T029 commits (the Tape script), the agent emits a
   **mid-invocation handback message**:

   ```
   HARD STOP — T030 maintainer-side action required.

   Action: from a Linux machine with vhs installed, run:
     vhs scripts/render_demo_gif.tape -o docs/assets/hero.gif

   Expected output: a ~30-second 800×500 GIF at docs/assets/hero.gif.

   Then commit with:
     git add docs/assets/hero.gif docs/maintainers.md
     git commit -m "assets: record hero GIF via vhs (T030)" -m "<body
     citing T029 tape SHA>"

   Surface "T030 done" to resume Phase B'-1 at T031 (README rewrite).
   ```

2. Agent freezes. Does NOT proceed to T031. Does NOT touch any other
   file (other Phase B'-1 work is sequentially downstream of T030 in
   the dependency graph; there is nothing parallel-safe to do during
   the stop).

3. On maintainer resumption with "T030 done" (or equivalent), agent
   verifies:
   - `docs/assets/hero.gif` exists.
   - `git log -1 docs/assets/hero.gif` returns a recent commit.
   - The file is non-empty (size > 100 KB is a reasonable floor for a
     30-second GIF at this resolution).
   If any check fails, surface and re-stop; do not proceed.

4. Resume at T031.

If the maintainer is not on Linux and cannot record vhs locally, the
fallback (per DR-13) is to use the existing
`Speckit-context-prompts/spec-02-portfolio-polish/02-plan.md` § 11.3
fallback chain (asciinema → terminalizer → static screenshot). The
agent does NOT pick the fallback unilaterally; the maintainer chooses
and announces the choice as part of resumption.

### 8.2 General no-go list (inherited from Phase A)

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
  `.venv-phase-b1-check/` (the temporary US2-IT verify venv, if used).
- Any GitHub API mutation (no PR creation / merging / closing without
  maintainer say-so — see DR-10 below; the agent prepares PR
  descriptions but does NOT `gh pr create`).

---

## 9. Cross-Track Hygiene Policy (INHERITED FROM PHASE A + REGISTRY NOTE)

Spec-02 ships in parallel with US5 algorithm work on
`feature/us5-extreme-point-benchmark`. Both branches were cut from
`develop` at `08ca725` and will eventually merge back to `develop`.
Phase A's hygiene rules carry forward; Phase B'-1 adds a registry
note for T033 gallery generation.

### 9.1 No-touch list (US5 territory — DR-9 INHERITED LOCKED)

The agent MUST NOT create, modify, rename, or delete any of these
files during Phase B'-1:

- `src/bin_packer_3d/algorithms/extreme_point.py` (US5 creates this)
- `src/bin_packer_3d/algorithms/maximal_rectangles.py` (US5 creates this)
- `src/bin_packer_3d/benchmark/runner.py` (US5 implements this)
- Any new file inside `src/bin_packer_3d/benchmark/` beyond what
  spec-01 US5 Part 1 already merged.

Reading these files for context is allowed (e.g., to understand the
existing `ALGORITHMS` registry interface for T033). Writing is not.

### 9.2 Touch-with-care list

The agent MAY touch these files but must add a `cross-track: spec-02
Phase B'-1` body line to the commit and minimise the diff:

- `src/bin_packer_3d/cli.py` (T039 adds `--static-format` to the
  `pack` command). US5 may later add CLI options to `pack` for
  algorithm-specific tuning; coordinate via the existing
  Click-decorator pattern — do not pre-stage US5 options here.
- `src/bin_packer_3d/visualization/plotter.py` (T037+T038 extend the
  rendering pipeline). US5 does NOT touch plotter.py per the spec-01
  US5 task list, but if a rebase surfaces a conflict, STOP and
  surface.

### 9.3 Rebase protocol

If during Phase B'-1 the maintainer reports that US5 Part 2 has
merged to `develop`, the agent:

1. STOPS at the current task boundary (does NOT continue mid-task).
2. Runs `git fetch origin develop`.
3. Runs `git rebase origin/develop` on `feature/002-portfolio-polish`.
4. If rebase clean: re-runs `pytest tests/ -m "not slow"` to confirm
   the post-Phase-A 118-test baseline still passes alongside the new
   US5 tests. Surface a one-line confirmation: `rebased on develop @
   <sha>; tests still green`.
5. If rebase produces conflicts: STOP and surface the conflicted file
   list. The maintainer decides resolution.
6. **Phase B'-1 specific**: if US5 Part 2 merged AFTER T028 (the
   regenerator) shipped, the regenerator's algorithm-table output
   will now include `extreme_point` and `maximal_rectangles`. The
   agent re-runs `python scripts/regenerate_readme.py` and amends
   the T032 commit (if not yet pushed) or creates a follow-up
   regeneration commit. T025/T027 drift tests will need re-running.

### 9.4 Registry-driven gallery (FR-004 — NEW vs Phase A)

T033 generates one PNG per algorithm key in `ALGORITHMS`. The
implementation MUST iterate `ALGORITHMS.keys()` — NEVER hardcode the
three known packers (`bfd`, `ffd`, `shelf`). This way:

- If US5 Part 2 merges to develop AFTER Phase B'-1 closes, the next
  Phase (B'-2 or beyond) reruns T033's equivalent step automatically
  and the gallery extends without code changes.
- If US5 Part 2 merges DURING Phase B'-1 (mid-rebase per §9.3), T033
  picks up the new packers in the same invocation. The agent does NOT
  pre-emptively render PNGs for them if rebasing is the maintainer's
  call — the agent surfaces the merge first, then proceeds.

### 9.5 Drift-test note (forward-looking)

T042 (in Phase B'-2, US3 docs) introduces
`tests/integration/test_docs_build.py` asserting "every key in
`ALGORITHMS` has a corresponding `docs/algorithms/<key>.md`". If US5
Part 2 has merged before Phase B'-2, the new `extreme_point` and
`maximal_rectangles` algorithm pages will need stubs. This is **NOT**
a Phase B'-1 concern — flagged here for situational awareness so the
agent does not pre-stage docs for not-yet-merged packers.

---

## 10. Memory & Telemetry

The project uses Engram for cross-session memory. Protocol per
project `CLAUDE.md` + global Engram instructions.

### 10.1 After each tight task group

Call `mem_save` with:

- **title**: verb + object, e.g., "Plotter: branded theme + stats
  overlay applied (T037)".
- **type**: `decision`, `bugfix`, `architecture`, `pattern`, `config`,
  `discovery`, `learning`.
- **topic_key**: stable key under the reserved spec-02 Phase B'-1
  prefix (see §10.3).
- **content** in the 4-block format (`**What**`, `**Why**`, `**Where**`,
  `**Learned**`).
- **scope**: `project`.

### 10.2 At phase close (before exiting)

Call `mem_session_summary` per the §Protocol in the project
`CLAUDE.md` / global Engram instructions. Use the Goal / Instructions /
Discoveries / Accomplished / Next Steps / Relevant Files shape.

### 10.3 Topic keys for spec-02 Phase B'-1

Reserve these prefixes — do not drift:

- `3d-bin-packing/spec-02/phase-b-prime-1/precursor/kaleido-pin` (DR-11 commit)
- `3d-bin-packing/spec-02/phase-b-prime-1/us2/plotter/*` (T035, T037, T038)
- `3d-bin-packing/spec-02/phase-b-prime-1/us2/cli/*` (T039)
- `3d-bin-packing/spec-02/phase-b-prime-1/us2/snapshot/*` (T036)
- `3d-bin-packing/spec-02/phase-b-prime-1/us2/verify` (T040, T041)
- `3d-bin-packing/spec-02/phase-b-prime-1/us1/regenerator/*` (T025, T027, T028)
- `3d-bin-packing/spec-02/phase-b-prime-1/us1/vhs/*` (T029, T030)
- `3d-bin-packing/spec-02/phase-b-prime-1/us1/readme/*` (T026, T031, T032)
- `3d-bin-packing/spec-02/phase-b-prime-1/us1/gallery/*` (T033)
- `3d-bin-packing/spec-02/phase-b-prime-1/us1/verify` (T034)
- `3d-bin-packing/spec-02/phase-b-prime-1/closed` (single observation at handback)

Cross-cutting findings (e.g., a gotcha that affects Phase B'-2 too):
`3d-bin-packing/spec-02/cross-phase/*`.

### 10.4 Pre-flight memory search

Before the first task of the invocation, run
`mem_search query:"spec-02 phase-b-prime-1"` AND
`mem_search query:"spec-02 phase-a closed"`. The first surfaces any
prior partial Phase B'-1 work; the second pulls the Phase A
retrospective into context (mandatory — DR-11 below).

If hits to incomplete Phase B'-1 work are returned, retrieve via
`mem_get_observation` and surface to the maintainer before proceeding
— this prevents double-writes if a prior session crashed.

### 10.5 Do NOT save

- Trivial "task T0XX completed" pings (already in git log).
- Content copied from the spec / plan / constitution (already on disk).
- Code snippets the agent just wrote (already committed).

DO save: non-obvious decisions, gotchas discovered, deviations from
plan, anything that saves a future session a re-investigation.
Phase B'-1 is expected to surface 2-4 such observations (Plotly JSON
shape gotchas, Kaleido subprocess behaviours under load, vhs timing
calibration if the maintainer reports the first GIF cut is wrong).

---

## 11. Environment Setup

Before the first task of the invocation, the agent runs a pre-flight
check and reports status. If any check fails: STOP and surface.

### 11.1 Pre-flight checks

| Check | Command | Pass condition |
|---|---|---|
| Python version | `python --version` | `3.11`, `3.12`, `3.13`, or `3.14` |
| Package install | `pip show bin_packer_3d` | editable install at repo path |
| Dev extras present | `python -c "import pytest, ruff, mypy, hypothesis, colorspacious"` | all importable |
| Phase A theme reachable | `python -c "from bin_packer_3d import VisualisationStyle, BIN_PACKER_3D_DARK, BIN_PACKER_3D_LIGHT, apply_theme, colour_for_box"` | exits 0 |
| Phase A dataset present | `ls examples/headline.csv examples/headline.seed docs/assets/palette_colourblind_check.png` | all three exist |
| Phase A retrospective read | `mem_search "spec-02 phase-a closed"` → `mem_get_observation` | observation retrieved into context |
| Kaleido NOT YET pinned | `grep -E 'kaleido.*<\s*1\.0\.0' pyproject.toml` | exits 1 BEFORE the DR-11 precursor commit lands; exits 0 AFTER |
| vhs reachable for T030 | `which vhs && vhs --version` on the maintainer's Linux machine | reported, not blocking |
| Pre-commit active | `pre-commit --version` + `.git/hooks/pre-commit` exists | both yes |
| Current branch | `git branch --show-current` | `feature/002-portfolio-polish` |
| Active feature pointer | `jq -r .feature_directory < .specify/feature.json` | `specs/002-portfolio-polish` |
| Working-tree state | `git status --porcelain` | empty OR contains only `.serena/memories/sessions/*.md` from prior session |
| Remote configured | `git remote -v` | `origin` points at Bruno-Ghiberto/3D_BIN_PACKING |
| Baseline tests green | `pytest tests/ -m "not slow"` | ≥118 pass, 0 fail (post-Phase-A baseline) |
| US5 territory clean | check that `src/bin_packer_3d/algorithms/extreme_point.py`, `maximal_rectangles.py`, and any new files under `benchmark/` are either absent OR untouched since the most recent `develop` rebase | absent OR untouched |

### 11.2 DR-11 precursor commit — kaleido<1.0.0 pin

After pre-flight passes and BEFORE T035 ships, the agent lands a
one-commit precursor:

```
File: pyproject.toml
Change: in [project.optional-dependencies] viz extras list, replace
        "kaleido>=0.2.1"
        with
        "kaleido>=0.2.1,<1.0.0"

Commit subject: chore(deps): pin kaleido<1.0.0 (DR-11)
Body lines:
  - Phase A discovery #7: Kaleido 1.x is incompatible with Plotly
    5.x's fig.write_image() pathway. Pin the upper bound to <1.0.0
    so the static-export branch (T038) ships against the working
    contract. Migration to Kaleido 1.x + `kaleido.write_fig()` API
    is a separate spec.
  - Refs: 3d-bin-packing/spec-02/phase-a/closed (Engram), discovery
    item 7.
```

This is NOT a `tasks.md` task — it is a DR-11-mandated precursor.
The agent edits `tasks.md` to add a new line at the top of Phase 1
referencing the precursor (e.g., `[x] T024a Pin kaleido<1.0.0 in
pyproject.toml viz extras (DR-11 precursor per
04-implement-phase-b-prime-1.md § 11.2)`), or — alternatively — the
agent leaves `tasks.md` unmodified and the precursor stands alone in
the git log, citing this prompt as authority. **Default**: leave
`tasks.md` unmodified; the prompt is the authority. Surface the
choice to the maintainer if they prefer the explicit tasks.md edit.

After the precursor lands:

- Re-run `pip install -e .[viz]` to materialise the new pin.
- Confirm `python -c "import kaleido; print(kaleido.__version__)"`
  reports a version `<1.0.0`. If the resolved version is already
  `1.x` cached, run `pip install --upgrade --force-reinstall
  'kaleido<1.0.0'`.

If the maintainer starts the invocation from a cold start (new
machine, new clone), `CONTRIBUTING.md` contains the full setup path
— the agent may point the maintainer at it rather than running setup
itself.

---

## 12. Completion & Handoff

### 12.1 "Done" per Phase B'-1

All of:

1. Every task in T025..T041 has its checkbox flipped to `[x]` in
   `tasks.md` (agent edits this at commit time for the relevant task).
2. DR-11 precursor commit landed before T035.
3. Both Independent-Test gates green (§5.2.1 US2 + §5.2.2 US1).
4. Cross-cutting post-phase gates green (§5.3).
5. CHANGELOG.md `[Unreleased] / ### Added` updated with ≥2 bullets
   covering the viz + docs additions (§4.5).
6. Phase-close progress report emitted (§3.4) AND
   `mem_session_summary` saved (§10.2).

### 12.2 PR strategy (DR-10 SUPERSEDED FOR PHASE B'-1 — chained PRs)

Phase A's DR-10 ("single PR") is **superseded** for Phase B'-1.
Phase B'-1 estimates 800-1000 LOC across two user stories with one
hard cross-US dependency. Per the `chained-pr` skill (>400 LOC
threshold) and the maintainer's session-load decision (recorded at
authoring time of this prompt):

- **Chain 1 (PR-A)**: US2 only — T035..T041 plus the DR-11 precursor.
  Branch: `feature/002-portfolio-polish-us2` (cut from
  `feature/002-portfolio-polish` head). Target: `feature/002-portfolio-polish`
  via merge-commit (preserve granular history). Estimated ~350-450
  LOC. Opens immediately after G-US2-IT gates pass.
- **Chain 2 (PR-B)**: US1 only — T025..T034. Branch:
  `feature/002-portfolio-polish-us1` (cut from
  `feature/002-portfolio-polish` head AFTER chain 1 merges). Target:
  `feature/002-portfolio-polish` via merge-commit. Estimated
  ~400-550 LOC. Opens immediately after G-US1-IT gates pass.
- **Final integration PR**: `feature/002-portfolio-polish` → `develop`
  remains the per-spec-02 phase target. Opens at Phase C release
  ceremony (NOT at end of Phase B'-1).

The agent does NOT open the PRs — that is a maintainer action after
the agent's handback reports green for each gate. The agent prepares
the PR description in `.git/SCRATCH_PR_<US>_DESCRIPTION.md` (untracked)
with title, summary, test plan, and "linked spec / plan / tasks"
references. Maintainer copies + opens.

If during Phase B'-1 the maintainer chooses to override chained PRs
and ship as a single PR with `size:exception` (revoking the DR-10
supersede), surface the request and continue without rebasing into
separate chains — but DO surface that the override creates a single
review unit that the `chained-pr` skill recommends against.

### 12.3 When to re-run `/speckit-tasks`

If during Phase B'-1 the agent identifies a task that is MISSING from
`tasks.md` (not a task-reorder issue — a genuine gap), it STOPS and
surfaces. The maintainer decides whether to:

- re-run `/speckit-tasks` to regenerate (tasks.md rewritten end-to-end),
  OR
- amend `tasks.md` inline with the new task ID (e.g., `T041a`).

The agent does NOT extend `tasks.md` on its own. The one exception is
the DR-11 precursor `T024a` line discussed in §11.2 — and that is
recorded as a "default = don't edit tasks.md" with maintainer opt-in.

### 12.4 When to re-run `/speckit-analyze`

Optional at Phase B'-1 close. MANDATORY at each release-milestone
boundary (next one: pre-Phase-C-tag). MANDATORY if any spec/plan
amendment occurs mid-implementation.

### 12.5 Handback message shape at invocation end

```
Invocation complete — Phase B'-1 US2 + US1 (2 of 7).
Result: <PASS | PARTIAL | FAIL>
Precursor: DR-11 kaleido<1.0.0 pin landed (commit <SHA>)
Commits: M on feature/002-portfolio-polish across two chains
  Chain 1 (US2): K commits — PR-A draft ready at
    .git/SCRATCH_PR_US2_DESCRIPTION.md
  Chain 2 (US1): L commits — PR-B draft ready at
    .git/SCRATCH_PR_US1_DESCRIPTION.md
Tests: 118 baseline + N new = total / 0 fail
Coverage: X.X% (≥90% required)
Mypy: 0 errors (strict)
US2 Independent-Test: <PASS | FAIL — <which sub-criteria>>
US1 Independent-Test: <PASS | FAIL — <which sub-criteria>>
  G-US1-IT-6 manual (maintainer): <PENDING | PASS | FAIL>
Hard stops encountered:
  - T030 vhs hero recording — resolved by maintainer at <commit SHA>
Cross-track events: <e.g., rebased on develop @ <sha> | none>
Next recommended invocation: Phase B'-2 (US3 docs site, T042..T058)
Open questions for maintainer (if any):
  - <e.g., G-US1-IT-6 manual gate verification>
  - <e.g., install-footprint re-baseline request if budget exceeded>
```

---

## 13. Open Questions

At authoring time of THIS prompt — **two**, both already resolved as
locked DRs below. Recorded here for traceability:

1. **PR sizing for Phase B'-1** (RESOLVED — DR-10 supersede, chained
   PRs). Captured in §12.2.
2. **Kaleido version pin direction** (RESOLVED — DR-11, pin to
   `<1.0.0`, defer Kaleido 1.x migration). Captured in §11.2 +
   Phase A discovery #7.

If the maintainer invokes `/speckit-implement` for Phase B'-1 with
new open questions surfacing, the agent MUST refuse to proceed and
request resolution first. Keeping this section limited to resolved
items (or empty) is itself a gate.

### Future-proofing

Examples of what WOULD belong here if it arose:

- "Does the stats overlay reveal a Plotly version constraint that
  conflicts with the ADR-001 pin?"
- "If `bin-packer pack` exits non-zero on `examples/headline.csv` for
  any registered algorithm during T033, do we treat that as a Phase
  B'-1 blocker or surface it as a spec-01 US5 quality issue?"
- "If the maintainer's vhs recording at T030 produces a hero.gif
  larger than the GitHub README image budget (~10 MB recommended,
  hard cap unspecified), do we trim, re-record at lower fps, or
  swap to a static screenshot?"

---

## Locked decisions summary

| DR | Topic | Locked value |
|---|---|---|
| DR-1 | Execution scope per invocation | **Per phase/bundle** — 7 invocations for spec-02 (inherited from Phase A) |
| DR-2 | Commit grouping | One commit per task OR tight test+impl pair OR `[P]`-island composite; never whole-phase (inherited) |
| DR-3 | AI attribution on commits | NONE — no Co-Authored-By, no robot trailer (inherited) |
| DR-4 | Interactive vs unattended | Interactive — maintainer triggers each invocation (inherited) |
| DR-5 | Coverage gate | Overall ≥90% blocking; per-module ≥70% warning-only (inherited) |
| DR-6 | Branch & PR workflow | Direct commits on `feature/002-portfolio-polish`; intermediate PRs per chain into the feature branch; release tag from `main` in Phase C (inherited) |
| DR-7 | Task re-ordering authority | None — agent surfaces ordering concerns, never reorders silently (inherited) |
| DR-8 | Mid-invocation skill use | Read-only only (`/sc:explain`, `/sc:analyze`); no scope-changing skills (inherited) |
| DR-9 | US5 cross-track no-touch list | 3 files locked (`algorithms/extreme_point.py`, `algorithms/maximal_rectangles.py`, `benchmark/runner.py`); rebase-on-merge protocol enforced (inherited) |
| **DR-10** (SUPERSEDED) | **PR sizing for Phase B'-1** | **Chained PRs** — Chain 1 (US2, T035..T041 + DR-11 precursor) + Chain 2 (US1, T025..T034), both targeting `feature/002-portfolio-polish` via merge-commit; ~350-450 LOC + ~400-550 LOC respectively |
| **DR-11** | **Kaleido version pin** | `kaleido>=0.2.1,<1.0.0` — Phase A discovery #7 (Kaleido 1.x incompatible with Plotly 5.x `fig.write_image()` pathway); migration to Kaleido 1.x `write_fig()` API deferred to a future spec |
| **DR-12** | **Cross-US ordering inside Phase B'-1** | **US2 before US1** — T035..T041 ships first (Chain 1); T025..T034 ships second (Chain 2). Rationale: T037+T038 (Kaleido static export) unblocks T033 (gallery PNGs) without temporary placeholders |
| **DR-13** | **T030 hard-stop protocol** | Agent halts after T029 commits, surfaces mid-invocation handback with the vhs invocation command; maintainer records hero.gif on Linux + commits; agent resumes at T031 on "T030 done" signal. Fallback chain (asciinema → terminalizer → static screenshot) per plan-context-prompt § 11.3 if vhs unavailable; maintainer chooses |

These are not suggestions. The agent executing `/speckit-implement`
against spec-02 Phase B'-1 MUST obey all thirteen.
