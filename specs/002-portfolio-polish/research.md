# Research: Portfolio Polish of `bin-packer-3d`

**Phase**: 0 (research) | **Feeds**: `plan.md` § Constitution Check, § Project Structure, § Phased Milestones | **Date**: 2026-05-13

This document ratifies the 12 architectural decisions taken in `Speckit-context-prompts/spec-02-portfolio-polish/02-plan.md` § 5 as ACCEPTED ADRs, and resolves the 7 plan-level open questions listed in `plan.md` § Open Questions. Every ADR includes Decision / Rationale / Alternatives Considered per the SpecKit convention.

---

## Architectural Decisions

### ADR-001 — Plotly static-export backend + version pin

**Status**: ACCEPTED.

**Decision**: Kaleido as Plotly's static-export backend, declared under `[project.optional-dependencies]` group `viz`. Plotly itself pinned to `>=5.18.0,<6.0.0` to lock major-version palette internals and minimise risk of screenshot pixel drift across minor versions.

**Rationale**:

- Kaleido is the canonical, maintained Plotly static-export backend, supporting PNG, JPEG, SVG, PDF, EPS. It is deterministic by-byte across runs given the same Plotly version and figure input.
- Kaleido's install footprint (~50 MB depending on platform) would breach FR-027's +5% install-footprint budget if added as a mandatory runtime dep. Optional placement under `[viz]` preserves the budget for users who want only interactive HTML.
- The Plotly upper bound is justified by the project's reliance on deterministic colour-and-layout output (SC-006, ADR-007). Plotly minor versions occasionally adjust default palette internals or marker rendering, which would silently drift snapshot tests. Pinning the major version contains that risk; an explicit Plotly 6.x upgrade becomes a separate spec when the time comes.

**Alternatives considered**:

- **Orca (Plotly's previous backend)**: Deprecated since 2021; no longer maintained. Rejected.
- **Server-side rendering via headless Chrome (e.g., Pyppeteer)**: Heavier dependency, more failure modes, less deterministic. Rejected.
- **Hand-curated screenshots only**: Trades automation for maintenance burden; conflicts with FR-023 (every visual claim regeneratable by command). Rejected.
- **Plotly with no upper bound**: Risks screenshot drift on every Plotly minor release. Rejected.

**Implementation scope**:

- `pyproject.toml`: `plotly` line → `"plotly>=5.18.0,<6.0.0"`; new optional `viz = ["kaleido>=0.2.1"]`.
- `src/bin_packer_3d/visualization/plotter.py`: static-export branch guarded by `try: import kaleido except ImportError`; clear instruction message on failure: `"Static export requires 'kaleido'. Install with: pip install 'bin-packer-3d[viz]'"`.
- README install snippet shows both forms — basic install and `[viz]` extra.

---

### ADR-002 — Documentation framework, host, and deploy mechanism

**Status**: ACCEPTED.

**Decision**: `mkdocs-material` for build; GitHub Pages for hosting; GitHub-official `actions/deploy-pages` action for deployment (not `peaceiris/actions-gh-pages`).

**Rationale**:

- `mkdocs-material` is already declared as an optional dependency (`[docs]` extra) and its default theme is documented WCAG-AA conformant. Markdown-first authoring matches the project's existing documentation style. `mkdocstrings[python]` plugin provides automatic API reference from docstrings.
- GitHub Pages is zero-infrastructure: the repo is already on GitHub; no separate provider account, no DNS, no payment.
- `actions/deploy-pages` (the GitHub-official action) integrates with the modern Pages "Build with GitHub Actions" source mode, which uses Pages' built-in artefact protocol. This avoids maintaining a `gh-pages` branch (the older pattern used by `peaceiris/actions-gh-pages`), keeping the repo tree cleaner. The official action also handles permissions correctly via `pages: write` + `id-token: write`.

**Alternatives considered**:

- **`peaceiris/actions-gh-pages`**: Well-known, widely used, mature; but maintains a `gh-pages` branch and requires more permissions juggling. Inferior to the official action for new setups. Rejected.
- **Sphinx + Read the Docs**: Heavier; requires RST or MyST; external service account needed. Overkill for a sole-maintainer project. Rejected.
- **`mkdocs gh-deploy` command inside CI**: Works but bypasses the Pages "Build with GitHub Actions" mode and the artefact upload step; less observable. Rejected.

**Implementation scope**:

- `mkdocs.yml` at repo root, Material theme + `mkdocstrings[python]` plugin + navigation per plan.md § 3.
- `.github/workflows/docs-deploy.yml`:
  - Trigger: `push: branches: [main]`.
  - Permissions: `pages: write`, `id-token: write`, `contents: read`.
  - Jobs: `build` (`pip install '.[docs]'` + `mkdocs build --strict` + `actions/upload-pages-artifact`) and `deploy` (`actions/deploy-pages`).
- One-time maintainer setup in `docs/maintainers.md` § Pages setup: repo Settings → Pages → Source = "GitHub Actions".

---

### ADR-003 — Algorithm-page strategy (docs/algorithms/)

**Status**: ACCEPTED.

**Decision**: Hybrid approach — `mkdocstrings` auto-generates the API reference page (`docs/api/index.md`); each `docs/algorithms/<key>.md` is hand-written prose with YAML front-matter that MUST match the corresponding `ALGORITHMS` registry entry; a drift test enforces the registry-doc agreement at build time.

**Rationale**:

- `mkdocstrings` alone produces dry pages derived from docstrings — fine for API ref, but insufficient for explaining algorithm intuition, complexity reasoning, and literature citations to an operations-research practitioner.
- Hand-written prose adds the framing a reviewer wants: where the algorithm comes from, what kind of problem it suits, what its weaknesses are, and a literature citation. This is the storytelling that justifies the project as a portfolio piece.
- The drift test (asserting every registry key has a matching `.md` and every front-matter matches registry metadata) enforces Principle I (Contract Honesty) without requiring full autogeneration. The runtime is still the source of truth.

**Alternatives considered**:

- **Full autogeneration (mkdocstrings only)**: Loses prose framing. Rejected.
- **Pure hand-written, no drift test**: Allows divergence between registry and docs. Violates Principle I. Rejected.
- **Hand-written WITH manual review of registry alignment at each release**: Adds maintainer toil; not automation-grade. Rejected.

**Implementation scope**:

- Each `docs/algorithms/<key>.md` opens with:
  ```yaml
  ---
  key: bfd                       # MUST match an ALGORITHMS registry key
  complexity: "O(n log n)"       # MUST match the registry entry's complexity_class
  citation: "Crainic et al., 2008"
  ---
  ```
- `tests/integration/test_docs_build.py`:
  - Assert every key in `ALGORITHMS` has a corresponding `docs/algorithms/<key>.md`.
  - Assert each page's front-matter `key:` value is a valid registry entry.
  - Assert each front-matter `complexity:` matches the registry entry's `complexity_class`.

---

### ADR-004 — Comparison-table sync (README)

**Status**: ACCEPTED.

**Decision**: Generated-section pattern — README contains a block bounded by explicit markers:

```markdown
<!-- BEGIN: ALGORITHMS_TABLE (auto-generated by scripts/regenerate_readme.py; do not edit) -->
... regenerator output ...
<!-- END: ALGORITHMS_TABLE -->
```

The script `scripts/regenerate_readme.py` reads the `ALGORITHMS` registry and overwrites the block. CI test `tests/integration/test_readme_drift.py` re-runs the script and asserts `git diff --exit-code README.md` is clean.

**Rationale**:

- Principle I (Contract Honesty) demands the runtime is the truth. A drift-test-only approach (handwritten content + post-hoc test) allows divergence between commits — author edits the registry, forgets to update the README, the test catches it AT PR time but the local clone still drifts.
- The markered-block pattern is industry standard (`automd`, `mike`, many tooling pipelines). It is intentionally visible in the README so contributors know not to edit by hand.
- One script for three regenerated sections (table, Highlights, Project Structure) keeps the regeneration surface minimal.

**Alternatives considered**:

- **Drift test only, hand-written table**: Allows divergence. Rejected.
- **Inject via mkdocs macros at site build time**: Couples README to the docs build; README is also displayed on GitHub where macros don't run. Rejected.
- **GitHub-specific includes (e.g., README.md.in)**: Non-standard; tooling fragmentation. Rejected.

**Implementation scope**:

- `scripts/regenerate_readme.py` is the single source for table generation; extends to Highlights (ADR-005) and Project Structure (FR-016).
- README is initially authored with placeholder markers; the script populates them.
- Optional pre-commit hook (manual stage) runs the script on README edits.

---

### ADR-005 — Highlights-section sync (README)

**Status**: ACCEPTED.

**Decision**: Generated-section pattern, same shape as ADR-004, for the README "Highlights" block. Sources:

- Algorithm count: `len(ALGORITHMS)`
- Test count: parsed from `pytest --collect-only -q | tail -1`
- CI check count: parsed from `.github/workflows/_ci-core.yml` `jobs:` keys
- Supported Python versions: parsed from `pyproject.toml` `classifiers`
- License: parsed from `pyproject.toml` `license`
- Property-based-testing presence: presence of `tests/property/` directory
- Strict-mypy presence: `[tool.mypy] strict = true` in `pyproject.toml`

**Rationale**:

- FR-026 explicitly states drift between Highlights numbers and live state is a Contract Honesty violation. Automated regeneration eliminates the drift surface.
- The seven Highlight inputs are all parseable from canonical files (pyproject, workflow, pytest collect); no manual maintenance required.

**Alternatives considered**:

- **Manual annual review**: Inevitable drift. Rejected.
- **CI-only verification**: Catches drift at PR time but locally-stale content slips into commits. Rejected.

**Implementation scope**:

- `scripts/regenerate_readme.py` extends to populate the Highlights block.
- `tests/integration/test_highlights_drift.py` asserts post-regen diff is empty.

---

### ADR-006 — Visualisation theme implementation pattern

**Status**: ACCEPTED.

**Decision**: Plotly figure template (`go.layout.Template`) authored once in `src/bin_packer_3d/visualization/theme.py`. Two variants: `BIN_PACKER_3D_DARK` and `BIN_PACKER_3D_LIGHT`. Applied via `apply_theme(fig: go.Figure, variant: Literal["dark", "light"] = "dark") -> go.Figure`. Per-call overrides are limited to run-specific data (figure title text, stats overlay values).

**Rationale**:

- Plotly's template mechanism is the idiomatic way to brand figures: declarative, composable, single source of truth. A theme dict is more debuggable than scattered `update_layout` calls.
- Light/dark variants cover the most common reader contexts (README rendered on GitHub light/dark mode, slide deck embedding).
- Keeping the theme as data (a constant) plus a thin function (`apply_theme`) makes it trivial to test (assert the template name) and to mock in tests of downstream rendering.

**Alternatives considered**:

- **Per-call `update_layout` overrides**: More code, easier to drift; harder to maintain consistency. Rejected.
- **Single theme variant**: Less flexible for dark-mode embedding. Rejected.
- **Subclass `go.Figure`**: Overengineered; breaks Plotly idiom. Rejected.

**Implementation scope**:

- `theme.py` exports:
  - `BIN_PACKER_3D_DARK: go.layout.Template`
  - `BIN_PACKER_3D_LIGHT: go.layout.Template`
  - `apply_theme(fig: go.Figure, variant: Literal["dark", "light"] = "dark") -> go.Figure`
- `plotter.py` calls `apply_theme(fig)` after figure construction.

---

### ADR-007 — Deterministic colour assignment

**Status**: ACCEPTED.

**Decision**: Stable hash of box identifier → palette index. Hash function: BLAKE2b (stdlib `hashlib.blake2b`, 16-bit digest). Base palette: ColorBrewer "Set3" (12 colours, qualitative). For inputs with >12 boxes, cycle the palette via modulo indexing. The colourblind-safety claim is re-verified empirically by ADR-012 — ColorBrewer's documentation rates Set3 as "colourblind safe" with caveats; the empirical verification artefact is the ground truth.

**Rationale**:

- BLAKE2b is deterministic across Python versions (no `PYTHONHASHSEED`-style randomisation, unlike `hash()`). Identical box IDs always produce identical digests.
- ColorBrewer palettes are widely cited in cartography and data visualisation; "Set3" is one of the qualitative palettes flagged colourblind-friendly. The empirical re-verification (ADR-012) keeps the project honest.
- 16-bit digest (`digest_size=2`) is sufficient for palette indexing (modulo 12 always succeeds); using a smaller digest reduces collision-clustering bias.

**Alternatives considered**:

- **Python's built-in `hash()`**: Subject to `PYTHONHASHSEED` randomisation; non-deterministic across processes. Rejected.
- **MD5 / SHA-1 / SHA-256**: All deterministic; BLAKE2b is faster on modern CPUs and avoids any cryptographic-deprecation concerns (MD5/SHA-1) while being a stdlib primitive (unlike xxhash). Selected.
- **Viridis / Plotly default palettes**: Continuous palettes (viridis) are weak for categorical data; Plotly default is not documented colourblind-safe. Rejected.
- **Custom hand-tuned palette**: More work, no clear gain over ColorBrewer. Rejected.

**Implementation scope**:

- `palette.py`:
  - `SET3: tuple[str, ...]` — 12 hex strings from ColorBrewer Set3.
  - `colour_for_box(box_id: str, palette: tuple[str, ...] = SET3) -> str`:
    1. `digest = hashlib.blake2b(box_id.encode("utf-8"), digest_size=2).digest()`
    2. `index = int.from_bytes(digest, "big") % len(palette)`
    3. Return `palette[index]`.
- Unit test asserts: same box ID → same colour; >12 unique box IDs cycle through the palette without crashing; specific known box ID maps to a documented colour (regression guard).

---

### ADR-008 — Demo command surface

**Status**: ACCEPTED.

**Decision**: `bin-packer demo` — a new Click subcommand registered alongside the existing `pack | info | init`. Defaults: output dir `examples/output/`, dataset `examples/headline.csv`, static format `png`. Optional `--dataset PATH` overrides the headline dataset; `--static-format {png,svg}` switches the static export format.

**Rationale**:

- Consistent with the existing CLI pattern (`pack | info | init`); zero new tool surface (no Makefile, no standalone script).
- Reuses existing packer / loader / metrics code without duplication.
- Iterates `ALGORITHMS` registry, honouring the "every registered algorithm" semantics of FR-020.

**Alternatives considered**:

- **`make demo` Makefile target**: Adds a new tooling surface; harder to discover; less integrated. Rejected.
- **Standalone `scripts/run_demo.py`**: Same issue; doesn't show up under `bin-packer --help`. Rejected.
- **Top-level entry point `bin-packer-demo`**: Pollutes the entry-point namespace; reviewer expects subcommands. Rejected.

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
- Output structure (per `contracts/demo-command.md`):
  ```text
  examples/output/
  ├── bfd/
  │   ├── bin_1.html
  │   ├── bin_1.png
  │   ├── placements.csv
  │   └── summary.json
  ├── ffd/   └── ...
  ├── shelf/ └── ...
  └── comparison.md            # per-algorithm metrics table
  ```
- 60-second total budget enforced via `tests/integration/test_demo_command.py`.

---

### ADR-009 — Headline dataset generator approach

**Status**: ACCEPTED (algorithm-choice sub-decision below).

**Decision (high-level)**: Deterministic seeded generator. Output contract LOCKED (script path, seed-file schema, output CSV columns, byte-identical regen, ≥60% BFD utilisation acceptance, pure-Python). Sub-decision below selects the specific generation algorithm.

**Sub-decision — generation algorithm**: **Bin-feasibility-driven recursive guillotine cuts**. Algorithm:

1. Initialise a virtual bin at default dimensions (860×890×1040 mm).
2. Select a target volume utilisation (e.g., 0.65 = 65% to comfortably exceed FR-034's ≥60% floor).
3. Compute the target packed volume `V_pack = target_utilisation × bin_volume`.
4. Recursively partition the packed region via random guillotine cuts (seeded RNG) until the desired box count is reached OR each sub-region's volume falls below `min_box_volume_mm3`.
5. Collect the leaf-region dimensions as boxes.
6. Shuffle the box list with the same seed (preserves the determinism guarantee).
7. Write to CSV in the format expected by existing loaders.

**Rationale for the algorithm choice**:

- **Bin-feasibility-driven** guarantees a feasible solution exists at the target utilisation — by construction, the boxes were CUT FROM a feasible packing. This eliminates the risk that the generator emits an infeasible instance and the test fails.
- The algorithm is pure-Python (no native deps), trivially deterministic given a seed, and produces realistic box-size distributions (within ranges constrained by the cut depth).
- Total runtime ≤5 seconds for ~50 boxes — well within the test budget.

**Alternatives considered**:

- **Rejection sampling from a parameter distribution**: Simpler conceptually; sample (n_boxes, dimension-distribution) pairs, run BFD, reject if utilisation < threshold. But: no feasibility guarantee; failure rate depends on parameter choice; longer convergence time; harder to reason about. Rejected as the primary approach.
- **Adapted Martello-Vigo synthetic generator**: Closer to OR literature but heavier implementation; matches benchmark conventions but US5 reserves that vocabulary (BR1..BR8 are coming). Mixing it here creates scope crossover with US5. Rejected.
- **Hand-author + seed-based shuffle**: Not really "procedural"; lacks parametric control. Rejected.

**Output contract** (LOCKED, regardless of algorithm):

- Script path: `scripts/generate_headline_dataset.py`.
- Input: seed file (`examples/headline.seed`) containing `{seed: int, target_utilisation: float, bin_dimensions: [l,w,h], n_boxes: int, min_box_volume_mm3: float}`.
- Output: `examples/headline.csv` matching the CSV column order `ITEM,W,H,L,CANTIDAD,CAJA,DESCRIPCION` (compatible with the existing loader).
- Byte-identical regeneration: same seed → same CSV bytes.
- Acceptance: generated dataset packed by BFD on default bin dims yields overall utilisation ≥60%.
- Pure-Python implementation; no native deps.

---

### ADR-010 — Versioning workflow + tag ceremony

**Status**: ACCEPTED.

**Decision**: `hatch version 0.3.0rc1` for the `pyproject.toml` bump; manual `git tag -a v0.3.0-rc1` on the merge commit; release notes drafted from `CHANGELOG.md` `[Unreleased]` entries via `gh release create --draft`. No PyPI publish.

**Rationale**:

- Hatch is the existing build backend (`[build-system] requires = ["hatchling"]`); `hatch version` updates `pyproject.toml` cleanly without sed/manual-edit risk.
- Manual tag preserves the lightweight release flow established by v0.1.0 / v0.2.0 history. No additional CI machinery required for the tag itself.
- PyPI publishing is explicitly out of scope per `spec.md § Out of Scope` (PyPI reserved for stable v0.3.0 once US5 lands).

**Alternatives considered**:

- **bumpver / setuptools-scm**: Adds a new tool surface; hatch is already integrated. Rejected.
- **Manual edit of pyproject.toml**: Error-prone (typos, version-string drift). Rejected.
- **Automated tag via GitHub Action**: Premature; the maintainer wants explicit control over the release commit. Rejected.

**Implementation scope**:

- Phase C ceremony (plan.md § Phased Milestones) details the exact command sequence.
- `docs/maintainers.md` § Release ceremony documents the procedure.

---

### ADR-011 — Reproducibility test pattern (snapshot tests)

**Status**: ACCEPTED.

**Decision**: Snapshot tests for visual-claim reproducibility (FR-023 → SC-005). Expected outputs stored under `tests/fixtures/expected/<test_name>.{html,png,json}`. Compare bytes for HTML and JSON; pixel-equal for PNG (Kaleido + Plotly are pinned per ADR-001, so pixel-equality is achievable). Snapshots regeneratable via documented maintainer procedure; CI never updates snapshots.

**Rationale**:

- Snapshot tests are the strongest guarantee for Principle IV (Reproducibility). The snapshot bytes ARE the artefact; comparison is unambiguous.
- Using stdlib `filecmp` and `hashlib` avoids adding `pytest-snapshot` as a dep, preserving the FR-027 install-footprint budget.
- Maintainer-only snapshot regeneration prevents CI from silently absorbing drift.

**Alternatives considered**:

- **`pytest-snapshot` library**: Adds a dep; provides marginal convenience. Rejected.
- **Functional tests (assert N files exist, assert types match)**: Weaker; can't catch palette drift or layout changes. Rejected.
- **Mocking the renderer**: Tests the test, not the artefact. Rejected.

**Implementation scope**:

- `tests/fixtures/expected/` populated by Phase A initial run.
- `tests/integration/test_visualisation_e2e.py` compares HTML byte-for-byte and PNG pixel-for-pixel.
- `docs/maintainers.md` § Snapshot maintenance documents the regeneration procedure.

---

### ADR-012 — Accessibility verification tooling

**Status**: ACCEPTED.

**Decision**: `colorspacious` Python library for colourblind palette verification (deuteranopia + protanopia simulation). Pure-Python, supports Python 3.11–3.14, declared under `[project.optional-dependencies]` group `dev`.

Verification script `scripts/verify_palette_colourblind.py`:

1. Render each palette colour as a sample swatch.
2. Apply `colorspacious` simulation for deuteranopia and protanopia.
3. Compute pairwise CIELAB ΔE distance between adjacent palette entries under each simulation.
4. Assert minimum ΔE ≥ threshold (decided below in Open Question 6: ΔE ≥ 15).
5. Emit `docs/assets/palette_colourblind_check.png` — a montage showing the palette under normal vision, deuteranopia simulation, and protanopia simulation.

`mkdocs-material` WCAG-AA verification: manual for this phase. Documented procedure in `docs/maintainers.md` § Accessibility verification (verify default theme contrast against WCAG-AA criteria; record any overrides). Full automated audit (axe-core, Pa11y, Lighthouse a11y) is explicitly deferred per `spec.md § Assumptions`.

**Rationale**:

- `colorspacious` is mature, well-cited (Nathan Moroney et al.), lightweight, pure-Python — keeps cross-Python compat trivial.
- Manual mkdocs-material verification is acceptable because the default theme is documented as AA-conformant; spec-02 doesn't introduce theme overrides, so risk is low.
- Deferring the full WCAG audit matches the spec's `Q5` clarification ("baseline a11y, not full audit").

**Alternatives considered**:

- **Browser-based simulation only (e.g., Coblis)**: Manual; not reproducible; can't fail CI. Rejected as primary; kept as supplementary.
- **`daltonize` / similar libraries**: Less mature; smaller community. Rejected.
- **Adding axe-core / Pa11y CI gate**: Out of scope per `Q5` decision. Deferred.

**Implementation scope**:

- `pyproject.toml` `dev` extras include `colorspacious`.
- `scripts/verify_palette_colourblind.py` produces the verification artefact.
- `tests/unit/test_palette_colourblind.py` asserts ΔE threshold.

---

## Open Questions Resolved

### Open Question 1 — Kaleido on Python 3.14 install footprint + compatibility

**Decision**: Kaleido 0.2.1+ supports Python 3.11 / 3.12 / 3.13 / 3.14 via pre-built wheels for `manylinux2014_x86_64`, `macosx_*`, `win_amd64`. Install footprint on a clean venv: ~50 MB unpacked (Chromium-headless binaries). The optional-extra placement (`pip install 'bin-packer-3d[viz]'`) keeps the basic install footprint unchanged.

**Verification method**: `pip download kaleido --no-deps --python-version 3.14 --dest /tmp/k` and `du -sh /tmp/k` on a clean checkout, recorded in `docs/maintainers.md` § Dependency footprints during Phase A.

**Risk handling**: If a future Kaleido release breaks Python 3.14 support, the CI matrix's 3.14 row would fail; pin to the last-known-good version in `pyproject.toml` (`kaleido>=0.2.1,<X.Y.Z`).

---

### Open Question 2 — `colorspacious` Python 3.14 support

**Decision**: `colorspacious` (latest 1.1.x at writing time) is pure-Python with `numpy` as its only runtime dep. It installs cleanly on Python 3.11–3.14 via `pip install colorspacious`.

**Verification method**: `pip install colorspacious` on a clean Python 3.14 venv; import `colorspacious.cspace_convert`. Recorded in `docs/maintainers.md` § Dependency footprints during Phase A.

**Risk handling**: If install fails on a future Python release, fallback to hand-curated palette swatch + manual browser-based verification artefact (Coblis output committed as image). The functional impact is unchanged.

---

### Open Question 3 — mkdocs-material WCAG-AA defaults verification

**Decision**: mkdocs-material's default Material theme (light variant) passes WCAG-AA contrast (4.5:1 for normal text, 3:1 for large text) per the theme's own documentation. The dark variant similarly passes. Spec-02 introduces NO theme overrides (no `extra.palette` customisation in `mkdocs.yml`), so the default conformance carries forward.

**Verification method**: Manual check during Phase B. Open the published docs site in a desktop browser; use the browser's accessibility tools (Chrome DevTools Lighthouse a11y, Firefox Accessibility Inspector) to confirm contrast ratios on the homepage, an algorithm page, and the API reference page. Record findings in `docs/maintainers.md` § Accessibility verification.

**Fallback if shortfall surfaces**: Add a minimal `extra.palette` override in `mkdocs.yml` that nudges the offending colours into AA range. Document the override.

---

### Open Question 4 — Plotly version-pin choice

**Decision**: `plotly>=5.18.0,<6.0.0`. The lower bound matches the existing pyproject specifier; the upper bound is the load-bearing constraint that prevents major-version palette / layout internals drift.

**Verification method**: Phase A baseline snapshot tests captured against the latest Plotly 5.x at install time. If a later 5.x minor release breaks snapshots, the test fails and the maintainer either updates snapshots (rare, documented) or pins more tightly.

**Risk handling**: If a future bugfix release of Plotly 5.x silently changes a palette default, snapshot tests catch it. Tighter pinning (e.g., `==5.24.*`) is a fallback if 5.x minor releases prove disruptive.

---

### Open Question 5 — Headline dataset generator algorithm choice

**Decision**: Bin-feasibility-driven recursive guillotine cuts. See ADR-009 sub-decision above.

**Verification method**: `tests/unit/test_dataset_generator.py` asserts (a) seed → byte-identical CSV, (b) generated dataset achieves ≥60% BFD utilisation on default bin dims, (c) runtime ≤5s.

---

### Open Question 6 — Colourblind ΔE threshold

**Decision**: ΔE ≥ 15 in CIELAB between adjacent palette entries under deuteranopia and protanopia simulation.

**Rationale**: Literature on colour-difference perception (Sharma 2017, Witzel & Gegenfurtner 2018) generally identifies ΔE ≈ 2.3 as the just-noticeable-difference threshold, ΔE ≈ 5–10 as "clearly different", and ΔE ≥ 15 as "easily distinguishable across viewers". For accessibility, ΔE ≥ 15 is the appropriate floor — it ensures that even simulated colourblind viewers can tell adjacent palette entries apart without effort.

**Verification method**: `scripts/verify_palette_colourblind.py` computes pairwise ΔE for the 12-entry Set3 palette under both simulations; asserts every pair ≥ 15.

**Fallback if the Set3 palette fails the threshold**: Drop to a more conservative palette (e.g., a hand-curated subset of Tableau colourblind 10) and re-verify. Document the swap.

---

### Open Question 7 — GitHub Pages first-deploy procedure

**Decision**: One-time manual maintainer action, documented in `docs/maintainers.md` § Pages setup:

1. Navigate to repo Settings → Pages.
2. Source → "GitHub Actions".
3. No custom domain (out of scope for spec-02).
4. Save.

After the source is set, the first push to `main` (or first manual workflow trigger) builds the docs and deploys to `https://bruno-ghiberto.github.io/3D_BIN_PACKING/`.

**Verification method**: First-deploy verified by hand during Phase B; subsequent deploys are automatic.

**Risk handling**: If the first deploy fails (permissions, settings drift), the workflow logs surface the exact GitHub error. The maintainer fixes settings and re-triggers. The live site stays at whatever was last successfully deployed (or empty if first deploy is failing).

---

## Summary Table

| Decision | Status | ADR | Open Question |
|---|---|---|---|
| Kaleido + Plotly pin | ACCEPTED | ADR-001 | OQ1, OQ4 |
| MkDocs Material + GitHub Pages + actions/deploy-pages | ACCEPTED | ADR-002 | OQ7 |
| Hybrid algorithm-page strategy | ACCEPTED | ADR-003 | — |
| Generated comparison table | ACCEPTED | ADR-004 | — |
| Generated Highlights section | ACCEPTED | ADR-005 | — |
| Plotly figure template | ACCEPTED | ADR-006 | — |
| BLAKE2b + ColorBrewer Set3 | ACCEPTED | ADR-007 | — |
| `bin-packer demo` Click subcommand | ACCEPTED | ADR-008 | — |
| Bin-feasibility-driven generator | ACCEPTED | ADR-009 | OQ5 |
| hatch version + manual tag | ACCEPTED | ADR-010 | — |
| Snapshot tests via stdlib | ACCEPTED | ADR-011 | — |
| colorspacious + manual mkdocs WCAG | ACCEPTED | ADR-012 | OQ2, OQ3, OQ6 |

All 12 ADRs ACCEPTED. All 7 open questions resolved. No `[NEEDS CLARIFICATION]` leaks into downstream artefacts.

---

**End of research.**
