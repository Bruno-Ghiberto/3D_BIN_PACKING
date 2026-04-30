# Phase 0 Research: Public-Release Hardening

**Feature**: `001-public-release-hardening`
**Date**: 2026-04-23
**Input**: [plan.md](./plan.md), [spec.md](./spec.md), [constitution v1.0.0](../../.specify/memory/constitution.md)

This document resolves every technology / architecture unknown surfaced by the
Technical Context in `plan.md`. Each decision is recorded as an Architecture
Decision Record (ADR) with Status = **Accepted** at plan time. The ADRs also
satisfy Spec FR-024 (≥ 3 ADRs at `docs/adr/`) — the Phase C implementation
commits copy these ADRs into `docs/adr/NNNN-<slug>.md` files with matching
numbering.

---

## ADR-0001 — Algorithm Registry Pattern

**Status**: Accepted
**Context**: Spec US1 (Contract Integrity) requires the set of accepted
`PackerConfig.strategy` values, the CLI `info` listing, and the documented
algorithms to be the same set — sourced from a single registry (FR-001, FR-002,
FR-003). The current codebase has `PackerConfig.strategy` as a hardcoded
`Literal["ffd", "bfd", "extreme_points"]` that does not match the implemented
set (`ffd`, `shelf`).

**Decision**: Implement a module-level registry in
`src/bin_packer_3d/algorithms/__init__.py`:

```python
ALGORITHMS: dict[str, type[BasePacker]] = {}

def register(name: str) -> Callable[[type[BasePacker]], type[BasePacker]]:
    def wrapper(cls: type[BasePacker]) -> type[BasePacker]:
        if name in ALGORITHMS:
            raise ValueError(f"algorithm '{name}' already registered")
        ALGORITHMS[name] = cls
        return cls
    return wrapper

def get_strategies() -> tuple[str, ...]:
    return tuple(sorted(ALGORITHMS))
```

`PackerConfig.strategy` becomes:

```python
strategy: str = Field(default="ffd")

@field_validator("strategy")
@classmethod
def _strategy_must_be_registered(cls, v: str) -> str:
    if v not in ALGORITHMS:
        raise ValueError(
            f"unknown strategy '{v}'. Registered: {sorted(ALGORITHMS)}"
        )
    return v
```

**Rationale**: A dict registry + pydantic validator trivially makes the
`PackerConfig` schema and the runtime agree — they read from the same source.
An automated test (`tests/unit/test_registry.py`) asserts this equality on
every CI run (FR-006). Adding a new algorithm is a single `@register("name")`
decorator — no config or CLI edit required (FR-002).

**Alternatives considered**:

- **Static `Literal` of algorithm names**: rejected. Defeats the purpose — any
  code change that adds an algorithm must also edit the `Literal`, reintroducing
  the drift the spec aims to eliminate.
- **Entry-point discovery via `importlib.metadata.entry_points`**: rejected as
  over-engineered for v1.0.0 — adds packaging complexity (users installing
  plugins) that exceeds the spec's scope. Can be layered on top later.

---

## ADR-0002 — Documentation Tooling

**Status**: Accepted
**Context**: FR-020, FR-021 require a documentation site generated from the
repository, published on every push to `main`, containing problem intro,
quickstart, per-algorithm reference, constraints reference, benchmark page,
auto-generated API reference, contributing guide. Spec §Assumptions allows
either MkDocs Material or Sphinx.

**Decision**: **MkDocs Material** with the `mkdocstrings[python]` plugin.

**Rationale**:
- **Markdown-first** authoring matches the existing README style; no RST or
  MyST conversion overhead for a single-maintainer project.
- **`mkdocstrings`** generates API reference directly from docstrings — no
  separate build step, no `autodoc` invocation.
- **Built-in search** (lunr.js) — no external service.
- **`mkdocs gh-deploy`** — one-command GitHub Pages deploy; no webhook setup.
- **Themed out of the box** — dark/light mode, mobile responsive.
- **Wide ecosystem** — `mkdocs-material` is the de facto standard for Python
  library docs (FastAPI, pydantic, typer, rich).

**Alternatives considered**:

- **Sphinx + Furo + `sphinx-autodoc`**: rejected. More powerful (intersphinx,
  docutils extensions, automated cross-references) but requires RST or MyST.
  The added power is not needed here; the ergonomic tax is not worth paying.
- **pdoc3 / pdoc**: rejected. API-only tools. The spec's documentation
  requirements include prose pages (problem intro, algorithm guidance,
  benchmarks) that pdoc does not handle well.

---

## ADR-0003 — CI Workflow Architecture

**Status**: Accepted
**Context**: FR-010, FR-011, FR-012, FR-017 require CI on every push and PR,
multi-Python matrix (3.11–3.14), coverage reporting, merge blocking on gate
failure. The project is pure Python, single-maintainer, on a free tier.

**Decision**: Adopt the thin-dispatcher + reusable-core pattern battle-tested
in `The-Embedinator`:

**Files**:
- `.github/workflows/ci.yml` — minimal dispatcher. Triggers: `push` to `main`,
  `pull_request` (opened, synchronize, reopened, ready_for_review). Calls
  `_ci-core.yml` via `workflow_call`. Concurrency group
  `ci-${{ github.ref }}` with `cancel-in-progress: true`.
- `.github/workflows/_ci-core.yml` — reusable core. All job logic lives here.
  Called from `ci.yml` (on push / PR) and `release.yml` (on tag). Emits
  `outputs.all_passed: true/false`.
- `.github/workflows/release.yml` — tag trigger. Calls `_ci-core.yml` as a
  blocking gate; publish jobs only run if `all_passed == 'true'`.
- `.github/workflows/security.yml` — CodeQL Python, weekly + on push. ADVISORY.
- `.github/workflows/pr-title.yml` — Conventional Commits enforcement.
- `.github/workflows/benchmark-br1.yml` — per-push BR1 benchmark + artefact.

**Required jobs in `_ci-core.yml`** (Python matrix 3.11 / 3.12 / 3.13 / 3.14
on `ubuntu-latest`):

| Job | Command | Required |
|---|---|---|
| `lint` | `ruff check src/ tests/ --output-format=github` | YES |
| `format-check` | `ruff format --check src/ tests/` | YES |
| `type-check` | `mypy --strict src/bin_packer_3d/` | YES |
| `test` (matrix) | `pytest tests/ --cov=src/bin_packer_3d --cov-report=xml --cov-fail-under=90 -m "not slow"` | YES |
| `pip-audit` | `pip-audit --requirement requirements.txt` | YES |
| `docs-build` | `mkdocs build --strict` | YES (Phase C onwards) |
| `pre-commit-parity` | `pre-commit run --all-files` | YES |
| `aggregate` | computes `all_passed` from required jobs | — |

**Pinned actions**: every `uses:` reference uses a full commit SHA with a
version comment (`actions/checkout@<sha> # v4.3.1`). Dependabot updates the
SHAs weekly.

**Coverage**: `codecov/codecov-action` uploads `coverage.xml` with flag
`library`; threshold enforced in `codecov.yml`.

**Rationale**: The dispatcher/core split is copied from a project where it has
already survived refactors and multi-wave rollouts. The `aggregate` job with
`if: always()` guarantees the caller always receives an `all_passed` boolean,
even when an earlier job failed — critical for downstream gating.

**Alternatives considered**:

- **Single flat workflow**: rejected. As jobs grow, release and security
  workflows would need to duplicate the gate logic.
- **Tox-driven matrix**: rejected. GitHub Actions matrix gives better
  observability (per-version status checks on PRs); tox adds a local-vs-CI
  drift risk.

---

## ADR-0004 — Documentation Host

**Status**: Accepted
**Context**: FR-020 requires the documentation site to publish on every push
to `main`. Spec §Assumptions allows any free-tier host.

**Decision**: **GitHub Pages** via `mkdocs gh-deploy --force` in a dedicated
`docs-deploy` job within `release.yml` (and an auxiliary `docs.yml` that
deploys on every push to `main`, not only on release tags — so `main`'s docs
are always current).

**Rationale**:
- **Zero additional infrastructure**; the repo is already on GitHub.
- **No external accounts** — unlike Read the Docs which requires signup and
  webhooks.
- **Custom domain support** if ever needed (not at v1.0.0).
- **Built-in HTTPS**, no cert management.
- **`gh-pages` branch workflow** — auto-created on first `mkdocs gh-deploy`.

**Alternatives considered**:

- **Read the Docs (free tier)**: rejected. External account overhead; the
  webhook + RTD config split introduces drift risk for a solo maintainer.
- **Netlify / Vercel**: rejected. Static site hosts optimised for JS apps;
  the Python-docs flow is simpler on GH Pages.

---

## ADR-0005 — Performance Budget Methodology

**Status**: Accepted
**Context**: Constitution Principle VIII ("Performance Discipline")
explicitly requires optimisations be justified by measurement and prescribes
an optimisation ladder (better algorithm → NumPy vectorisation → JIT → native
extension). Spec §Out of Scope defers native extensions entirely.

**Decision**: **Measure-first**. No numeric performance target is committed
before a baseline benchmark is captured. The Phase B deliverable is a baseline
against Bischoff & Ratcliff BR1 for every registered algorithm; subsequent
optimisation work (if any) uses that baseline as the lower bound.

**Rationale**:
- Principle VIII is explicit: "Premature optimization — an optimization
  shipped without a profile proving it targets a real hot path — is a merge
  blocker."
- The project is NP-hard; meaningful perf claims require a reproducible
  reference instance (Principle IV).
- Bischoff & Ratcliff BR1 is the de facto 3D-BPP reference instance.

**Enforcement**:
- `benchmark/run_baseline.py` captures the first baseline; results live at
  `docs/benchmarks/results/baseline.json`.
- Any PR adding a performance optimisation must cite the baseline in its
  description and attach a new benchmark artefact.
- Regression > 25 % utilisation drop or 25 % runtime increase surfaces a CI
  warning (Constitution §Benchmark workflow).

**Alternatives considered**:

- **Set provisional targets by domain convention** (e.g. "pack 50 boxes in
  500 ms"): rejected. Contradicts Principle VIII; targets unbacked by
  measurement are folklore.

---

## ADR-0006 — PyPI Package Name & Distribution

**Status**: Accepted
**Context**: FR-070, FR-071, FR-072 require PyPI publishing via OIDC trusted
publisher, `pipx install <name>` installability, and a published Docker
image. Spec §Assumptions allows a fallback name if `bin-packer-3d` is taken.

**Decision**: Reserve `bin-packer-3d` on PyPI immediately (Phase A, first
action). OIDC trusted publisher configured on PyPI dashboard:

- **Publisher**: GitHub Actions
- **Owner**: `Bruno-Ghiberto`
- **Repository**: `3D_BIN_PACKING`
- **Workflow**: `release.yml`
- **Environment**: `pypi`

**Fallback priority**:

1. `bin-packer3d`
2. `binpacker3d`
3. `bin-packing-3d`
4. `bin-packer-3d-ghiberto` (last resort)

**Docker image** (FR-072): published to GitHub Container Registry at
`ghcr.io/bruno-ghiberto/bin-packer-3d`. Multi-stage Dockerfile:

```dockerfile
# Stage 1: builder
FROM python:3.11-slim AS builder
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip wheel --no-deps --wheel-dir /wheels .

# Stage 2: runtime (non-root)
FROM python:3.11-slim
RUN useradd -m -u 1000 app
USER app
WORKDIR /home/app
COPY --from=builder /wheels/*.whl /tmp/
RUN pip install --user --no-cache-dir /tmp/*.whl && rm /tmp/*.whl
ENTRYPOINT ["bin-packer"]
CMD ["info"]
```

**Rationale**:
- OIDC eliminates long-lived token risk (Constitution §Security).
- GHCR is free-tier, GitHub-native; no external registry accounts.
- Multi-stage build keeps the final image minimal; non-root satisfies FR-072.

**Alternatives considered**:

- **Docker Hub**: rejected. Free-tier rate limits on anonymous pulls degrade
  user experience; GHCR has no such limits for public images.
- **Long-lived `PYPI_API_TOKEN`**: rejected. Explicitly forbidden by
  Constitution §Release workflow and spec FR-070.

---

## ADR-0007 — Constraint Framework

**Status**: Accepted
**Context**: US7 (FR-060..FR-064) requires a constraint abstraction that every
algorithm consults, supporting per-box allowed orientations and per-box
maximum supported weight, extensible to a future stability constraint without
changing the base algorithm interface.

**Decision**: **Constraint ABC + registry + visitor contract on the algorithm
base class**.

**Shape**:

```python
# src/bin_packer_3d/constraints/base.py
class Constraint(ABC):
    """A rule consulted before accepting a candidate placement."""

    @abstractmethod
    def check(
        self,
        placement: Placement,
        box: Box,
        bin: Bin,
        existing_placements: Sequence[Placement],
    ) -> ConstraintResult:
        """Return PASS or a rejection with a reason string."""
```

```python
# src/bin_packer_3d/algorithms/base.py  (existing BasePacker extended)
class BasePacker(ABC):
    def _accept(self, placement, box, bin, placements) -> bool:
        for c in self.config.constraints:
            if not c.check(placement, box, bin, placements).ok:
                return False
        return True
```

`PackerConfig.constraints: list[Constraint] = Field(default_factory=list)` —
user passes constraint instances at config time. Built-in constraints:

- `AllowedOrientations(box_id=..., orientations={"flat", "upright"})` — FR-061
- `SupportedWeight()` — inspects `Box.max_supported_weight`, rejects stacks
  whose cumulative overhead weight exceeds the support limit (FR-062)

**Rationale**:
- `pydantic` is already a runtime dep; `Constraint` instances serialise
  cleanly when carried on `PackerConfig`.
- The visitor shape (`check(placement, ...)`) is a narrow interface; adding
  a future `StabilityConstraint` requires no changes to any algorithm's
  placement loop (FR-063).
- Rejection reasons are surfaced to the `--explain` trace (US6), giving
  users an actionable debugging signal.

**Alternatives considered**:

- **Monkey-patching the `BasePacker.accept` method per algorithm**: rejected.
  Violates Open/Closed — new constraints would require algorithm edits.
- **Full plugin system with entry points**: rejected. Over-engineered; the
  spec's constraint set is small and known.
- **Subclass-based composition (mixins)**: rejected. Multiple inheritance in
  Python is brittle; visitor pattern is less fragile.

---

## ADR-0008 — Observability Stack

**Status**: Accepted
**Context**: US6 (FR-050..FR-054) requires structured, controllable logging
at the library level and an `--explain` mode at the CLI level. Constitution
V mandates stdlib `logging` with `NullHandler` and no global side-effects.

**Decision**: **stdlib `logging` with a thin `StructuredAdapter`** — zero new
runtime dependencies.

**Shape**:

```python
# src/bin_packer_3d/observability.py
import logging

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(f"bin_packer_3d.{name}")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger

class StructuredAdapter(logging.LoggerAdapter):
    """Adapter that adds extra fields to LogRecord for machine readers."""

    def process(
        self, msg: str, kwargs: MutableMapping[str, Any]
    ) -> tuple[str, MutableMapping[str, Any]]:
        extra = kwargs.setdefault("extra", {})
        if "fields" in extra:
            extra.update(extra.pop("fields"))
        return msg, kwargs
```

**Events emitted** (DEBUG unless noted):
- `packing.start` — algorithm, box count, bin dimensions
- `packing.end` — success rate, runtime, placements placed
- `packing.attempt` — per-box attempt (`--explain` trace)
- `packing.reject` — per-box rejection with constraint reason
- `loader.warning` (WARNING) — malformed row, with row number + reason

**CLI verbosity mapping** (FR-051):
- `--quiet` → `WARNING`
- default → `INFO`
- `--verbose` → `DEBUG`

**Rationale**:
- Principle V forbids heavyweight logging frameworks as runtime deps.
- `structlog` and `loguru` would force downstream consumers onto those
  frameworks — a library citizenship violation.
- stdlib `logging.LoggerAdapter` with `extra` fields gives structured output
  to any consumer that reads `LogRecord.__dict__` (e.g. JSON formatters).

**Alternatives considered**:

- **`structlog`**: rejected. Runtime dep; consumers inherit its initialisation
  conventions.
- **`loguru`**: rejected. Monkey-patches the stdlib logger; Principle V
  violation.

---

## ADR-0009 — Coordinate Convention

**Status**: Accepted (ratified in Constitution v1.0.0 §Technology Baseline)
**Context**: Non-obvious and easy to misuse; future algorithms cannot
redefine it without breakage.

**Decision**: `X = length`, `Y = width`, `Z = height`. Locked in the
constitution; changing requires a MAJOR amendment and a migration plan.

**Rationale**:
- Matches the existing `Box` and `Bin` attribute order in the codebase.
- Z-up is the convention in most 3D packing literature (Bischoff & Ratcliff;
  Crainic, Perboli, Tadei); keeps citations consistent.
- Documented in `docs/problem.md` with a diagram as part of Phase C (FR-021).

**Alternatives considered**:

- **Y-up (Plotly default)**: rejected. Forces a mental flip against
  the literature; increases reviewer cognitive load.

---

## ADR-0010 — Algorithm Portfolio

**Status**: Accepted
**Context**: FR-040 requires Best-Fit Decreasing, Extreme Point, and at least
one additional family (Maximal Rectangles, Skyline, or Layer-building).
FR-006 requires CI to assert the accepted strategies equal the registry
keys. SC-006 requires at least one new algorithm to show ≥ 5 percentage
points utilisation delta OR ≥ 1 bin delta from FFD on a bundled instance.

**Decision**: **Implement BFD, Extreme Point, and Maximal Rectangles**. Keep
existing FFD and Shelf. Total portfolio at `v0.3.0`:

| Strategy key | Family | Status | File |
|---|---|---|---|
| `ffd` | First-Fit Decreasing (volume) | existing — register | `algorithms/ffd.py` |
| `shelf` | Shelf-based | existing — register | `algorithms/shelf.py` |
| `bfd` | Best-Fit Decreasing | NEW Phase B | `algorithms/bfd.py` |
| `extreme_point` | Extreme-Point (Crainic 2008) | NEW Phase B | `algorithms/extreme_point.py` |
| `maximal_rectangles` | Maximal Rectangles | NEW Phase B | `algorithms/maximal_rectangles.py` |

**Rationale**:
- **BFD** is explicitly named in FR-040 and is the natural companion to the
  existing FFD — same sort key, different placement rule; shares utilities.
- **Extreme Point** (Crainic, Perboli, Tadei 2008) is explicitly named in
  FR-040 and is the canonical 3D-BPP heuristic in the OR literature.
- **Maximal Rectangles** is the third family — chosen over Skyline and
  Layer-building because:
  - It is 3D-native (Skyline generalises awkwardly from 2D).
  - It gives a distinctly different packing quality profile; meeting SC-006
    (≥ 5 pp utilisation delta or ≥ 1 bin delta) is more likely.
  - Reference implementations and pseudo-code are widely published.

**Per-algorithm acceptance**:
- Each implementation has a dedicated unit test file with algorithm-specific
  cases.
- All algorithms are enrolled in `tests/property/test_invariants.py` via
  parametrisation over `ALGORITHMS` — FR-046.
- Each algorithm has a `docs/algorithms/<name>.md` reference page with
  complexity, pseudo-code, citation, and guidance.

**Alternatives considered**:

- **Skyline only as third family**: rejected. Generalises awkwardly from
  2D; implementation cost high for uncertain SC-006 payoff.
- **Layer-building only as third family**: rejected. Good utilisation
  profile but less distinct from existing Shelf; weaker OR-signal for
  reviewers.
- **Add all five** (Skyline + Layer-building + Maximal Rectangles):
  rejected for v1.0.0. Scope creep; each algorithm adds test + doc work.
  Deferred to a future spec.

---

## Research summary

| Unknown (from plan.md Technical Context) | Decision | ADR |
|---|---|---|
| How does `PackerConfig.strategy` stay in sync with registered algorithms? | Dict registry + pydantic validator | 0001 |
| Which documentation tool? | MkDocs Material + mkdocstrings | 0002 |
| What CI workflow architecture? | Thin dispatcher + reusable core | 0003 |
| Where do the docs live? | GitHub Pages via `mkdocs gh-deploy` | 0004 |
| How are perf budgets set? | Measure-first — no targets before baseline | 0005 |
| What is the PyPI name and publish method? | `bin-packer-3d` + OIDC trusted publisher + GHCR Docker | 0006 |
| What shape does the constraint framework take? | `Constraint` ABC + registry, visitor on `BasePacker` | 0007 |
| What is the observability stack? | stdlib `logging` + `StructuredAdapter` | 0008 |
| What coordinate convention? | X = length, Y = width, Z = height (locked in constitution) | 0009 |
| Which algorithms ship at v0.3.0? | FFD + Shelf (existing) + BFD + Extreme Point + Maximal Rectangles (new) | 0010 |

No `NEEDS CLARIFICATION` remains. Phase 1 may proceed.
