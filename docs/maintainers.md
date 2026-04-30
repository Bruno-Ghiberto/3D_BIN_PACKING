# Maintainer Operations

Hand-off notes for actions that the implementing agent CANNOT take on its
own — typically because they require GitHub UI access, dashboard logins,
or destructive history rewrites. The agent prepares the ground; the
maintainer executes here.

This file is not user-facing documentation; it documents the privileged
levers a release maintainer needs to keep the project in a known-good
state. End users should look at the docs site once it is live (US3, T072).

---

## 1. Branch protection on `main` (T040, FR-017)

The `_ci-core.yml` workflow defines six gating jobs plus an aggregate;
`pr-title.yml` adds a seventh check; `security.yml` is informational. To
satisfy FR-017 ("a PR with a lint violation, a type error, or a failing
test MUST be blocked from merging"), branch protection on `main` must
require every gate from the dispatcher and PR-title workflows. CodeQL
stays advisory at v0.2.0.

### Required status checks

Configure these in **Settings → Branches → Branch protection rules → main**
under "Require status checks to pass before merging". The strings below are
the EXACT check-names GitHub reports — discovered empirically by running
PR #1 once. The `core /` prefix on most checks comes from `ci.yml`'s
`core:` job calling `_ci-core.yml` via `workflow_call`; do not omit it.

| Check name (paste into GitHub UI)       | Source workflow                  | Job ID inside the workflow      |
|-----------------------------------------|----------------------------------|---------------------------------|
| `core / Lint (ruff check)`              | `ci.yml` → `_ci-core.yml`        | `lint`                          |
| `core / Format (ruff format --check)`   | `ci.yml` → `_ci-core.yml`        | `format-check`                  |
| `core / Type check (mypy --strict)`     | `ci.yml` → `_ci-core.yml`        | `type-check`                    |
| `core / Tests (Python 3.11)`            | `ci.yml` → `_ci-core.yml`        | `test` (matrix: 3.11)           |
| `core / Dependency audit (pip-audit)`   | `ci.yml` → `_ci-core.yml`        | `pip-audit`                     |
| `core / Pre-commit (run --all-files)`   | `ci.yml` → `_ci-core.yml`        | `pre-commit-parity`             |
| `core / Aggregate gate result`          | `ci.yml` → `_ci-core.yml`        | `aggregate`                     |
| `Conventional Commits`                  | `pr-title.yml`                   | `validate`                      |

Notes:

- The matrix expands to Python `3.11 / 3.12 / 3.13 / 3.14` in T049
  (Invocation 10, v0.3.0). When that lands, add `core / Tests (Python 3.12)`,
  `core / Tests (Python 3.13)`, and `core / Tests (Python 3.14)` as
  additional required checks.
- `core / Aggregate gate result` is the OR-bar that downstream workflows
  read via `outputs.all_passed`; keeping it required protects against any
  individual gate being silently disabled.
- `CodeQL (Python)` from `security.yml` is **advisory**. Do NOT add it to
  required checks at v0.2.0 — false positives or rule churn would block
  merges. Re-evaluate at v1.0.0 release per the constitution's security
  review. (GitHub also reports a meta-check named just `CodeQL` from the
  Security tab; that one is also advisory and should not be added.)

### Other branch-protection rules to enable

In the same "Branch protection rules → main" form:

- **Require a pull request before merging** — yes; require approvals
  from `CODEOWNERS` (1 reviewer; the maintainer self-reviews until a
  co-maintainer joins).
- **Require linear history** — yes; preserves bisectability.
- **Require conversation resolution before merging** — yes; PR feedback
  must be addressed.
- **Restrict who can push to matching branches** — yes; only the
  CODEOWNERS team (currently `@Bruno-Ghiberto`).
- **Allow force pushes** — **no** (Constitution §Branching/commit
  conventions).
- **Allow deletions** — **no** (the branch is the ledger).

### Verifying the configuration

After applying the rules, open a draft PR that intentionally introduces
each violation in turn:

1. Add a `print()` in `src/bin_packer_3d/__init__.py` (lint will catch it
   under the `D`/`B` rule sets).
2. Add `def f(x): return x.bogus` (mypy will catch it under `--strict`).
3. Add `def test_red(): assert 0` (pytest will fail).

Each must independently block the PR from merging. This is **SC-014** —
record the proof in `docs/compliance/v1.0.0-audit.md` when assembling the
release-readiness audit.

---

## 2. Codecov repo activation (T038, T039)

Codecov is a free-tier service for public repos. Steps:

1. Sign in at <https://app.codecov.io> with the GitHub account that owns
   `Bruno-Ghiberto/3D_BIN_PACKING`.
2. Authorise the Codecov GitHub App on this repo (no organisation
   permission needed for personal repos).
3. Public repos do not require a token — `codecov-action` uploads
   anonymously by default. **Do not** add `CODECOV_TOKEN` as a repo
   secret unless this repo is moved into a private context; the workflow
   already references the secret with a no-op fallback.
4. Verify the first upload after the next CI run lands; the README badge
   resolves once Codecov has a baseline.

The flag is `library` (scoped to `src/bin_packer_3d/` per `codecov.yml`).
The 90 % project + patch target is hard-blocking once T049 enables
`--cov-fail-under=90` in pytest. Until then, Codecov surfaces coverage as
informational PR comments.

---

## 3. Reserved sections (deferred to later invocations)

These will be authored by future tasks; placeholders are listed here so
the structure of `maintainers.md` stays predictable:

- **PyPI OIDC trusted publisher** — T140, US8 release engineering.
  Will document Publisher / Owner / Repository / Workflow / Environment
  values for the dashboard at <https://pypi.org/manage/account/publishing/>.
- **GHCR image visibility** — T139, US8. Promote the package from
  "private" to "public" once the first image lands.
- **GitHub Pages source** — T072, US3. Switch Pages source to the
  `gh-pages` branch deployed by `mkdocs gh-deploy`.
- **Repository topics, description, homepage URL** — T157, Polish.

Each section will be filled in as those tasks land, with a back-reference
to the task ID and a one-paragraph maintenance contract.
