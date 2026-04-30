"""Pre-commit / CI parity assertion (T032, FR-013).

Verifies that every hook configured in ``.pre-commit-config.yaml`` also runs
in CI via ``_ci-core.yml``, satisfying FR-013 ("contributors fail fast
locally"). Two complementary checks:

1. ``test_pre_commit_config_has_hooks`` — sanity, the config is non-empty.
2. ``test_ci_runs_pre_commit_parity`` — ``_ci-core.yml`` declares a
   ``pre-commit-parity`` job that runs ``pre-commit run --all-files``,
   guaranteeing every hook executes in CI.

Design choice: rather than asserting a literal one-to-one job-per-hook
mapping (which would couple CI job IDs to upstream hook IDs we don't
control — e.g. ``ruff`` vs. our ``lint`` job), we test that CI runs the
pre-commit suite end-to-end. Adding a new hook automatically gets CI
coverage with no test edit. The literal phrasing in tasks.md ("hook IDs
are a subset of jobs") is interpreted operationally: every hook fires
in CI because the parity job runs all hooks. Resolved per the
spec-01-enhancing implement-context §7 step 4 — low-stakes test-design
default; documented here in the commit body when this test was authored.

Exercises: FR-013.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    assert isinstance(loaded, dict), f"{path}: top-level YAML must be a mapping"
    return loaded


def test_pre_commit_config_has_hooks() -> None:
    """``.pre-commit-config.yaml`` must declare at least one hook."""
    config_path = _repo_root() / ".pre-commit-config.yaml"
    assert config_path.exists(), "missing .pre-commit-config.yaml"

    config = _load_yaml(config_path)
    repos = config.get("repos", [])
    assert isinstance(repos, list) and repos, ".pre-commit-config.yaml has no repos"

    hook_ids: list[str] = []
    for repo in repos:
        assert isinstance(repo, dict), "every entry under `repos:` must be a mapping"
        for hook in repo.get("hooks", []) or []:
            assert isinstance(hook, dict), "every hook entry must be a mapping"
            hook_id = hook.get("id")
            assert isinstance(hook_id, str) and hook_id, "every hook needs an `id:`"
            hook_ids.append(hook_id)

    assert hook_ids, ".pre-commit-config.yaml declares no hooks"


def test_ci_runs_pre_commit_parity() -> None:
    """``_ci-core.yml`` must declare a ``pre-commit-parity`` job that runs all hooks."""
    workflow_path = _repo_root() / ".github" / "workflows" / "_ci-core.yml"
    assert workflow_path.exists(), "missing .github/workflows/_ci-core.yml"

    workflow = _load_yaml(workflow_path)
    jobs = workflow.get("jobs")
    assert isinstance(jobs, dict) and jobs, "_ci-core.yml declares no jobs"

    assert "pre-commit-parity" in jobs, (
        "_ci-core.yml must declare a `pre-commit-parity` job (FR-013)"
    )

    parity_job = jobs["pre-commit-parity"]
    assert isinstance(parity_job, dict), "`pre-commit-parity` job must be a mapping"

    steps = parity_job.get("steps", [])
    assert isinstance(steps, list) and steps, (
        "`pre-commit-parity` job must declare at least one step"
    )

    run_commands: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        run_value = step.get("run")
        if isinstance(run_value, str):
            run_commands.append(run_value)

    joined = "\n".join(run_commands)
    assert "pre-commit run --all-files" in joined, (
        "`pre-commit-parity` must invoke `pre-commit run --all-files` "
        "to mirror local hooks (FR-013)"
    )
