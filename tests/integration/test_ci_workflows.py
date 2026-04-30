"""Workflow YAML-syntax validation (T031, FR-010, FR-016).

Asserts that every workflow file expected by the v0.2.0 CI bootstrap exists
under ``.github/workflows/``, parses as valid YAML, and declares a top-level
``name:`` field. A workflow that doesn't parse — or lacks a name — is invisible
in the GitHub Actions UI and silently fails to run; FR-010 ("CI MUST run on
every push and PR") is empty unless these mechanical preconditions hold.

Scope: structural validation only. Job semantics (matrix shape, action SHAs,
coverage threshold) are exercised by tests in this module's siblings and by
the ``aggregate``/``pre-commit-parity`` jobs themselves at runtime.

PyYAML 1.1 caveat: in YAML 1.1 the bare key ``on`` is parsed as the boolean
``True`` because of the legacy ``yes/no/on/off`` boolean shorthand. This test
only reads the ``name`` key, so the caveat does not affect us — but anyone
extending this module to inspect the trigger spec must access ``data[True]``,
not ``data["on"]``.

Exercises: FR-010, FR-016 (workflow plumbing), FR-014 (security workflow).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


_EXPECTED_WORKFLOWS: tuple[str, ...] = (
    "ci.yml",
    "_ci-core.yml",
    "pr-title.yml",
    "security.yml",
)


@pytest.mark.parametrize("workflow_name", _EXPECTED_WORKFLOWS)
def test_workflow_exists_parses_and_is_named(workflow_name: str) -> None:
    """Each expected workflow must exist, parse as YAML, and declare ``name:``."""
    path = _repo_root() / ".github" / "workflows" / workflow_name
    assert path.exists(), f"missing workflow file: .github/workflows/{workflow_name}"

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    assert isinstance(data, dict), (
        f".github/workflows/{workflow_name}: top-level YAML must be a mapping"
    )

    assert "name" in data, f".github/workflows/{workflow_name}: missing top-level `name:` field"

    name_value = data["name"]
    assert isinstance(name_value, str) and name_value.strip(), (
        f".github/workflows/{workflow_name}: `name:` must be a non-empty string"
    )
