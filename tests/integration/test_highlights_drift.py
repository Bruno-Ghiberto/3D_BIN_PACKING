"""README Highlights block drift guard (T027, US1 Independent-Test G-US1-IT-3).

Drives the ADR-005 contract: the ``<!-- BEGIN: HIGHLIGHTS -->`` block
in ``README.md`` is a deterministic projection of *live repository
state*. Its numeric facts MUST match what an independent probe of the
same sources reports:

- algorithm count → ``bin_packer_3d.algorithms.ALGORITHMS`` keys
- test count → ``pytest --collect-only -q``
- CI check count → non-aggregate jobs in ``.github/workflows/_ci-core.yml``
- supported Python versions → ``pyproject.toml`` classifiers
- license → ``pyproject.toml`` ``project.license``

This test probes each source independently of the regenerator (so a
regenerator bug cannot mask drift), then asserts each value is
present verbatim in the committed Highlights block.

RED until the full US1 chain lands:

- T028 — author the regenerator (Highlights generator + marker block).
- T031 — rewrite README with ``<!-- BEGIN: HIGHLIGHTS -->`` markers.
- T032 — run the regenerator to populate the markers.

Until T031 commits the markers, the test fails on the "markers
present" assertion with a clear pointer to the missing scaffold.

Source FRs: FR-015 (Highlights from live state). Source ADRs:
ADR-004 (marker delimiters), ADR-005 (Highlights regeneration source).
"""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

from bin_packer_3d.algorithms import ALGORITHMS

REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"
PYPROJECT = REPO_ROOT / "pyproject.toml"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "_ci-core.yml"

_HIGHLIGHTS_BLOCK = re.compile(
    r"<!-- BEGIN: HIGHLIGHTS -->(?P<body>.*?)<!-- END: HIGHLIGHTS -->",
    re.DOTALL,
)
_PY_CLASSIFIER = re.compile(r"^Programming Language :: Python :: (\d+\.\d+)$")


def _live_algorithm_count() -> int:
    return len(ALGORITHMS)


def _live_test_count() -> int:
    """Collect-only invocation of pytest from the repo root.

    Uses ``-q`` to keep output compact and ``--no-header`` so the parse
    only sees the trailing summary line, which is the stable contract:
    ``N tests collected in Xs``.
    """
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "--no-header"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    match = re.search(r"(\d+)\s+tests?\s+collected", result.stdout)
    assert match, f"could not parse test count from pytest output:\n{result.stdout}"
    return int(match.group(1))


def _live_ci_check_count() -> int:
    """Count non-aggregate jobs in the core CI workflow.

    ``aggregate`` is the meta-job that consumes the others' results;
    it is not a CI "check" in the badge-counting sense. Every other
    top-level key under ``jobs:`` IS a check.
    """
    with CI_WORKFLOW.open("rb") as fh:
        doc = yaml.safe_load(fh)
    jobs = doc["jobs"]
    return sum(1 for name in jobs if name != "aggregate")


def _live_python_versions() -> list[str]:
    """Parse ``Programming Language :: Python :: X.Y`` classifiers."""
    with PYPROJECT.open("rb") as fh:
        meta = tomllib.load(fh)
    classifiers = meta["project"]["classifiers"]
    versions = []
    for c in classifiers:
        m = _PY_CLASSIFIER.match(c)
        if m:
            versions.append(m.group(1))
    return versions


def _live_license_text() -> str:
    with PYPROJECT.open("rb") as fh:
        meta = tomllib.load(fh)
    lic = meta["project"]["license"]
    # PEP 621 license can be {"text": "..."} or {"file": "..."}; the
    # project uses the inline form.
    return lic["text"] if isinstance(lic, dict) and "text" in lic else str(lic)


def _highlights_block() -> str:
    content = README.read_text(encoding="utf-8")
    match = _HIGHLIGHTS_BLOCK.search(content)
    if not match:
        pytest.fail(
            "README missing <!-- BEGIN: HIGHLIGHTS --> / <!-- END: HIGHLIGHTS --> "
            "markers — implement T031 (README rewrite) before this gate is meaningful"
        )
    return match.group("body")


def test_highlights_block_present() -> None:
    """T031 contract: README has the HIGHLIGHTS marker block (ADR-004)."""
    content = README.read_text(encoding="utf-8")
    has_begin = "<!-- BEGIN: HIGHLIGHTS -->" in content
    has_end = "<!-- END: HIGHLIGHTS -->" in content
    assert has_begin, "missing <!-- BEGIN: HIGHLIGHTS --> marker (T031)"
    assert has_end, "missing <!-- END: HIGHLIGHTS --> marker (T031)"


def test_highlights_reports_algorithm_count() -> None:
    block = _highlights_block()
    n = _live_algorithm_count()
    assert str(n) in block, (
        f"Highlights block does not mention algorithm count {n} "
        f"(ALGORITHMS keys: {sorted(ALGORITHMS)})"
    )


def test_highlights_reports_test_count() -> None:
    block = _highlights_block()
    n = _live_test_count()
    assert str(n) in block, (
        f"Highlights block does not mention test count {n} (probed via `pytest --collect-only -q`)"
    )


def test_highlights_reports_ci_check_count() -> None:
    block = _highlights_block()
    n = _live_ci_check_count()
    assert str(n) in block, (
        f"Highlights block does not mention CI check count {n} "
        f"(non-aggregate jobs in {CI_WORKFLOW.relative_to(REPO_ROOT)})"
    )


def test_highlights_reports_python_versions() -> None:
    block = _highlights_block()
    versions = _live_python_versions()
    missing = [v for v in versions if v not in block]
    assert not missing, (
        f"Highlights block missing Python versions {missing} "
        f"(pyproject classifiers report {versions})"
    )


def test_highlights_reports_license() -> None:
    block = _highlights_block()
    license_text = _live_license_text()
    assert license_text in block, (
        f"Highlights block does not mention license {license_text!r} (pyproject project.license)"
    )
