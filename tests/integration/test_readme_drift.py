"""README drift guard (T025, US1 Independent-Test G-US1-IT-1 / G-US1-IT-4).

Drives the regenerator-idempotency contract in spec-02:

- ``python scripts/regenerate_readme.py`` MUST produce no changes to
  ``README.md`` when the committed README already reflects live repo
  state (the regenerator is a deterministic projection of repo state
  into three marker-delimited blocks per ADR-004).
- ``python scripts/regenerate_readme.py --check`` is the script's own
  self-test path (G-US1-IT-4) — same semantics, exits 0 when there is
  no drift and non-zero with a diff hint when there is.

These tests are RED until the full US1 chain lands:

- T028 — author the regenerator (currently a missing file → tests fail
  with ``FileNotFoundError`` once subprocess tries to invoke it).
- T031 — rewrite ``README.md`` with the three ``<!-- BEGIN: SECTION -->``
  ``/`` ``<!-- END: SECTION -->`` blocks the regenerator targets.
- T032 — run the regenerator once on the rewritten README and commit the
  populated output.

After T032, the regenerator is a no-op against the committed README;
drift only ever appears when live repo state changes (a new algorithm
registered, a CI job added, a Python version dropped, the test count
moving) without the README being regenerated. The CI catches that drift
at PR time.

Source FRs: FR-014 (README structure), FR-015 (badges from live state),
FR-016 (Project Structure block). Source ADRs: ADR-004 (marker
delimiters), ADR-005 (Highlights live-state source).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"
REGENERATOR = REPO_ROOT / "scripts" / "regenerate_readme.py"


def test_regenerator_script_exists() -> None:
    """T028 contract: ``scripts/regenerate_readme.py`` must exist.

    Red until T028 lands; turns green when the script is authored.
    """
    assert REGENERATOR.is_file(), (
        f"regenerator missing at {REGENERATOR.relative_to(REPO_ROOT)} — "
        "T028 of US1 is unimplemented"
    )


def test_regenerate_readme_is_idempotent() -> None:
    """G-US1-IT-1: regenerator produces no drift on a clean README.

    Reads ``README.md``, invokes the regenerator, re-reads, asserts
    byte-equal. Skips with a clear pointer if the regenerator does not
    yet exist (T028) so the test surfaces the missing dependency, not a
    confusing ``FileNotFoundError`` from ``subprocess``.
    """
    if not REGENERATOR.is_file():
        pytest.fail(
            "regenerator missing — implement T028 first; "
            f"expected at {REGENERATOR.relative_to(REPO_ROOT)}"
        )

    before = README.read_bytes()
    result = subprocess.run(
        [sys.executable, str(REGENERATOR)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"regenerator exited {result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    after = README.read_bytes()
    assert before == after, (
        "regenerator drifted README on a clean re-run — either the "
        "regenerator is non-deterministic or the committed README is "
        "stale; re-run `python scripts/regenerate_readme.py` and commit"
    )


def test_regenerate_readme_check_mode_succeeds() -> None:
    """G-US1-IT-4: ``--check`` mode equivalent to drift assertion.

    The regenerator's ``--check`` flag is the script's self-test: it
    computes what the regenerated README would be and exits 0 when that
    matches the committed README, non-zero otherwise. Equivalent to
    G-US1-IT-1 but exercises the script's own check path.
    """
    if not REGENERATOR.is_file():
        pytest.fail(
            "regenerator missing — implement T028 first; "
            f"expected at {REGENERATOR.relative_to(REPO_ROOT)}"
        )

    result = subprocess.run(
        [sys.executable, str(REGENERATOR), "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"`regenerate_readme.py --check` exited {result.returncode}\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
