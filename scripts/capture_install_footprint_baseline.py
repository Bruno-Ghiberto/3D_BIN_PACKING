#!/usr/bin/env python3
"""Capture the install-footprint baseline for `tests/integration/test_install_footprint.py`.

Creates a clean virtual environment with the current Python interpreter,
performs a fresh `pip install` of the repository (no extras), and records
two `du -sb` measurements to `tests/fixtures/install_footprint_baseline.json`:

- `package_bytes`: size of the installed `site-packages/bin_packer_3d/`.
- `total_site_packages_bytes`: full size of the venv's `site-packages/`,
  including transitive deps.

The regression guard test asserts current measurements stay within
+5% of these values (FR-027 / SC-007).

Usage::

    python scripts/capture_install_footprint_baseline.py

**Choose the HIGHEST supported Python** when capturing the baseline.
Newer Python interpreters produce larger pre-compiled `.pyc` artefacts
at install time (per-version bytecode format). Capturing on the
highest version means every other version in the CI matrix
(3.11 / 3.12 / 3.13) measures fewer bytes and stays well below the
+5% budget — the regression guard then only fires on real dependency
bloat, never on cross-version `.pyc` variance.

Re-run whenever a deliberate dependency change widens the footprint
beyond the +5% budget. The commit that updates the baseline JSON should
explain why (e.g., "Plotly minor bump adds 2 MB; new headroom 3 MB").

Spec-02 Phase A task: T009.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPO_ROOT / "tests" / "fixtures" / "install_footprint_baseline.json"
TOLERANCE_PERCENT = 5


def _du_sb(target: Path) -> int:
    """Return `du -sb <target>` in bytes (apparent file-size sum)."""
    out = subprocess.run(
        ["du", "-sb", str(target)],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(out.stdout.split()[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=BASELINE_PATH,
        help=f"Output JSON path (default: {BASELINE_PATH.relative_to(REPO_ROOT)})",
    )
    args = parser.parse_args()

    py_version = platform.python_version()
    sys.stdout.write(f"Capturing install footprint with Python {py_version}...\n")
    sys.stdout.flush()

    with tempfile.TemporaryDirectory(prefix="footprint-baseline-") as tmp:
        venv_dir = Path(tmp) / "venv"
        subprocess.run(
            [sys.executable, "-m", "venv", str(venv_dir)],
            check=True,
            capture_output=True,
        )
        pip = venv_dir / "bin" / "pip"

        sys.stdout.write(f"Installing {REPO_ROOT} (no extras)...\n")
        sys.stdout.flush()
        subprocess.run(
            [str(pip), "install", "--quiet", str(REPO_ROOT)],
            check=True,
        )

        site_packages = next((venv_dir / "lib").glob("python*/site-packages"))
        package_bytes = _du_sb(site_packages / "bin_packer_3d")
        total_bytes = _du_sb(site_packages)

    payload = {
        "schema_version": 1,
        "python_version": py_version,
        "measured_at": date.today().isoformat(),
        "package_bytes": package_bytes,
        "total_site_packages_bytes": total_bytes,
        "tolerance_percent": TOLERANCE_PERCENT,
        "notes": (
            "Captured on the highest supported Python at the time of measurement. "
            "Lower CI matrix rows (3.11 / 3.12 / 3.13) install slightly smaller "
            "byte-compiled artefacts, so they stay comfortably below the +5% budget. "
            "Re-capture only on intentional dependency changes; the commit "
            "updating this file MUST explain the cause."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    sys.stdout.write(
        f"Wrote {args.out.relative_to(REPO_ROOT)}: "
        f"package={package_bytes:,} bytes, total={total_bytes:,} bytes "
        f"(+{TOLERANCE_PERCENT}% tolerance)\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
