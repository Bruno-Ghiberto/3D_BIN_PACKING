"""Install-footprint regression guard (FR-027, SC-007).

Asserts that a fresh `pip install` of `bin-packer-3d` (no extras) does
not exceed +5% of the committed baseline footprint. Marked `slow`
because it spins up a clean venv and installs the package plus its
transitive dependencies (~20-40 seconds depending on wheel cache).

Baseline source: `tests/fixtures/install_footprint_baseline.json`.
Captured by `scripts/capture_install_footprint_baseline.py` (T009).
The CI install-footprint job (T024) invokes this test by path, which
bypasses the default `-m "not slow"` marker filter.

Two budgets are enforced:

- `package_bytes`: size of the installed `site-packages/bin_packer_3d/`
  directory.
- `total_site_packages_bytes`: full size of the venv's `site-packages/`
  including transitive deps.

Both must satisfy `current <= baseline * 1.05`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = REPO_ROOT / "tests" / "fixtures" / "install_footprint_baseline.json"
TOLERANCE = 1.05  # +5% per FR-027 / SC-007


def _measure(site_packages: Path) -> tuple[int, int]:
    """Return (package_bytes, total_site_packages_bytes) via du -sb.

    Matches the task description (T009) literally. `du -sb` reports
    apparent byte size, which is what we want for a portable footprint
    metric (independent of filesystem block size).
    """
    pkg_dir = site_packages / "bin_packer_3d"
    pkg_out = subprocess.run(
        ["du", "-sb", str(pkg_dir)],
        capture_output=True,
        text=True,
        check=True,
    )
    total_out = subprocess.run(
        ["du", "-sb", str(site_packages)],
        capture_output=True,
        text=True,
        check=True,
    )
    pkg_bytes = int(pkg_out.stdout.split()[0])
    total_bytes = int(total_out.stdout.split()[0])
    return pkg_bytes, total_bytes


@pytest.fixture(scope="module")
def baseline() -> dict[str, int | str]:
    """Load the committed baseline JSON.

    Fails the test if the baseline is missing — the regression guard is
    meaningless without a reference point. Re-generate via
    `python scripts/capture_install_footprint_baseline.py`.
    """
    if not BASELINE_PATH.exists():
        pytest.fail(
            f"Install-footprint baseline missing: {BASELINE_PATH}. "
            "Regenerate via `python scripts/capture_install_footprint_baseline.py`."
        )
    return json.loads(BASELINE_PATH.read_text())


@pytest.mark.slow
@pytest.mark.integration
def test_install_footprint_within_baseline(baseline: dict[str, int | str], tmp_path: Path) -> None:
    """A fresh install must not exceed +5% of the recorded baseline."""
    venv_dir = tmp_path / "footprint-venv"
    subprocess.run(
        [sys.executable, "-m", "venv", str(venv_dir)],
        check=True,
        capture_output=True,
    )
    pip = venv_dir / "bin" / "pip"
    subprocess.run(
        [str(pip), "install", "--quiet", str(REPO_ROOT)],
        check=True,
        capture_output=True,
    )

    site_packages = next((venv_dir / "lib").glob("python*/site-packages"))
    current_pkg, current_total = _measure(site_packages)

    baseline_pkg = int(baseline["package_bytes"])
    baseline_total = int(baseline["total_site_packages_bytes"])
    budget_pkg = int(baseline_pkg * TOLERANCE)
    budget_total = int(baseline_total * TOLERANCE)

    assert current_pkg <= budget_pkg, (
        f"Package footprint regression: site-packages/bin_packer_3d/ "
        f"is {current_pkg} bytes, exceeds budget {budget_pkg} "
        f"(baseline {baseline_pkg} + 5%)."
    )
    assert current_total <= budget_total, (
        f"Total install footprint regression: site-packages/ "
        f"is {current_total} bytes, exceeds budget {budget_total} "
        f"(baseline {baseline_total} + 5%)."
    )
