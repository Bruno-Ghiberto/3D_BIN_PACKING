"""Headline dataset generator regression tests (FR-034 / SC-005 / SC-006).

Three assertions against `scripts/generate_headline_dataset.py`:

1. **Byte-identical regeneration** — running the generator twice with the
   same seed file produces the same CSV bytes (Principle IV).
2. **BFD utilisation floor** — running BFD on the generated dataset against
   the default 860x890x1040 mm bin yields overall volume utilisation
   >= 60% (FR-034 acceptance criterion).
3. **Runtime budget** — generation completes in <= 5 seconds on commodity
   hardware.

The generator is invoked as a CLI script (subprocess) so the test exercises
the documented command-line contract from
`specs/002-portfolio-polish/contracts/headline-dataset.md`. The script and
seed file are authored in T012 + T013; this test is committed in a red
state (T011) before either lands.
"""

from __future__ import annotations

import filecmp
import subprocess
import sys
import time
from pathlib import Path

from bin_packer_3d.algorithms.bfd import BestFitDecreasingPacker
from bin_packer_3d.config import PackerConfig
from bin_packer_3d.data.loaders import load_boxes_from_csv

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR_SCRIPT = REPO_ROOT / "scripts" / "generate_headline_dataset.py"
SEED_FILE = REPO_ROOT / "examples" / "headline.seed"

# Default bin per the project (matches PackerConfig defaults).
BIN_LENGTH = 860.0
BIN_WIDTH = 890.0
BIN_HEIGHT = 1040.0
BIN_VOLUME = BIN_LENGTH * BIN_WIDTH * BIN_HEIGHT

UTILISATION_FLOOR = 0.60
RUNTIME_BUDGET_S = 5.0


def _run_generator(out_path: Path) -> None:
    """Invoke the generator CLI; surface stdout/stderr on failure."""
    subprocess.run(
        [
            sys.executable,
            str(GENERATOR_SCRIPT),
            "--seed",
            str(SEED_FILE),
            "--out",
            str(out_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_byte_identical_regeneration(tmp_path: Path) -> None:
    """Same seed file -> byte-identical CSV (Principle IV)."""
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    _run_generator(first)
    _run_generator(second)
    assert filecmp.cmp(first, second, shallow=False), (
        "Generator output is not byte-identical across two runs with the "
        "same seed file. Principle IV (Reproducibility) is violated."
    )


def test_bfd_utilisation_meets_floor(tmp_path: Path) -> None:
    """Generated dataset achieves >= 60% BFD utilisation on default bin dims (FR-034)."""
    out = tmp_path / "headline.csv"
    _run_generator(out)

    report = load_boxes_from_csv(str(out))
    boxes = list(report.boxes)
    assert boxes, "Generator emitted zero boxes — empty dataset"

    config = PackerConfig(bin_length=BIN_LENGTH, bin_width=BIN_WIDTH, bin_height=BIN_HEIGHT)
    packer = BestFitDecreasingPacker(config)
    result = packer.pack(boxes)

    total_bin_volume = result.bins_used * BIN_VOLUME
    packed_volume = sum(p.box.width * p.box.height * p.box.length for p in result.placements)
    utilisation = packed_volume / total_bin_volume if total_bin_volume else 0.0

    assert utilisation >= UTILISATION_FLOOR, (
        f"BFD utilisation {utilisation:.2%} is below the FR-034 floor of "
        f"{UTILISATION_FLOOR:.0%}. Re-tune target_utilisation in "
        f"examples/headline.seed (recommended: 0.65)."
    )


def test_runtime_within_budget(tmp_path: Path) -> None:
    """Generator completes in <= 5 seconds (FR-034 budget)."""
    out = tmp_path / "headline.csv"
    start = time.monotonic()
    _run_generator(out)
    elapsed = time.monotonic() - start
    assert elapsed <= RUNTIME_BUDGET_S, (
        f"Generator took {elapsed:.2f}s, exceeds the {RUNTIME_BUDGET_S}s budget. "
        "Profile the algorithm or I/O — likely a regression."
    )
