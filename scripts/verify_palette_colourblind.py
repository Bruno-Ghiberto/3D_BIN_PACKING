#!/usr/bin/env python3
"""Verify SET3 colourblind safety and emit the verification montage (ADR-012).

Procedure:

1. Read the canonical SET3 palette from
   :mod:`bin_packer_3d.visualization.palette`.
2. Apply :mod:`colorspacious` simulation for **deuteranopia** and
   **protanopia** (severity=100 per ADR-012).
3. Compute the pairwise CIELAB ΔE between **adjacent** palette entries
   under each simulation; assert min ΔE ≥ 15 (or fail with the offending
   pair).
4. Render a 3-row montage (normal vision / deuteranopia / protanopia)
   with hex labels via :mod:`matplotlib` and save to
   ``docs/assets/palette_colourblind_check.png``.

Run modes:

- Default: render the PNG and run the ΔE assertion. Maintainer re-runs
  after any SET3 reordering (T020 commits the resulting artefact).
- ``--check-only``: skip rendering; only re-verify ΔE thresholds.
  Used by the Phase A "Foundation Ready" gate and by anyone who wants
  a fast acceptance check without touching matplotlib.

Dependencies: ``colorspacious`` (dev extras); ``matplotlib`` (for the
PNG montage — required only when rendering, not for ``--check-only``).

Spec-02 Phase A task: T018.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from colorspacious import cspace_convert

from bin_packer_3d.visualization.palette import SET3

DELTA_E_FLOOR = 15.0
DEFAULT_OUT = (
    Path(__file__).resolve().parents[1] / "docs" / "assets" / "palette_colourblind_check.png"
)

CVD_TYPES: tuple[str, ...] = ("deuteranomaly", "protanomaly")
CVD_LABELS: dict[str, str] = {
    "deuteranomaly": "Deuteranopia (severity 100)",
    "protanomaly": "Protanopia (severity 100)",
}


def _hex_to_rgb01(hex_str: str) -> tuple[float, float, float]:
    """Convert "#RRGGBB" -> (r, g, b) in [0, 1]."""
    hex_str = hex_str.lstrip("#")
    return (
        int(hex_str[0:2], 16) / 255.0,
        int(hex_str[2:4], 16) / 255.0,
        int(hex_str[4:6], 16) / 255.0,
    )


def _palette_rgb01() -> np.ndarray:
    """Return the SET3 palette as an (N, 3) array of sRGB floats."""
    return np.array([_hex_to_rgb01(c) for c in SET3])


def _simulate_cvd(rgb: np.ndarray, cvd_type: str) -> np.ndarray:
    """Return the palette as it appears under the given colour-vision deficiency."""
    return np.asarray(
        cspace_convert(
            rgb,
            {"name": "sRGB1+CVD", "cvd_type": cvd_type, "severity": 100},
            "sRGB1",
        )
    )


def _adjacent_min_delta_e(rgb: np.ndarray) -> tuple[float, int]:
    """Min adjacent CIELAB ΔE and the index of the weakest pair (left side)."""
    lab = np.asarray(cspace_convert(rgb, "sRGB1", "CIELab"))
    diffs = np.diff(lab, axis=0)
    distances = np.sqrt(np.sum(diffs * diffs, axis=1))
    worst_idx = int(np.argmin(distances))
    return float(distances[worst_idx]), worst_idx


def _check_thresholds(rgb_normal: np.ndarray) -> dict[str, tuple[float, int]]:
    """Run the ΔE check under each CVD; return per-CVD (min_dE, worst_pair_idx)."""
    results: dict[str, tuple[float, int]] = {}
    for cvd in CVD_TYPES:
        simulated = _simulate_cvd(rgb_normal, cvd)
        results[cvd] = _adjacent_min_delta_e(simulated)
    return results


def _report(results: dict[str, tuple[float, int]]) -> bool:
    """Print per-CVD results; return True iff all pass the floor."""
    all_pass = True
    for cvd, (min_de, idx) in results.items():
        ok = min_de >= DELTA_E_FLOOR
        all_pass = all_pass and ok
        marker = "PASS" if ok else "FAIL"
        sys.stdout.write(
            f"  {CVD_LABELS[cvd]:32s} min adjacent dE = {min_de:6.2f}  "
            f"(weakest pair: {SET3[idx]} <-> {SET3[idx + 1]})  [{marker}]\n"
        )
    return all_pass


def _render_montage(out_path: Path, rgb_normal: np.ndarray) -> None:
    """Save a 3-row montage to `out_path` (normal / deuteranopia / protanopia)."""
    import matplotlib.pyplot as plt  # local import — only needed when rendering

    rows: list[tuple[str, np.ndarray]] = [
        ("Normal vision", rgb_normal),
        (CVD_LABELS["deuteranomaly"], _simulate_cvd(rgb_normal, "deuteranomaly")),
        (CVD_LABELS["protanomaly"], _simulate_cvd(rgb_normal, "protanomaly")),
    ]

    fig, axes = plt.subplots(len(rows), 1, figsize=(12, 4.5), constrained_layout=True)
    fig.suptitle(
        f"SET3 colourblind verification — min adjacent ΔE ≥ {DELTA_E_FLOOR:.0f} (ADR-012)",
        fontsize=12,
        fontweight="bold",
    )

    for ax, (label, rgb) in zip(axes, rows, strict=True):
        clamped = np.clip(rgb, 0.0, 1.0)
        ax.imshow(clamped.reshape(1, -1, 3), aspect="auto")
        ax.set_yticks([])
        ax.set_xticks(range(len(SET3)))
        ax.set_xticklabels(SET3, fontsize=8, family="monospace", rotation=0)
        ax.set_xlim(-0.5, len(SET3) - 0.5)
        ax.set_title(label, loc="left", fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output PNG path (default: {DEFAULT_OUT.relative_to(Path.cwd())})",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Skip rendering; only verify the ΔE threshold.",
    )
    args = parser.parse_args(argv)

    rgb_normal = _palette_rgb01()

    sys.stdout.write(f"Verifying SET3 ({len(SET3)} entries) against ΔE floor {DELTA_E_FLOOR}...\n")
    results = _check_thresholds(rgb_normal)
    all_pass = _report(results)

    if not args.check_only:
        sys.stdout.write(f"Rendering montage -> {args.out}\n")
        _render_montage(args.out, rgb_normal)

    if not all_pass:
        sys.stdout.write(
            "\nVERIFICATION FAILED: one or more adjacent pairs are below "
            "the floor. Reorder SET3 (in both palette.py and theme.py) and "
            "re-run.\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
