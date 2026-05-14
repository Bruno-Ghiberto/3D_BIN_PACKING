"""Deterministic colour assignment + colourblind safety tests (ADR-007 + ADR-012).

Three assertions against `bin_packer_3d.visualization.palette`:

1. **Determinism (FR-006, SC-006)**: ``colour_for_box(box_id)`` returns the
   same colour every time for the same input.
2. **Palette cycling**: with >12 unique inputs, every output is still drawn
   from the SET3 palette (modulo indexing works correctly; no crashes).
3. **Colourblind safety (FR-037 / ADR-012)**: under both deuteranopia and
   protanopia simulation via `colorspacious`, the minimum pairwise CIELAB
   ΔE between **adjacent** SET3 entries stays at or above 15. ADR-012
   reads "adjacent palette entries" — the perceptual-distinguishability
   guarantee for a categorical palette where order is fixed.

The palette module is authored in T019; this test is committed in a red
state (T017) beforehand.
"""

from __future__ import annotations

import numpy as np
from bin_packer_3d.visualization.palette import SET3, colour_for_box
from colorspacious import cspace_convert

DELTA_E_FLOOR = 15.0


def _hex_to_rgb01(hex_str: str) -> tuple[float, float, float]:
    """Convert "#RRGGBB" to a (r, g, b) tuple of floats in [0, 1]."""
    hex_str = hex_str.lstrip("#")
    return (
        int(hex_str[0:2], 16) / 255.0,
        int(hex_str[2:4], 16) / 255.0,
        int(hex_str[4:6], 16) / 255.0,
    )


def _palette_lab_under(cvd_type: str) -> np.ndarray:
    """Return SET3 in CIELab after simulating the given colour-vision deficiency."""
    rgb = np.array([_hex_to_rgb01(c) for c in SET3])  # shape (N, 3)
    simulated_rgb = cspace_convert(
        rgb, {"name": "sRGB1+CVD", "cvd_type": cvd_type, "severity": 100}, "sRGB1"
    )
    return np.asarray(cspace_convert(simulated_rgb, "sRGB1", "CIELab"))


def _adjacent_min_delta_e(lab: np.ndarray) -> float:
    """Minimum Euclidean ΔE between consecutive Lab entries."""
    diffs = np.diff(lab, axis=0)
    distances = np.sqrt(np.sum(diffs * diffs, axis=1))
    return float(np.min(distances))


def test_colour_for_box_is_deterministic() -> None:
    """Same box_id -> same colour across calls and across runs."""
    box_id = "BOX-042"
    first = colour_for_box(box_id)
    second = colour_for_box(box_id)
    third = colour_for_box(box_id)
    assert first == second == third, (
        "colour_for_box returned different colours for the same box_id — "
        "Principle IV (Reproducibility) violated."
    )
    # Sanity: result is a 7-char hex string from the SET3 palette.
    assert first.startswith("#") and len(first) == 7
    assert first in SET3


def test_palette_cycles_for_many_ids() -> None:
    """100 unique box_ids stay inside SET3 (modulo cycling never crashes)."""
    colours = {colour_for_box(f"BOX-{i:04d}") for i in range(100)}
    # Every emitted colour MUST belong to SET3 (no off-palette values).
    assert colours.issubset(set(SET3)), f"Emitted colours outside SET3: {colours - set(SET3)}"
    # 100 inputs should exercise the cycle and produce multiple distinct
    # outputs — sanity that the hash isn't accidentally constant.
    assert len(colours) > 1, "colour_for_box appears to be constant"


def test_colourblind_safety_deuteranopia() -> None:
    """Adjacent SET3 entries stay distinguishable under deuteranopia (ΔE >= 15)."""
    lab = _palette_lab_under("deuteranomaly")
    min_de = _adjacent_min_delta_e(lab)
    assert min_de >= DELTA_E_FLOOR, (
        f"Deuteranopia: min adjacent ΔE = {min_de:.2f} is below the "
        f"ADR-012 floor of {DELTA_E_FLOOR}. Reorder SET3 or replace the "
        f"weakest pair."
    )


def test_colourblind_safety_protanopia() -> None:
    """Adjacent SET3 entries stay distinguishable under protanopia (ΔE >= 15)."""
    lab = _palette_lab_under("protanomaly")
    min_de = _adjacent_min_delta_e(lab)
    assert min_de >= DELTA_E_FLOOR, (
        f"Protanopia: min adjacent ΔE = {min_de:.2f} is below the "
        f"ADR-012 floor of {DELTA_E_FLOOR}. Reorder SET3 or replace the "
        f"weakest pair."
    )
