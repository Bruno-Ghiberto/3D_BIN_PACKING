"""Deterministic colour assignment for visualised boxes (ADR-007).

Exports:

- :data:`SET3`: the canonical 12-entry ColorBrewer Set3 palette (hex strings).
- :func:`colour_for_box`: maps a box identifier to a stable palette entry
  via BLAKE2b hashing.

The hash is deterministic across runs, processes, machines, and Python
versions — unlike the built-in :func:`hash`, BLAKE2b is not subject to
``PYTHONHASHSEED`` randomisation. The 16-bit digest is reduced modulo the
palette length to yield an index; identical box IDs always produce
identical colours.

Colourblind safety: the SET3 ordering used here keeps adjacent entries at
CIELAB ΔE >= 15 under both deuteranopia and protanopia simulation
(verified empirically by ``scripts/verify_palette_colourblind.py`` and
asserted by ``tests/unit/test_palette_colourblind.py``).

Contract: ``specs/002-portfolio-polish/contracts/visualisation-theme.md``.
"""

from __future__ import annotations

import hashlib

# ColorBrewer Set3 (12 entries). The canonical ordering for this project;
# mirrored by `bin_packer_3d.visualization.theme.SET3`. Reordering
# requires re-running scripts/verify_palette_colourblind.py and committing
# the regenerated docs/assets/palette_colourblind_check.png artefact.
SET3: tuple[str, ...] = (
    "#D9D9D9",  # grey
    "#FFED6F",  # yellow
    "#80B1D3",  # blue
    "#FB8072",  # salmon
    "#BEBADA",  # lavender
    "#FDB462",  # orange
    "#8DD3C7",  # turquoise
    "#B3DE69",  # green
    "#FCCDE5",  # pink
    "#FFFFB3",  # cream
    "#BC80BD",  # purple
    "#CCEBC5",  # mint
)
# Ordering rationale: greedy nearest-far search across CIELAB
# distances under deuteranopia + protanopia simulation yields
# min adjacent dE = 39.80 (well above the ADR-012 floor of 15).
# Asserted by tests/unit/test_palette_colourblind.py.


def colour_for_box(box_id: str, palette: tuple[str, ...] = SET3) -> str:
    """Return the deterministic palette colour for a box identifier.

    Args:
        box_id: Stable identifier for the box (e.g. ``"BOX-001"``).
        palette: Optional override; defaults to :data:`SET3`.

    Returns:
        A 7-character hex string of the form ``"#RRGGBB"``.

    The mapping is:

    1. ``digest = blake2b(box_id.utf8, digest_size=2)``
    2. ``index = int.from_bytes(digest, "big") % len(palette)``
    3. Return ``palette[index]``.

    Identical ``box_id`` always returns identical output across Python
    versions and processes (BLAKE2b is not subject to hash randomisation).
    """
    digest = hashlib.blake2b(box_id.encode("utf-8"), digest_size=2).digest()
    index = int.from_bytes(digest, "big") % len(palette)
    return palette[index]
