"""Placement.colour augmentation tests (data-model.md, FR-006).

Three assertions for the `Placement.colour` cached property added in T022:

1. **Deterministic per placement**: same Placement instance returns the
   same colour across repeated `.colour` accesses (cached_property
   semantics + `colour_for_box` determinism).
2. **Palette membership**: the colour is drawn from `SET3`; placements
   with different box ids may or may not collide (modulo 12 cycling is
   expected), but every result MUST be a SET3 entry.
3. **Not exported**: `Placement.to_dict()` (used by
   `save_placements_to_csv`) does NOT include a `colour` field —
   adding the cached property does not alter the on-disk CSV schema.

T023 is committed red before T022 lands; the
`AttributeError: 'Placement' object has no attribute 'colour'` failure
mode confirms the test is genuinely exercising new behaviour.
"""

from __future__ import annotations

from bin_packer_3d.models.box import Box
from bin_packer_3d.models.placement import Placement
from bin_packer_3d.visualization.palette import SET3, colour_for_box


def _placement(box_id: str) -> Placement:
    """Build a minimal Placement with a deterministic Box id."""
    box = Box(id=box_id, width=10.0, height=10.0, length=10.0)
    return Placement(box=box, bin_id=1, x0=0.0, y0=0.0, z0=0.0, x1=10.0, y1=10.0, z1=10.0)


def test_colour_is_deterministic_per_placement() -> None:
    """Same placement -> same colour across calls (cached_property + hash determinism)."""
    p = _placement("BOX-042")
    first = p.colour
    second = p.colour
    third = p.colour
    assert first == second == third
    # And the colour matches the canonical colour_for_box mapping.
    assert first == colour_for_box("BOX-042")


def test_colour_is_drawn_from_palette() -> None:
    """Every colour produced by Placement.colour is a SET3 hex entry."""
    for i in range(20):
        p = _placement(f"BOX-{i:04d}")
        assert p.colour in SET3, f"Placement.colour = {p.colour!r} for {p.box.id} is not in SET3"


def test_colour_not_in_csv_export() -> None:
    """to_dict() (the CSV exporter source) does NOT carry a 'colour' field."""
    p = _placement("BOX-001")
    record = p.to_dict()
    assert "colour" not in record, (
        "Placement.to_dict() leaks the `colour` cached property into "
        "placements.csv — existing exporter contract is broken."
    )
    # Sanity: the placement still has its colour available in memory.
    _ = p.colour
