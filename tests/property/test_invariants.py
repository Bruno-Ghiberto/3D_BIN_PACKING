"""Property-based invariants for every registered packer (T085, FR-046).

Hypothesis generates small box sets and runs each algorithm in
:data:`bin_packer_3d.algorithms.ALGORITHMS` against them, asserting the
four invariants from spec §FR-046:

1. **Non-overlap** — placements within the same bin do not intersect.
2. **In-bounds** — every placement lies entirely inside its bin.
3. **Volume conservation** — placed volume + unpacked volume == input
   volume (within float tolerance).
4. **Box-count conservation** — placed count + unpacked count == input
   count.

The suite parametrises over ``ALGORITHMS.keys()`` at collection time so
adding a new packer (US5 BFD / Extreme Point / Maximal Rectangles, US7
constraint-aware variants) automatically enrols it in the property
checks. Failures from a future algorithm surface as a hypothesis blob in
the CI log per the ``ci`` profile registered in ``conftest.py``.

Sizing rationale: dimensions are kept small (``[1, 30]`` for boxes,
``80x80x80`` bin) so a default Hypothesis run completes in tens of
milliseconds per algorithm. The ``ci`` profile bumps ``max_examples``
to 200; the default profile uses 100 so local iteration stays brisk.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from bin_packer_3d.algorithms import ALGORITHMS
from bin_packer_3d.config import PackerConfig
from bin_packer_3d.models.box import Box
from bin_packer_3d.models.placement import Placement

pytestmark = pytest.mark.property


_ALGO_KEYS = sorted(ALGORITHMS.keys())

_BIN_DIMS = (80.0, 80.0, 80.0)


@st.composite
def _box(draw: st.DrawFn) -> Box:
    """Generate a small Box with positive integer dimensions."""
    return Box(
        id=draw(
            st.text(
                alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=1, max_size=4
            )
        ),
        width=float(draw(st.integers(min_value=1, max_value=30))),
        height=float(draw(st.integers(min_value=1, max_value=30))),
        length=float(draw(st.integers(min_value=1, max_value=30))),
    )


_box_list = st.lists(_box(), min_size=1, max_size=12, unique_by=lambda b: b.id)


def _bin_config() -> PackerConfig:
    return PackerConfig(bin_length=_BIN_DIMS[0], bin_width=_BIN_DIMS[1], bin_height=_BIN_DIMS[2])


def _placement_overlaps(a: Placement, b: Placement) -> bool:
    """Two placements overlap iff their open intervals intersect on every axis."""
    return (
        a.x0 < b.x1 and b.x0 < a.x1 and a.y0 < b.y1 and b.y0 < a.y1 and a.z0 < b.z1 and b.z0 < a.z1
    )


@pytest.mark.parametrize("algo_key", _ALGO_KEYS)
@given(boxes=_box_list)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_no_overlap_within_bin(algo_key: str, boxes: list[Box]) -> None:
    """No two placements in the same bin overlap (FR-046 invariant 1)."""
    packer = ALGORITHMS[algo_key](_bin_config())
    result = packer.pack(boxes)

    for bin_obj in result.bins:
        placements = bin_obj.placements
        for i, a in enumerate(placements):
            for b in placements[i + 1 :]:
                assert not _placement_overlaps(a, b), (
                    f"{algo_key}: overlap in bin {bin_obj.id} between {a} and {b}"
                )


@pytest.mark.parametrize("algo_key", _ALGO_KEYS)
@given(boxes=_box_list)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_in_bounds(algo_key: str, boxes: list[Box]) -> None:
    """Every placement lies within bin bounds (FR-046 invariant 2)."""
    packer = ALGORITHMS[algo_key](_bin_config())
    result = packer.pack(boxes)

    for bin_obj in result.bins:
        for p in bin_obj.placements:
            assert p.x0 >= 0 and p.y0 >= 0 and p.z0 >= 0, (
                f"{algo_key}: negative origin in placement {p}"
            )
            assert p.x1 <= bin_obj.length + 1e-6, (
                f"{algo_key}: placement {p} exceeds bin length {bin_obj.length}"
            )
            assert p.y1 <= bin_obj.width + 1e-6, (
                f"{algo_key}: placement {p} exceeds bin width {bin_obj.width}"
            )
            assert p.z1 <= bin_obj.height + 1e-6, (
                f"{algo_key}: placement {p} exceeds bin height {bin_obj.height}"
            )


@pytest.mark.parametrize("algo_key", _ALGO_KEYS)
@given(boxes=_box_list)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_volume_conservation(algo_key: str, boxes: list[Box]) -> None:
    """Input volume == placed volume + unpacked volume (FR-046 invariant 3)."""
    packer = ALGORITHMS[algo_key](_bin_config())
    result = packer.pack(boxes)

    input_volume = sum(b.volume for b in boxes)
    placed_volume = sum(p.box.volume for bin_obj in result.bins for p in bin_obj.placements)
    unpacked_volume = sum(b.volume for b in result.unpacked_boxes)

    assert math.isclose(input_volume, placed_volume + unpacked_volume, rel_tol=1e-9), (
        f"{algo_key}: volume conservation violated — "
        f"input={input_volume}, placed={placed_volume}, unpacked={unpacked_volume}"
    )


@pytest.mark.parametrize("algo_key", _ALGO_KEYS)
@given(boxes=_box_list)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_count_conservation(algo_key: str, boxes: list[Box]) -> None:
    """Input count == placed count + unpacked count (FR-046 invariant 4)."""
    packer = ALGORITHMS[algo_key](_bin_config())
    result = packer.pack(boxes)

    placed_count = sum(len(bin_obj.placements) for bin_obj in result.bins)
    unpacked_count = len(result.unpacked_boxes)

    assert placed_count + unpacked_count == len(boxes), (
        f"{algo_key}: count conservation violated — "
        f"input={len(boxes)}, placed={placed_count}, unpacked={unpacked_count}"
    )
