#!/usr/bin/env python3
"""Generate the headline CSV dataset from a deterministic seed file (ADR-009).

Bin-feasibility-driven recursive guillotine cuts: a virtual sub-region of
`bin_dimensions` scaled to `target_utilisation * bin_volume` is recursively
partitioned by random axis-and-position cuts (seeded RNG) until the desired
box count is reached. Because every leaf region is a slice of a packable
volume, the resulting box set is guaranteed feasible — the regression test
`tests/unit/test_dataset_generator.py::test_bfd_utilisation_meets_floor`
confirms that BFD on the default bin clears the FR-034 60% floor.

See:
- `specs/002-portfolio-polish/research.md` ADR-009
- `specs/002-portfolio-polish/contracts/headline-dataset.md`

Spec-02 Phase A task: T012.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import click

BOX_TYPES = ("TYPE_A", "TYPE_B", "TYPE_C")
CSV_COLUMNS = ("ITEM", "W", "H", "L", "CANTIDAD", "CAJA", "DESCRIPCION")


def _validate_seed(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate the seed-file schema; raise ValueError with the offending field name."""
    required = {
        "seed": int,
        "target_utilisation": float,
        "bin_dimensions": list,
        "n_boxes": int,
        "min_box_volume_mm3": float,
    }
    for field, expected in required.items():
        if field not in payload:
            raise ValueError(f"Seed file missing required field: '{field}'")
        if not isinstance(payload[field], expected):
            raise ValueError(
                f"Seed file field '{field}' has wrong type: "
                f"expected {expected.__name__}, got {type(payload[field]).__name__}"
            )

    if payload["seed"] < 0:
        raise ValueError("seed must be a non-negative integer")
    if not (0.0 < payload["target_utilisation"] < 1.0):
        raise ValueError("target_utilisation must be a float in (0, 1)")
    if payload["n_boxes"] <= 0:
        raise ValueError("n_boxes must be a positive integer")
    if payload["min_box_volume_mm3"] <= 0:
        raise ValueError("min_box_volume_mm3 must be a positive float")

    dims = payload["bin_dimensions"]
    if len(dims) != 3:
        raise ValueError("bin_dimensions must contain exactly 3 floats (l, w, h)")
    if any(not isinstance(d, (int, float)) or d <= 0 for d in dims):
        raise ValueError("bin_dimensions entries must be positive floats")
    if payload["min_box_volume_mm3"] >= dims[0] * dims[1] * dims[2]:
        raise ValueError("min_box_volume_mm3 must be strictly less than the bin's total volume")
    return payload


def _partition(
    rng: random.Random,
    seed_region: tuple[float, float, float],
    n_boxes: int,
    min_volume: float,
) -> list[tuple[float, float, float]]:
    """Recursively cut `seed_region` into ~n_boxes leaves via random guillotine cuts.

    Stops early if every remaining region is smaller than `2 * min_volume`
    (no safe further split exists). The largest region is always picked next
    so leaf sizes stay reasonably balanced.
    """
    regions: list[tuple[float, float, float]] = [seed_region]

    while len(regions) < n_boxes:
        regions.sort(key=lambda r: r[0] * r[1] * r[2], reverse=True)
        target = regions[0]
        target_vol = target[0] * target[1] * target[2]
        if target_vol < 2 * min_volume:
            break

        # Bias toward splitting the longest dimension so leaves stay
        # close to cube-ish and BFD doesn't fragment around slivers.
        axis = max(range(3), key=lambda i: target[i])
        # Cut in a tight 45-55% band so the two halves are near-equal —
        # BFD greedy fitting packs uniform-sized leaves more reliably
        # than wildly heterogeneous ones.
        cut_ratio = rng.uniform(0.45, 0.55)
        cut_pos = target[axis] * cut_ratio

        left = list(target)
        right = list(target)
        left[axis] = cut_pos
        right[axis] = target[axis] - cut_pos

        regions[0] = (left[0], left[1], left[2])
        regions.append((right[0], right[1], right[2]))

    return regions


def _format_dim(value: float) -> float:
    """Round to 1 decimal place to keep CSV cells tidy and reproducible."""
    return round(value, 1)


def _write_csv(out_path: Path, boxes: list[tuple[float, float, float]]) -> None:
    """Write boxes to CSV with the contract-mandated column order."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(CSV_COLUMNS)
        for idx, (length, width, height) in enumerate(boxes, start=1):
            box_id = f"HEAD-{idx:03d}"
            box_type = BOX_TYPES[(idx - 1) % len(BOX_TYPES)]
            writer.writerow(
                [
                    box_id,
                    _format_dim(width),
                    _format_dim(height),
                    _format_dim(length),
                    1,
                    box_type,
                    f"Generated box {box_id}",
                ]
            )


@click.command(help="Generate the headline CSV dataset from a JSON seed file.")
@click.option(
    "--seed",
    "seed_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the JSON seed file.",
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Output CSV path (parent directories are created if missing).",
)
def main(seed_path: Path, out_path: Path) -> None:
    payload = json.loads(seed_path.read_text())
    payload = _validate_seed(payload)

    rng = random.Random(payload["seed"])
    bin_dims = tuple(float(d) for d in payload["bin_dimensions"])
    target_util = payload["target_utilisation"]

    # Linear scale per axis so the sub-region volume equals
    # target_util * bin_volume; the remainder is "air" that no box occupies.
    scale = target_util ** (1.0 / 3.0)
    seed_region = (bin_dims[0] * scale, bin_dims[1] * scale, bin_dims[2] * scale)

    regions = _partition(
        rng,
        seed_region,
        n_boxes=payload["n_boxes"],
        min_volume=payload["min_box_volume_mm3"],
    )

    # Seeded shuffle so emission order is non-trivial but deterministic.
    rng.shuffle(regions)

    _write_csv(out_path, regions)
    sys.stdout.write(
        f"Wrote {len(regions)} boxes to {out_path} "
        f"(seed={payload['seed']}, target_utilisation={target_util}).\n"
    )


if __name__ == "__main__":
    main()
