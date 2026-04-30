"""Benchmark result dataclass + JSON serialiser (T100, FR-045, FR-046).

Represents one (algorithm × instance) measurement. Every per-push CI
benchmark, every release-gate full BR1..BR8 sweep, and every local
``bin-packer benchmark`` invocation produces a list of these and emits
them via the JSON shape defined in
``specs/001-public-release-hardening/contracts/benchmark-format.md``.

The schema is versioned (``schema_version: "1"`` at top level) so
downstream consumers — the docs-site benchmark page regenerator, the
release-notes template, external reviewers — can detect breaking
changes. Field renames, removals, and type changes bump the schema
version per the same policy as ``contracts/config-schema.md``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from bin_packer_3d.models.metadata import AlgorithmMetadata

SCHEMA_VERSION = "1"
"""Current benchmark JSON schema version. Bumped on breaking changes."""


@dataclass
class BenchmarkResult:
    """Single (algorithm × instance) benchmark measurement.

    Attributes:
        algorithm: Registry key of the packer that produced this result
            (e.g. ``"ffd"``). Matches ``metadata.name``.
        instance: Reference-instance name (e.g. ``"BR1"``). Matches the
            corresponding :class:`BenchmarkInstance.name` once that
            module lands in Part 2.
        n_boxes: Number of input boxes the run attempted to pack.
        n_bins_used: Number of bins the run actually opened. ``>= 0``.
        volume_utilisation: Placed-volume / opened-bin-volume ratio in
            ``[0.0, 1.0]``. Higher is better.
        success_rate: Percentage of boxes successfully placed in
            ``[0.0, 100.0]``. ``100.0`` means every input box landed.
        elapsed_seconds: Wall-clock seconds the packer spent in
            ``pack()``. ``>= 0``.
        metadata: Provenance record (algorithm, version, parameters,
            seed, UTC timestamp) — see
            :class:`bin_packer_3d.models.metadata.AlgorithmMetadata`.
        notes: Free-form annotation. Default empty. Used to surface the
            generated seed when the user did not pass one (so the result
            is replayable post-hoc).
    """

    algorithm: str
    instance: str
    n_boxes: int
    n_bins_used: int
    volume_utilisation: float
    success_rate: float
    elapsed_seconds: float
    metadata: AlgorithmMetadata
    notes: str = ""


def _json_default(obj: Any) -> Any:
    """JSON encoder for non-native types used in benchmark results."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def to_single_document(result: BenchmarkResult) -> dict[str, Any]:
    """Wrap a single result in the top-level ``kind="single"`` envelope.

    Matches ``contracts/benchmark-format.md`` §"SingleResult — one
    algorithm, one instance".
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "single",
        "result": asdict(result),
    }


def to_comparison_document(
    results: list[BenchmarkResult],
    *,
    seed: int | None,
    bin_packer_version: str,
    generated_at: datetime,
) -> dict[str, Any]:
    """Wrap a list of results in the top-level ``kind="comparison"`` envelope.

    Matches ``contracts/benchmark-format.md`` §"ComparisonTable —
    multiple algorithms × one or more instances".

    Args:
        results: Per-(algorithm, instance) measurements.
        seed: The seed that drove the comparison (forwarded into the
            ``summary`` block). ``None`` is recorded as ``null``.
        bin_packer_version: ``bin_packer_3d.__version__`` at run time.
        generated_at: UTC timestamp the comparison was assembled.
    """
    instances = {r.instance for r in results}
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "comparison",
        "results": [asdict(r) for r in results],
        "summary": {
            "n_algorithms": len({r.algorithm for r in results}),
            "n_instances": len(instances),
            "seed": seed,
            "generated_at": generated_at.isoformat(),
            "bin_packer_version": bin_packer_version,
        },
    }


def dumps(document: dict[str, Any]) -> str:
    """Serialise a benchmark document to a deterministic JSON string.

    The output uses 2-space indentation, sorts top-level keys for
    consistency across runs (FR-043 reproducibility), and emits
    ``datetime`` instances via ISO 8601.
    """
    return json.dumps(document, indent=2, sort_keys=True, default=_json_default)


__all__ = [
    "SCHEMA_VERSION",
    "BenchmarkResult",
    "dumps",
    "to_comparison_document",
    "to_single_document",
]
