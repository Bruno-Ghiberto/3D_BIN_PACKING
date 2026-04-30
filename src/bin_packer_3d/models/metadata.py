"""Algorithm provenance metadata (T096, FR-043, data-model §AlgorithmMetadata).

A frozen dataclass attached to every :class:`PackingResult` and
:class:`~bin_packer_3d.benchmark.results.BenchmarkResult` so downstream
artefacts (CI benchmark JSON, release notes, regression dashboards) can
reproduce a run from its provenance trail alone — algorithm name, library
version, the PackerConfig parameters that drove it, the seed in effect,
and the UTC timestamp the run started.

Reproducibility contract (FR-043, Constitution §IV): given identical
``name``, ``version``, ``parameters``, and ``seed``, two runs MUST
produce identical placement decisions. ``timestamp`` is informational
and is excluded from reproducibility comparisons by the benchmark JSON
schema (``contracts/benchmark-format.md`` §Reproducibility).

Serialisation: ``dataclasses.asdict`` produces a JSON-compatible dict;
the only non-trivial field is ``timestamp`` (a ``datetime``) which the
benchmark serialiser handles via ``json.dumps(..., default=str)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def _utcnow() -> datetime:
    """Return the current UTC time as a timezone-aware :class:`datetime`."""
    return datetime.now(tz=UTC)


@dataclass(frozen=True)
class AlgorithmMetadata:
    """Provenance record captured at the start of a packing or benchmark run.

    Attributes:
        name: Registry key of the algorithm that produced the result
            (e.g. ``"ffd"``, ``"bfd"``, ``"extreme_point"``). Matches the
            key under which the packer class is registered in
            :data:`bin_packer_3d.algorithms.ALGORITHMS`.
        version: ``bin_packer_3d.__version__`` at the time the run
            started. Pinning the version inside the result lets readers
            verify they're comparing same-codebase numbers.
        parameters: Snapshot of the relevant ``PackerConfig`` fields —
            bin dimensions, ``allow_rotation``, optional weight limit,
            and any future declarative knobs. Stored as a plain dict so
            JSON round-trips without custom encoders.
        seed: The seed used for any randomised step in the run. ``None``
            means the run was not seeded (deterministic algorithms only)
            and is recorded as ``null`` in JSON.
        timestamp: UTC moment the run started. Default is
            ``datetime.now(tz=timezone.utc)`` so callers who don't pass
            an explicit value still produce ISO-8601 compliant output.
    """

    name: str
    version: str
    parameters: dict[str, Any]
    seed: int | None
    timestamp: datetime = field(default_factory=_utcnow)
