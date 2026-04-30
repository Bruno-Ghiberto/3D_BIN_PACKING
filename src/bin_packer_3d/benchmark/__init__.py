"""Benchmark runner, instances, and result serialisation (T097, US5).

The ``bin_packer_3d.benchmark`` subpackage isolates everything related to
running registered algorithms against published reference instances
(Bischoff & Ratcliff 1995 BR1..BR8 by default) and serialising the
results in a stable JSON shape per
``specs/001-public-release-hardening/contracts/benchmark-format.md``.

This module is the package marker; concrete implementations live in:

- :mod:`bin_packer_3d.benchmark.results` — ``BenchmarkResult`` dataclass
- :mod:`bin_packer_3d.benchmark.instances` — ``BenchmarkInstance`` loader
- :mod:`bin_packer_3d.benchmark.formats` — text / JSON / Markdown output
- :mod:`bin_packer_3d.benchmark.runner` — ``BenchmarkRunner`` orchestrator
- :mod:`bin_packer_3d.benchmark.download` — fetch BR1..BR8 from upstream

Phase B Part 1 ships only the ``results`` module; the rest land in
subsequent invocations per the part-by-part US5 plan.
"""

from bin_packer_3d.benchmark.results import BenchmarkResult

__all__ = ["BenchmarkResult"]
