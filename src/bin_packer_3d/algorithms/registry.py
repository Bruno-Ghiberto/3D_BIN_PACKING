"""Algorithm registry — single source of truth for packer strategy keys.

Per ADR-0001 and ``contracts/api.md``, every concrete packer registers
itself here via :func:`register`. ``PackerConfig.strategy`` validation
and the ``bin-packer info`` CLI consult :data:`ALGORITHMS` at runtime
— no other source of strategy names exists. Adding a new algorithm is
a single-file change (decorate the class) with no corresponding edit
to config or CLI code.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from bin_packer_3d.algorithms.base import PackerBase

ALGORITHMS: dict[str, type[PackerBase]] = {}

_T = TypeVar("_T", bound=PackerBase)


def register(name: str) -> Callable[[type[_T]], type[_T]]:
    """Register a packer class under the given strategy name.

    The decorated class is added to :data:`ALGORITHMS` keyed by
    ``name``. The class is returned unchanged so decorator chaining
    and subclassing continue to work.

    Args:
        name: Strategy key consumed by ``PackerConfig.strategy`` and
            the ``bin-packer info`` CLI.

    Raises:
        ValueError: If ``name`` is already registered — surfaces the
            existing owner and the sorted list of registered names to
            aid debugging.
    """

    def decorator(cls: type[_T]) -> type[_T]:
        if name in ALGORITHMS:
            raise ValueError(
                f"strategy {name!r} already registered to "
                f"{ALGORITHMS[name].__name__}. "
                f"Registered strategies: {sorted(ALGORITHMS)}"
            )
        ALGORITHMS[name] = cls
        return cls

    return decorator


def get_strategies() -> list[str]:
    """Return the sorted list of registered strategy names."""
    return sorted(ALGORITHMS)
