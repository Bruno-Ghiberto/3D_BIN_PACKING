"""Packing algorithms for 3D bin packing.

Algorithms are registered in the module-level :data:`ALGORITHMS` dict
via the :func:`register` decorator (ADR-0001). ``PackerConfig.strategy``
validation consults this registry, so adding a new algorithm is a
single-file change — no edits to config or CLI required.

The concrete packer modules import :func:`register` directly from
``bin_packer_3d.algorithms.registry``; they do NOT import it from this
package's ``__init__``, which avoids partial-init cycles during
package import.
"""

from bin_packer_3d.algorithms.base import PackerBase
from bin_packer_3d.algorithms.ffd import FirstFitDecreasingPacker
from bin_packer_3d.algorithms.registry import ALGORITHMS, get_strategies, register
from bin_packer_3d.algorithms.shelf import ShelfPacker

__all__ = [
    "ALGORITHMS",
    "FirstFitDecreasingPacker",
    "PackerBase",
    "ShelfPacker",
    "get_strategies",
    "register",
]
