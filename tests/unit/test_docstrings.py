"""Public-symbol docstring coverage (T050, FR-026, Constitution VI NON-NEG.).

Asserts that every documentable symbol in ``bin_packer_3d.__all__`` carries a
non-empty docstring per Principle VI ("Documentation as Artefact"). The test
walks the package's public re-export list, resolves each name, and inspects
its docstring with ``inspect.getdoc`` so inheritance is honoured (a subclass
without an explicit docstring inherits its parent's).

What counts as "documentable":

- classes (including pydantic models, dataclasses, ABCs, adapters)
- functions and methods (including ``register``-style decorators)

What is intentionally skipped:

- dunder names like ``__version__`` (runtime constants — no ``__doc__``
  attachment point that's distinct from the module's own docstring)
- non-callable data attributes (e.g. the ``ALGORITHMS`` registry dict —
  the dict CLASS has a docstring; that's not what we're documenting here)

The skip list is deliberately narrow: anything with code behind it (a class
or a callable) is on the hook. If a future re-export lacks a docstring,
this test fails with the offender's name in the assertion message — the
fix is to add a docstring to the source declaration, not to extend the
skip list.

Exercises: FR-026, Principle VI.
"""

from __future__ import annotations

import inspect
from typing import Any

import bin_packer_3d


def _is_documentable(obj: Any) -> bool:
    """Return True for classes and callables (the symbols FR-026 covers).

    Pure data attributes (module-level constants, dict registries, version
    strings) carry no semantic docstring contract under FR-026 and are
    skipped. The ``inspect.isclass`` branch catches ABCs and dataclasses,
    while ``callable(obj) and not isinstance(obj, type)`` picks up plain
    functions without re-firing the class branch.
    """
    if inspect.isclass(obj):
        return True
    return callable(obj) and not isinstance(obj, type)


def test_public_symbols_have_docstrings() -> None:
    """Every callable / class in ``bin_packer_3d.__all__`` has a docstring."""
    missing: list[str] = []

    for name in bin_packer_3d.__all__:
        if name.startswith("__"):
            continue  # dunder constants like __version__

        obj = getattr(bin_packer_3d, name)

        if not _is_documentable(obj):
            continue  # data attributes (e.g. ALGORITHMS registry dict)

        doc = inspect.getdoc(obj)
        if not doc or not doc.strip():
            missing.append(name)

    assert not missing, (
        f"Public symbols missing a docstring (FR-026, Principle VI): {missing}. "
        "Add a docstring at the declaration site — do NOT extend the skip list "
        "in _is_documentable() to hide the gap."
    )
