"""Docs-runtime consistency guard (T051, completes T020 — US1 AC6 + US3 AC8).

When the documentation tree is authored in Invocation 13 (T057..T061), every
registered packing algorithm must have its own reference page at
``docs/algorithms/<name>.md``, and conversely every page in that directory
must name an algorithm the runtime can actually execute. Drift in either
direction is a Contract-Honesty regression (Principle I) — phantom pages or
phantom strategies erode the same trust the original audit caught.

The guard is forward-looking: while ``docs/algorithms/`` does not yet exist
(this is the v0.2.0 seed), the test ``skip``s with a pointer to the
implementing tasks. Once T057..T061 land in Invocation 13, the same test
flips to strict assertion mode without any rewrite — adding or removing an
algorithm OR a page automatically becomes a CI failure until the other side
is updated.

This is the Phase 0 deferred from T020 (US1 AC6) — postponed in Invocation 3
because authoring it against a missing directory would have been
permanently red. The skip-when-absent design now resolves the original
deferral cleanly.

Exercises: US1 AC6, US3 AC8, FR-001 / FR-002 (registry-docs alignment).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bin_packer_3d.algorithms import ALGORITHMS


def _docs_algorithms_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "docs" / "algorithms"


def test_docs_algorithms_match_registry() -> None:
    """Algorithm pages and registry keys must agree once docs are authored."""
    docs_dir = _docs_algorithms_dir()
    if not docs_dir.exists():
        pytest.skip(
            "docs/algorithms/ not yet authored (Invocation 13, tasks T057..T061). "
            "This guard activates once the directory exists."
        )

    pages = {path.stem for path in docs_dir.glob("*.md")}
    algorithms = set(ALGORITHMS)

    missing_pages = sorted(algorithms - pages)
    orphan_pages = sorted(pages - algorithms)

    assert not missing_pages, (
        f"Registered algorithm(s) without a docs page: {missing_pages}. "
        "Add a docs/algorithms/<name>.md per FR-021 / US3 AC8."
    )
    assert not orphan_pages, (
        f"Docs page(s) for unregistered algorithm(s): {orphan_pages}. "
        "Either delete the page or register the algorithm — phantom pages "
        "violate Principle I (Contract Honesty)."
    )
