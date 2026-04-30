"""SC-014 branch-protection evidence — DELIBERATELY VIOLATES types + tests.

This file MUST NEVER MERGE TO MAIN. It exists on the
``demo/sc-014-triple-violation`` branch only, to push a PR that exercises
the branch-protection rules on ``main`` per spec §Success Criteria SC-014:

> A representative pull request that deliberately introduces a lint
> violation, a type error, AND a failing test is blocked from merging.

Two of three axes are demonstrated here — type error + failing test —
both via gates that pre-commit does NOT run locally (``mypy --strict``
and ``pytest``), so this file commits cleanly without ``--no-verify``.

The lint-violation axis is structurally identical: any ``ruff`` rule
hit trips ``core / Lint (ruff check)`` and pre-commit blocks the
commit. Demonstrating it requires ``--no-verify``; that proof is
deferred to T156 (v1.0.0 polish) where the full triple-violation
evidence lands in ``docs/compliance/v1.0.0-audit.md``.

Exercises: SC-014.
"""

from __future__ import annotations


def demo_type_violation() -> int:
    """Trip mypy --strict for SC-014 evidence.

    Assigns a ``str`` literal to a name annotated ``int`` so mypy reports
    "incompatible types in assignment". The function is otherwise
    well-formed; ruff has no opinion on type assignments.
    """
    bad: int = "string"
    return bad


def test_demo_red() -> None:
    """Trip pytest for SC-014 evidence.

    Asserts an unsatisfiable equality so the ``core / Tests`` job exits
    non-zero. ``assert 1 == 2`` is used instead of ``assert False`` to
    sidestep ruff's ``B011`` (bugbear: ``-O`` strips ``assert False``);
    the failure mode is identical for pytest's purposes.
    """
    assert 1 == 2, "intentional failure for SC-014 branch-protection demo"
