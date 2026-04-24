#!/usr/bin/env python3
"""Release pre-publish gate — scan for unaudited dataset files (T083).

Scans the working tree AND the full git history for filenames matching
known business-artefact patterns. Any match NOT in the maintained
allow-list exits non-zero so a maintainer can audit the file (update
``DATASETS/AUDIT.md`` + this allow-list) before shipping.

Called from ``.github/workflows/release.yml`` in US8; also runnable
locally:

    python scripts/audit_datasets.py

Exit codes:
    0 — all deny-pattern matches are in the allow-list (or none exist)
    1 — at least one unauthorised match; see stderr for the file list
    2 — underlying git invocation failed

Maintenance: keep ``DENY_PATTERNS`` narrow (each entry risks false
positives that block releases) and keep ``ALLOW`` in lockstep with
``DATASETS/AUDIT.md``.
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Iterable

# Filename substrings that suggest business-origin or confidential
# datasets. Matches are checked against full tracked paths (e.g.
# ``DATASETS/PACKING LIST.xlsx``), so these need not include the
# parent directory.
DENY_PATTERNS: tuple[str, ...] = (
    r"PACKING LIST",
    r"PESO_P\.T",
    r"DIMENSIONES CAJAS",
    r"asignacion_cajas",
    r"placements_result",
)

# Files that match a DENY pattern but have been explicitly audited
# and cleared for release. Keep in sync with ``DATASETS/AUDIT.md``.
#
# The list covers three families:
#   1. Current tracked DATASETS/*.xlsx — the CNH reference inputs.
#   2. Early-history repo-root versions of the same xlsx files,
#      before they were moved under DATASETS/ (same content, older
#      path).
#   3. Historical generated outputs from running the legacy CODE/
#      scripts against the inputs above — same CNH workflow,
#      same maintainer clearance.
ALLOW: frozenset[str] = frozenset(
    {
        # Current DATASETS/*.xlsx (tracked, audited — see AUDIT.md).
        "DATASETS/PACKING LIST.xlsx",
        "DATASETS/PACKING LIST-11.xlsx",
        "DATASETS/DIMENSIONES CAJAS-NORMALIZADO.xlsx",
        "DATASETS/PESO_P.T.xlsx",
        # Early-history repo-root versions (moved into DATASETS/
        # in an earlier commit; same content).
        "PACKING LIST.xlsx",
        "DIMENSIONES CAJAS-NORMALIZADO.xlsx",
        "PESO_P.T.xlsx",
        # Historical generated outputs from legacy CODE/MAIN.py
        # runs — derivatives of the cleared inputs above.
        "DATASETS/asignacion_cajas_final.csv",
        "DATASETS/asignacion_cajas_final-_-.xlsx",
        "DATASETS/placements_result.csv",
    }
)


def _run(args: list[str]) -> str:
    """Run a git subcommand, returning stdout on success."""
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        sys.stderr.write(
            f"ERROR: {' '.join(args)} exited {result.returncode}: {result.stderr.strip()}\n"
        )
        raise SystemExit(2)
    return result.stdout


def worktree_files() -> set[str]:
    """Return every tracked file path in the working tree."""
    return {line for line in _run(["git", "ls-files"]).splitlines() if line}


def history_files() -> set[str]:
    """Return every file path that ever existed in git history."""
    raw = _run(
        [
            "git",
            "log",
            "--all",
            "--full-history",
            "--pretty=format:",
            "--name-only",
        ]
    )
    return {line for line in raw.splitlines() if line}


def matching(files: Iterable[str]) -> set[str]:
    """Return the subset of ``files`` that matches any DENY pattern."""
    compiled = [re.compile(p) for p in DENY_PATTERNS]
    return {f for f in files if any(r.search(f) for r in compiled)}


def main() -> int:
    """Return 0 if every deny-pattern match is allow-listed."""
    all_matches = matching(worktree_files()) | matching(history_files())
    unauthorised = sorted(f for f in all_matches if f not in ALLOW)

    if unauthorised:
        sys.stderr.write(
            "Deny-pattern match NOT in allow-list "
            "(update DATASETS/AUDIT.md + scripts/audit_datasets.py ALLOW):\n"
        )
        for path in unauthorised:
            sys.stderr.write(f"  {path}\n")
        return 1

    sys.stdout.write(f"OK — {len(all_matches)} deny-pattern matches, all present in allow-list.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
