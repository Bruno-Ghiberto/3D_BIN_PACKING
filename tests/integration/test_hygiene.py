"""Repository-hygiene guard — forbids hardcoded user-specific paths (T075).

Scans every tracked file under ``src/``, ``tests/``, and ``benchmark/``
(the public-facing surface per FR-030) for two kinds of user-specific
absolute paths: Windows drive-letter roots and POSIX home directories.
See the ``_PATH_REGEX`` constant below for the exact pattern.

A match indicates someone hardcoded their development environment
into a shipped artefact. The ``legacy/`` and ``DATASETS/``
directories are intentionally out of scope — the former preserves
historical reference material, the latter holds audited data files
not part of the library surface.

This is a forward guard. Unless future drift triggers detection,
the test passes without asserting red-first behaviour; detection
logic was empirically verified against the pre-move ``CODE/``
directory (7 hits) at authorship time — see the commit body for
evidence.

Note: this module avoids writing the literal patterns it scans for
anywhere except inside the compiled regex, so the test never matches
its own source.

Exercises: FR-030.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

_PATH_REGEX = re.compile(r"[A-Z]:\\|/home/[a-z_][a-z0-9_]*/")
_SCOPE_PREFIXES: tuple[str, ...] = ("src/", "tests/", "benchmark/")


def _tracked_files_in_scope(repo_root: Path) -> list[Path]:
    """Return tracked files under the T075 scope prefixes."""
    result = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files"],
        capture_output=True,
        text=True,
        check=True,
    )
    tracked = result.stdout.splitlines()
    return [repo_root / line for line in tracked if line.startswith(_SCOPE_PREFIXES)]


def test_no_hardcoded_user_paths_in_public_tree() -> None:
    """No tracked file in src/tests/benchmark contains user-specific absolute paths (FR-030)."""
    repo_root = Path(__file__).resolve().parents[2]
    offenders: list[tuple[str, int, str]] = []

    for path in _tracked_files_in_scope(repo_root):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if _PATH_REGEX.search(line):
                offenders.append((str(path.relative_to(repo_root)), lineno, line.strip()))

    assert not offenders, (
        "Hardcoded user-specific paths found in the public tree. "
        "Move reproductions to legacy/ or rewrite with env vars / pathlib. "
        "Offenders:\n" + "\n".join(f"  {f}:{ln} — {txt!r}" for f, ln, txt in offenders)
    )
