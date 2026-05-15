#!/usr/bin/env python3
"""README regenerator (T028, US1 — spec-02 Portfolio Polish).

Three marker-delimited sections in ``README.md`` are deterministic
projections of repository state. This script computes their canonical
content and replaces the body of each section between its
``<!-- BEGIN: NAME -->`` / ``<!-- END: NAME -->`` markers (ADR-004):

- ``ALGORITHMS_TABLE`` — comparison table sourced from the
  ``bin_packer_3d.algorithms.ALGORITHMS`` registry per
  ``contracts/algorithm-card-source.md``.
- ``HIGHLIGHTS`` — numeric facts sourced from live repository state
  per ADR-005 + ``contracts/algorithm-card-source.md``:
  test count (``pytest --collect-only -q``), CI check count (non-aggregate
  jobs in ``_ci-core.yml``), supported Python versions (pyproject
  classifiers), license (pyproject ``project.license``), and algorithm
  registry size.
- ``PROJECT_STRUCTURE`` — top-level filesystem layout filtered against
  the disposition table in ``contracts/repository-structure.md``.

Usage:

    python scripts/regenerate_readme.py            # rewrite README.md
    python scripts/regenerate_readme.py --check    # exit non-zero on drift

Exit codes:
    0 — README rewritten in-place (default) OR already in sync (--check)
    1 — drift detected (--check) OR markers missing in README
    2 — internal error (probe failed, registry unreachable, etc.)
"""

from __future__ import annotations

import argparse
import difflib
import re
import subprocess
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Final

import yaml

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
README: Final[Path] = REPO_ROOT / "README.md"
PYPROJECT: Final[Path] = REPO_ROOT / "pyproject.toml"
CI_WORKFLOW: Final[Path] = REPO_ROOT / ".github" / "workflows" / "_ci-core.yml"

# Make src/ importable when this script runs against a bare checkout
# (no editable install). The CI smoke job already installs the package,
# but a maintainer running the script locally without `pip install -e .`
# should still get a useful run.
sys.path.insert(0, str(REPO_ROOT / "src"))

_SECTION_BLOCK = re.compile(
    r"(?P<begin><!-- BEGIN: (?P<name>[A-Z_]+) -->)"
    r"(?P<body>.*?)"
    r"(?P<end><!-- END: (?P=name) -->)",
    re.DOTALL,
)
_PY_CLASSIFIER = re.compile(r"^Programming Language :: Python :: (\d+\.\d+)$")


# ---------------------------------------------------------------------------
# Live-state probes
# ---------------------------------------------------------------------------


def _load_pyproject() -> dict[str, object]:
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)


def _live_python_versions() -> list[str]:
    """Parse ``Programming Language :: Python :: X.Y`` classifiers."""
    meta = _load_pyproject()
    project = meta["project"]
    assert isinstance(project, dict)
    classifiers = project["classifiers"]
    assert isinstance(classifiers, list)
    out: list[str] = []
    for c in classifiers:
        if isinstance(c, str):
            m = _PY_CLASSIFIER.match(c)
            if m:
                out.append(m.group(1))
    return out


def _live_license_text() -> str:
    meta = _load_pyproject()
    project = meta["project"]
    assert isinstance(project, dict)
    lic = project["license"]
    if isinstance(lic, dict) and "text" in lic:
        return str(lic["text"])
    return str(lic)


def _live_test_count() -> int:
    """Collect-only invocation of pytest; parse the ``N collected`` line."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "--no-header"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    m = re.search(r"(\d+)\s+tests?\s+collected", result.stdout)
    if not m:
        raise RuntimeError(f"could not parse test count from pytest output:\n{result.stdout}")
    return int(m.group(1))


def _live_ci_check_count() -> int:
    """Count non-aggregate jobs in the core CI workflow."""
    with CI_WORKFLOW.open("rb") as fh:
        doc = yaml.safe_load(fh)
    jobs = doc["jobs"]
    return sum(1 for name in jobs if name != "aggregate")


# ---------------------------------------------------------------------------
# Section generators
# ---------------------------------------------------------------------------


def gen_algorithms_table() -> str:
    """Markdown table of every registered algorithm.

    Sources the display ``name`` property by instantiating each packer
    with a default :class:`PackerConfig` (whose ``strategy`` field is
    set to the registry key under test so the validator accepts it).
    ``complexity`` and ``description`` are ``ClassVar`` and read
    directly off the class.
    """
    from bin_packer_3d.algorithms import ALGORITHMS
    from bin_packer_3d.config import PackerConfig

    lines: list[str] = [
        "",
        "| Key | Algorithm | Complexity | Description |",
        "|---|---|---|---|",
    ]
    for key in sorted(ALGORITHMS):
        cls = ALGORITHMS[key]
        packer = cls(PackerConfig(strategy=key))
        lines.append(f"| `{key}` | {packer.name} | `{cls.complexity}` | {cls.description} |")
    lines.append("")
    return "\n".join(lines)


def gen_highlights() -> str:
    """Bullet list of live-state facts per ADR-005."""
    from bin_packer_3d.algorithms import ALGORITHMS

    n_algos = len(ALGORITHMS)
    keys_str = "`, `".join(sorted(ALGORITHMS))
    n_tests = _live_test_count()
    n_ci = _live_ci_check_count()
    py_versions = _live_python_versions()
    py_versions_str = " · ".join(py_versions)
    license_text = _live_license_text()

    lines: list[str] = [
        "",
        f"- **{n_algos} packing algorithms** registered: `{keys_str}`.",
        f"- **{n_tests} tests** in the suite; **{n_ci} CI checks** gate every PR.",
        f"- Supported Python: **{py_versions_str}**.",
        f"- Licensed under **{license_text}**.",
        "",
    ]
    return "\n".join(lines)


# Per contracts/repository-structure.md § Top-Level Disposition Table.
# Order matters — entries appear in the README block in this sequence.
# Entries that are not present on the filesystem are silently skipped
# by ``gen_project_structure()``; orphan FS entries (present but not
# listed here) are caught by ``test_structure_drift.py``.
_STRUCTURE_ENTRIES: list[tuple[str, str]] = [
    # source + tests + docs (the headline tree)
    ("src/bin_packer_3d/", "Library source code"),
    ("tests/", "Unit + integration + property tests"),
    ("docs/", "Documentation site source (mkdocs-material)"),
    ("examples/", "Curated demo datasets"),
    ("scripts/", "Maintainer tooling (audits, regenerators, recordings)"),
    # archive + benchmarks + planning
    ("DATASETS/", "Legacy data archive (audited per spec-01)"),
    ("legacy/", "Preserved Alpha-era code (excluded from wheel)"),
    ("benchmark/", "Benchmark engine scaffolding (US5 territory)"),
    ("Speckit-context-prompts/", "SDD planning artefacts (excluded from wheel)"),
    # root-level metadata files
    ("pyproject.toml", "Build + tool configuration"),
    ("MANIFEST.in", "Sdist manifest"),
    ("README.md", "This file"),
    ("LICENSE", "MIT license"),
    ("CHANGELOG.md", "Keep-a-Changelog log"),
    ("CONTRIBUTING.md", "Contributor onboarding"),
    # dotfile configuration
    (".editorconfig", "Editor configuration"),
    (".gitignore", "Git ignore rules"),
    (".pre-commit-config.yaml", "Pre-commit hook configuration"),
    (".github/", "GitHub configuration (workflows, templates)"),
    (".specify/", "SpecKit machinery"),
    (".serena/", "Serena tooling"),
]


def gen_project_structure() -> str:
    """Filesystem tree projection filtered against the disposition table.

    Iterates the contract's ordered entries; any entry that exists on
    the filesystem is included with its annotation. Entries that do
    NOT exist are silently skipped (forward-compatible with files
    being added or removed in later phases). A trailing ``...`` row
    points readers to the docs site for the full tree.
    """
    present = [(path, note) for path, note in _STRUCTURE_ENTRIES if (REPO_ROOT / path).exists()]

    # Column-align the descriptions for readability. 28 cols is wide
    # enough for every entry in the disposition table.
    pad = max(len(path) for path, _ in present) + 4

    lines = ["", "```text", "3D_BIN_PACKING/"]
    for path, note in present:
        # Every concrete row uses the branch connector; the final
        # ``└── ...`` row below is the only terminator. Readers who
        # want the full tree are pointed to the docs site (which the
        # docs build renders from ``mkdocs gen-files``).
        lines.append(f"├── {path:<{pad}}{note}")
    lines.append(f"└── {'...':<{pad}}See docs site for the full tree")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Block replacement
# ---------------------------------------------------------------------------


SectionGenerator = Callable[[], str]

_GENERATORS: dict[str, SectionGenerator] = {
    "ALGORITHMS_TABLE": gen_algorithms_table,
    "HIGHLIGHTS": gen_highlights,
    "PROJECT_STRUCTURE": gen_project_structure,
}


def regenerate(content: str) -> str:
    """Return ``content`` with every known section's body replaced.

    Raises:
        ValueError: if any expected ``<!-- BEGIN: NAME -->`` marker is
            missing from the input. The error names every missing
            section so the user knows which markers T031 owes.
    """
    missing = [
        name
        for name in _GENERATORS
        if f"<!-- BEGIN: {name} -->" not in content or f"<!-- END: {name} -->" not in content
    ]
    if missing:
        raise ValueError(
            "README missing marker block(s) for: "
            + ", ".join(sorted(missing))
            + " — implement T031 (README rewrite) before running the regenerator"
        )

    def _replace(match: re.Match[str]) -> str:
        name = match.group("name")
        gen = _GENERATORS.get(name)
        if gen is None:
            # Unknown marker — leave untouched. Useful when the README
            # gains a new marker block before the regenerator catches up.
            return match.group(0)
        body = gen()
        return f"{match.group('begin')}{body}{match.group('end')}"

    return _SECTION_BLOCK.sub(_replace, content)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _emit_diff(before: str, after: str) -> str:
    diff = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile="README.md (committed)",
        tofile="README.md (regenerated)",
        n=3,
    )
    return "".join(diff)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="regenerate_readme",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero on drift without rewriting README.md",
    )
    args = parser.parse_args(argv)

    before = README.read_text(encoding="utf-8")
    try:
        after = regenerate(before)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.check:
        if before == after:
            return 0
        print(
            "drift detected — README.md is out of sync with live repo state.",
            file=sys.stderr,
        )
        print(
            "run `python scripts/regenerate_readme.py` and commit the result.",
            file=sys.stderr,
        )
        sys.stderr.write(_emit_diff(before, after))
        return 1

    if before == after:
        # No-op; do not touch mtime.
        return 0

    README.write_text(after, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
