"""DATASETS inventory guard — every file must be documented (T076).

Enforces FR-032: every ``.csv`` / ``.xlsx`` file in ``DATASETS/`` MUST
be referenced by name in ``DATASETS/README.md``. Files not yet
documented are a release-blocker.

Exercises: FR-032.
"""

from __future__ import annotations

from pathlib import Path

_DATASETS_DIR_NAME = "DATASETS"
_README_NAME = "README.md"
_TRACKED_EXTENSIONS: tuple[str, ...] = (".csv", ".xlsx")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_datasets_readme_exists() -> None:
    """DATASETS/README.md MUST exist (FR-032)."""
    readme = _repo_root() / _DATASETS_DIR_NAME / _README_NAME
    assert readme.is_file(), (
        f"{readme.relative_to(_repo_root())} must exist and document every "
        f"file in DATASETS/ (FR-032)"
    )


def test_every_dataset_file_referenced_in_readme() -> None:
    """Every DATASETS/*.{csv,xlsx} MUST be referenced by name in the README (FR-032)."""
    repo_root = _repo_root()
    datasets_dir = repo_root / _DATASETS_DIR_NAME
    readme = datasets_dir / _README_NAME

    assert datasets_dir.is_dir(), f"{datasets_dir} is missing"
    readme_text = readme.read_text(encoding="utf-8")

    data_files = sorted(
        p for p in datasets_dir.iterdir() if p.is_file() and p.suffix.lower() in _TRACKED_EXTENSIONS
    )

    missing = [p.name for p in data_files if p.name not in readme_text]
    assert not missing, (
        "DATASETS/README.md must reference every tracked data file by name. "
        f"Missing references: {missing}"
    )
