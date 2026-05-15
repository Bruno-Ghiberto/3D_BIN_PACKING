"""README image alt-text guard (T026, US1 Independent-Test G-US1-IT-2).

Drives the alt-text contract from FR-036: every embedded image in
``README.md`` must have non-empty alternative text. The hero image
``docs/assets/hero.gif`` is the load-bearing case per FR-022 (the
README MUST display a packed-bin animation in the first viewport) and
must itself carry a meaningful, non-empty alt attribute.

Two assertions, both regex-parsed via stdlib ``re`` (no markdown
library — the spec-02 quickstart is plain-text-friendly and the parse
must work in any environment):

1. The hero image at ``docs/assets/hero.gif`` is present in the README
   with non-empty alt text. RED until T031 (README rewrite) lands —
   the current README has no hero asset. This is the honest TDD-red
   anchor for the file, because the badges already carry alt text and
   a pure "no image has empty alt" lint would pass vacuously.

2. No image syntax anywhere in the README has empty alt text. This is
   the FR-036 lint that survives forever — it stays green as long as
   every ``![alt](path)`` keeps its alt.

Source FRs: FR-022 (hero asset), FR-036 (alt text mandatory).
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"

# Matches Markdown image syntax `![alt](path)`. Inline-only — multi-line
# alt is not part of the spec and would not be honoured by GitHub's
# Markdown renderer anyway.
_IMAGE_PATTERN = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)\s]+)(?:\s+\"[^\"]*\")?\)")

# Hero asset path mandated by FR-022 + tasks.md T031.
_HERO_PATH = "docs/assets/hero.gif"


def _extract_images(content: str) -> list[tuple[str, str]]:
    """Return ``[(alt, path), ...]`` for every ``![alt](path)`` in ``content``."""
    return [(m.group("alt"), m.group("path")) for m in _IMAGE_PATTERN.finditer(content)]


def test_hero_image_present_with_alt_text() -> None:
    """FR-022 + FR-036: README has the hero image AND its alt is non-empty.

    Red until T031 rewrites the README with the hero block. The hero
    path is exact per the spec; if the maintainer chooses a different
    extension (.webp, .mp4 poster, etc.) the spec must be amended first.
    """
    content = README.read_text(encoding="utf-8")
    images = _extract_images(content)

    hero_matches = [(alt, path) for alt, path in images if path == _HERO_PATH]
    assert hero_matches, (
        f"README must include the hero image at {_HERO_PATH!r} per FR-022; "
        f"none of the {len(images)} image syntax occurrences match"
    )

    for alt, _ in hero_matches:
        assert alt.strip(), (
            f"hero image {_HERO_PATH!r} has empty alt text — FR-036 mandates "
            "non-empty alt on every image; FR-022 makes the hero the critical case"
        )


def test_no_image_has_empty_alt_text() -> None:
    """FR-036: every ``![alt](path)`` in the README carries non-empty alt."""
    content = README.read_text(encoding="utf-8")
    images = _extract_images(content)

    empty = [path for alt, path in images if not alt.strip()]
    assert not empty, "images missing alt text (FR-036): " + ", ".join(repr(p) for p in empty)
