"""End-to-end visualisation snapshot test (T035, US2 Independent-Test gate).

Drives the contract in
``specs/002-portfolio-polish/contracts/visualisation-theme.md``: two
consecutive ``bin-packer pack`` runs on identical input must produce
byte-identical HTML and pixel-equal static-export PNG. The rendered HTML
embeds the branded title format + the four-key stats overlay (algorithm,
boxes placed, utilisation %, runtime).

These tests are red until T037 (theme + ``Placement.colour`` binding +
stats overlay), T038 (Kaleido static-export branch with ImportError
fallback), and T039 (``--static-format`` CLI flag) all land.

Source FRs: FR-006 (visual reproducibility), FR-007 (stats overlay),
FR-008 (runtime in overlay), FR-009 (static export). Source ADRs:
ADR-001 (Plotly + Kaleido pin), ADR-006 (theme), ADR-011 (overlay
layout). DR-11 precursor (kaleido<1.0.0) is required for the PNG
pathway to work against Plotly 5.x.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

pytest.importorskip("kaleido")  # viz extra required for static export

from click.testing import CliRunner  # noqa: E402

from bin_packer_3d.cli import main  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
HEADLINE_CSV = REPO_ROOT / "examples" / "headline.csv"


def _run_pack(output_dir: Path, static_format: str = "png") -> tuple[int, str]:
    """Invoke ``bin-packer pack`` against the headline dataset using BFD.

    Returns ``(exit_code, captured_output)`` for assertion clarity. Always
    runs with ``--visualize``; the visualisation pipeline is the surface
    under test.
    """
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "pack",
            str(HEADLINE_CSV),
            "--strategy",
            "bfd",
            "--visualize",
            "--static-format",
            static_format,
            "-o",
            str(output_dir),
        ],
    )
    return result.exit_code, result.output


class TestVisualisationE2E:
    """US2 Independent-Test gate — see Phase B'-1 prompt §5.2.1."""

    def test_html_byte_identical_across_runs(self, tmp_path: Path) -> None:
        """G-US2-IT-1: two consecutive runs produce byte-identical HTML.

        Driven by FR-006 + SC-005: visualisation reproducibility under
        the pinned Plotly + Kaleido combo (DR-11 precursor).
        """
        run1 = tmp_path / "run1"
        run2 = tmp_path / "run2"

        code1, out1 = _run_pack(run1)
        code2, out2 = _run_pack(run2)

        assert code1 == 0, f"first pack run failed (exit {code1}): {out1!r}"
        assert code2 == 0, f"second pack run failed (exit {code2}): {out2!r}"

        html1 = run1 / "bin_1.html"
        html2 = run2 / "bin_1.html"
        assert html1.exists(), (
            f"missing {html1}; directory contents: "
            f"{[p.name for p in run1.iterdir()] if run1.exists() else 'run1 missing'}"
        )
        assert html2.exists(), (
            f"missing {html2}; directory contents: "
            f"{[p.name for p in run2.iterdir()] if run2.exists() else 'run2 missing'}"
        )

        assert html1.read_bytes() == html2.read_bytes(), (
            "bin_1.html not byte-identical across two runs — "
            "non-determinism violates FR-006 + SC-005. "
            f"sizes: run1={html1.stat().st_size} run2={html2.stat().st_size}"
        )

    def test_png_pixel_equal_across_runs(self, tmp_path: Path) -> None:
        """G-US2-IT-2: two consecutive runs produce pixel-equal PNG static export.

        Driven by ADR-001 (Plotly + Kaleido pin) + DR-11 (kaleido<1.0.0).
        """
        run1 = tmp_path / "run1"
        run2 = tmp_path / "run2"

        code1, out1 = _run_pack(run1, static_format="png")
        code2, out2 = _run_pack(run2, static_format="png")

        assert code1 == 0, f"first pack run failed (exit {code1}): {out1!r}"
        assert code2 == 0, f"second pack run failed (exit {code2}): {out2!r}"

        png1 = run1 / "bin_1.png"
        png2 = run2 / "bin_1.png"
        assert png1.exists(), (
            f"missing {png1}; directory contents: "
            f"{[p.name for p in run1.iterdir()] if run1.exists() else 'run1 missing'}"
        )
        assert png2.exists(), (
            f"missing {png2}; directory contents: "
            f"{[p.name for p in run2.iterdir()] if run2.exists() else 'run2 missing'}"
        )

        digest1 = hashlib.sha256(png1.read_bytes()).hexdigest()
        digest2 = hashlib.sha256(png2.read_bytes()).hexdigest()
        assert digest1 == digest2, (
            "bin_1.png SHA256 mismatch across runs — "
            f"run1: {digest1[:16]}... run2: {digest2[:16]}... "
            f"sizes: run1={png1.stat().st_size} run2={png2.stat().st_size}"
        )

    def test_branded_title_format_in_html(self, tmp_path: Path) -> None:
        """G-US2-IT-3 part 1: rendered HTML contains the branded title format.

        Per ``VisualisationStyle.title_format`` =
        ``"{algorithm} — {dataset}"`` (theme.py), the title for a BFD
        pack of ``headline.csv`` must contain both algorithm and dataset
        tokens. The em-dash (U+2014) separator is part of the format
        constant.
        """
        out_dir = tmp_path / "run"
        code, output = _run_pack(out_dir)
        assert code == 0, f"pack failed: {output!r}"

        html_text = (out_dir / "bin_1.html").read_text()

        assert re.search(r"\bbfd\b", html_text, re.IGNORECASE), (
            "branded title missing algorithm token 'bfd'"
        )
        assert "headline" in html_text.lower(), "branded title missing dataset token 'headline'"
        assert "—" in html_text, (
            "branded title em-dash separator '—' (U+2014) missing — "
            "title_format from theme.py not applied"
        )

    def test_stats_overlay_in_html(self, tmp_path: Path) -> None:
        """G-US2-IT-3 part 2: rendered HTML contains the four-key stats overlay.

        Source: FR-007 + FR-008 + ADR-011. The overlay is a Plotly
        Annotation positioned per ``VisualisationStyle.stats_overlay_layout``;
        its text content must embed labels for: algorithm name, boxes
        placed (``N/total``), overall utilisation %, and runtime in ms.
        """
        out_dir = tmp_path / "run"
        code, output = _run_pack(out_dir)
        assert code == 0, f"pack failed: {output!r}"

        html_text = (out_dir / "bin_1.html").read_text()

        # Each row asserts one of the four overlay fields by a permissive
        # case-insensitive regex; exact label wording is implementation
        # discretion within the contract.
        for label, pattern in [
            ("algorithm", r"algorithm"),
            ("boxes-placed", r"boxes|placed"),
            ("utilisation", r"util[ai][sz]ation|%"),
            ("runtime", r"runtime|elapsed|\bms\b"),
        ]:
            assert re.search(pattern, html_text, re.IGNORECASE), (
                f"stats overlay missing the {label!r} field "
                f"(pattern {pattern!r} not found in rendered HTML)"
            )

    def test_svg_static_export_when_requested(self, tmp_path: Path) -> None:
        """G-US2-IT-5: ``--static-format svg`` produces a ``.svg`` (not ``.png``).

        Driven by T039 + ADR-008 (CLI ``--static-format`` flag).
        """
        out_dir = tmp_path / "run"
        code, output = _run_pack(out_dir, static_format="svg")
        assert code == 0, f"pack failed with --static-format svg: {output!r}"

        svg_files = list(out_dir.glob("bin_*.svg"))
        png_files = list(out_dir.glob("bin_*.png"))

        listing = [p.name for p in out_dir.iterdir()] if out_dir.exists() else []
        assert svg_files, (
            f"no bin_*.svg under {out_dir} when --static-format svg "
            f"requested; directory listing: {listing}"
        )
        assert not png_files, (
            f"unexpected bin_*.png under {out_dir} when --static-format svg "
            f"requested; directory listing: {listing}"
        )
