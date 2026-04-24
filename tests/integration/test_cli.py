"""Integration tests for the CLI (T017).

Exercises FR-003: ``bin-packer info`` lists exactly the registered
algorithms with their complexity class and description, sourced from
the ``ALGORITHMS`` registry at runtime — not hardcoded strings.
"""

from __future__ import annotations

from click.testing import CliRunner

from bin_packer_3d.algorithms import ALGORITHMS
from bin_packer_3d.cli import main


class TestInfoCommand:
    """``bin-packer info`` must stay in lockstep with the registry."""

    def test_info_lists_registry(self) -> None:
        """Every ALGORITHMS key appears in info output with complexity + description (FR-003)."""
        runner = CliRunner()
        result = runner.invoke(main, ["info"])

        assert result.exit_code == 0, f"info exited non-zero: {result.output!r}"
        output = result.output

        for name, cls in ALGORITHMS.items():
            assert name in output, f"registered strategy {name!r} missing from info output"
            assert (
                cls.complexity in output
            ), f"complexity for {name!r} missing from info output; expected {cls.complexity!r}"
            assert (
                cls.description in output
            ), f"description for {name!r} missing from info output; expected {cls.description!r}"
