"""Contract tests for the algorithm registry (T016).

Per ``data-model.md`` §new entities and ``contracts/api.md``, the
``ALGORITHMS`` dict is the single source of truth for valid strategy
keys. This file asserts that ``PackerConfig.strategy`` validation stays
coherent with the registry and that duplicate registration is rejected.

Exercises: FR-001, FR-002, FR-006, FR-047.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bin_packer_3d.algorithms import ALGORITHMS, get_strategies, register
from bin_packer_3d.algorithms.base import PackerBase
from bin_packer_3d.config import PackerConfig


class TestAlgorithmRegistry:
    """The registry IS the contract — config validation must stay in lockstep."""

    def test_config_strategies_match_registry(self) -> None:
        """PackerConfig.get_strategies() mirrors ALGORITHMS.keys() (FR-001, FR-002)."""
        assert set(PackerConfig.get_strategies()) == set(ALGORITHMS.keys())

    def test_module_get_strategies_matches_registry(self) -> None:
        """Module-level get_strategies() helper mirrors ALGORITHMS.keys()."""
        assert set(get_strategies()) == set(ALGORITHMS.keys())

    def test_duplicate_registration_raises(self) -> None:
        """Registering a name already in ALGORITHMS raises ValueError (FR-006)."""
        existing_name = next(iter(ALGORITHMS))

        with pytest.raises(ValueError):

            @register(existing_name)
            class _DuplicateProbe(PackerBase):
                @property
                def name(self) -> str:
                    return "duplicate"

                def _pack_impl(self, boxes):  # type: ignore[no-untyped-def]
                    raise NotImplementedError

    def test_unknown_strategy_error_lists_registered_names(self) -> None:
        """PackerConfig(strategy='bogus') error lists every registered name (FR-047)."""
        with pytest.raises(ValidationError) as exc_info:
            PackerConfig(strategy="bogus_strategy_xyz")

        msg = str(exc_info.value)
        for registered in ALGORITHMS:
            assert registered in msg, (
                f"error message must list registered strategy {registered!r}; got: {msg}"
            )
