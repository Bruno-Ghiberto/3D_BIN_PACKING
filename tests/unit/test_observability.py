"""Unit tests for the observability helpers.

Covers the public surface of :mod:`bin_packer_3d.observability` —
`get_logger` namespacing + idempotent NullHandler attach, and
`StructuredAdapter`'s field-lifting behaviour.

This file is seeded in Foundational phase to keep `observability.py`
above the per-module coverage threshold once T107/T108 activate it.
US6 tasks T110..T113 extend this file with verbosity-flag and
explain-trace scenarios.
"""

from __future__ import annotations

import logging

from bin_packer_3d.observability import StructuredAdapter, get_logger


class TestGetLogger:
    """Tests for :func:`bin_packer_3d.observability.get_logger`."""

    def test_empty_name_returns_package_root_logger(self) -> None:
        """`get_logger("")` returns the `bin_packer_3d` top-level logger."""
        logger = get_logger("")
        assert logger.name == "bin_packer_3d"

    def test_named_logger_is_namespaced(self) -> None:
        """`get_logger("foo.bar")` returns `bin_packer_3d.foo.bar`."""
        logger = get_logger("foo.bar")
        assert logger.name == "bin_packer_3d.foo.bar"

    def test_null_handler_attached_once(self) -> None:
        """Repeated calls are idempotent — NullHandler is added only once."""
        logger = get_logger("idempotent.test")
        null_count_1 = sum(1 for h in logger.handlers if isinstance(h, logging.NullHandler))
        # Call again; NullHandler count must not grow.
        get_logger("idempotent.test")
        null_count_2 = sum(1 for h in logger.handlers if isinstance(h, logging.NullHandler))
        assert null_count_1 == null_count_2 == 1


class TestStructuredAdapter:
    """Tests for :class:`bin_packer_3d.observability.StructuredAdapter`."""

    def test_fields_are_lifted_into_extra(self) -> None:
        """`extra["fields"]` entries are lifted to top-level `extra` kwargs.

        Consumers who install a JSON formatter read machine-readable
        attributes directly off the `LogRecord` — so the adapter MUST
        hoist nested `fields` up into the `extra` dict that stdlib
        logging expects.
        """
        adapter = StructuredAdapter(get_logger("adapter.test"), {})
        msg, kwargs = adapter.process(
            "packing.start",
            {"extra": {"fields": {"algorithm": "ffd", "n_boxes": 42}}},
        )
        assert msg == "packing.start"
        assert kwargs["extra"] == {"algorithm": "ffd", "n_boxes": 42}

    def test_adapter_preserves_other_extra_keys(self) -> None:
        """Non-`fields` entries in `extra` are preserved verbatim."""
        adapter = StructuredAdapter(get_logger("adapter.test"), {})
        msg, kwargs = adapter.process(
            "packing.end",
            {"extra": {"trace_id": "abc-123", "fields": {"runtime": 0.5}}},
        )
        assert msg == "packing.end"
        assert kwargs["extra"]["trace_id"] == "abc-123"
        assert kwargs["extra"]["runtime"] == 0.5

    def test_adapter_handles_missing_fields(self) -> None:
        """`extra` without a `fields` key passes through unchanged."""
        adapter = StructuredAdapter(get_logger("adapter.test"), {})
        _, kwargs = adapter.process(
            "packing.attempt",
            {"extra": {"box_id": "B001"}},
        )
        assert kwargs["extra"] == {"box_id": "B001"}

    def test_adapter_creates_extra_when_absent(self) -> None:
        """If the caller omits `extra`, the adapter creates an empty one."""
        adapter = StructuredAdapter(get_logger("adapter.test"), {})
        _, kwargs = adapter.process("packing.reject", {})
        assert kwargs["extra"] == {}

    def test_adapter_emits_a_log_record_with_lifted_fields(self, caplog) -> None:
        """End-to-end: logging through the adapter surfaces fields on the record."""
        adapter = StructuredAdapter(get_logger("adapter.emit"), {})
        with caplog.at_level(logging.DEBUG, logger="bin_packer_3d.adapter.emit"):
            adapter.debug(
                "packing.attempt",
                extra={"fields": {"box_id": "B007", "orientation": "flat"}},
            )
        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.message == "packing.attempt"
        assert record.box_id == "B007"  # lifted out of fields
        assert record.orientation == "flat"
