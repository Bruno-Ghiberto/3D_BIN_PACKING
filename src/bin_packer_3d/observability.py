"""Observability helpers for the bin_packer_3d library.

Uses only the Python standard library per Constitution §V (Library
Citizenship) and ADR-0008. No third-party logging framework is imported
or depended upon.

Usage inside the library::

    from bin_packer_3d.observability import get_logger

    logger = get_logger(__name__)   # or get_logger("algorithms.ffd")
    logger.debug("packing.start", extra={"algorithm": "ffd"})

Consumers are free to configure the `bin_packer_3d` logger (or any
descendant) however they like — add a `StreamHandler`, attach a JSON
formatter, wire it into `structlog`, etc. Until they do, the
`NullHandler` attached at import time silently drops messages, avoiding
the "No handlers could be found for logger X" warning.

The optional `StructuredAdapter` lifts machine-readable fields into the
log record so JSON-style consumers can emit structured output without
the library knowing about the formatter.
"""

from __future__ import annotations

import logging
from collections.abc import MutableMapping
from typing import Any

__all__ = ["get_logger", "StructuredAdapter"]

_PACKAGE_ROOT = "bin_packer_3d"


def get_logger(name: str = "") -> logging.Logger:
    """Return a namespaced logger with a `NullHandler` attached.

    The returned logger is always a child of the ``bin_packer_3d``
    namespace, so consumers can configure all library logging by
    targeting that single parent.

    Args:
        name: Sub-module name. Pass ``""`` (default) to get the
            package-level logger. Pass a dotted path (e.g.
            ``"algorithms.ffd"``) to get a child logger.

    Returns:
        A `logging.Logger` whose full name is either
        ``"bin_packer_3d"`` (when ``name == ""``) or
        ``"bin_packer_3d.<name>"``.

    Notes:
        The first call for a given logger attaches a `NullHandler` so
        consumers that have not configured logging never see
        "no handlers" warnings. Subsequent calls are no-ops on the
        handler — repeated `get_logger` is cheap and idempotent.
    """
    full_name = _PACKAGE_ROOT if not name else f"{_PACKAGE_ROOT}.{name}"
    logger = logging.getLogger(full_name)
    # Attach NullHandler once per logger. Check by type rather than identity
    # so manual reconfiguration by consumers is not clobbered.
    if not any(isinstance(h, logging.NullHandler) for h in logger.handlers):
        logger.addHandler(logging.NullHandler())
    return logger


class StructuredAdapter(logging.LoggerAdapter):  # type: ignore[type-arg]
    """Lift ``extra['fields']`` into LogRecord attributes for machine readers.

    Python's stdlib `logging.Logger.info(msg, extra={...})` already
    merges the ``extra`` dict into the `LogRecord`, so JSON formatters
    can read the fields via `record.__dict__`. This adapter lets callers
    nest fields under ``extra={"fields": {...}}`` for a consistent
    structured-logging shape that matches ADR-0008.

    Example::

        logger = StructuredAdapter(get_logger("benchmark.runner"), {})
        logger.debug(
            "packing.start",
            extra={"fields": {"algorithm": "ffd", "n_boxes": 42}},
        )

    Consumers who install a JSON formatter on the logger can then read
    ``record.algorithm`` and ``record.n_boxes`` directly.
    """

    def process(
        self,
        msg: Any,
        kwargs: MutableMapping[str, Any],
    ) -> tuple[Any, MutableMapping[str, Any]]:
        """Merge ``extra['fields']`` into the top-level ``extra`` dict."""
        extra = kwargs.setdefault("extra", {})
        fields = extra.pop("fields", None)
        if fields:
            extra.update(fields)
        return msg, kwargs
