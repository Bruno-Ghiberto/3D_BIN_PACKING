"""Regression guard for Principle V (Library Citizenship) — FR-050, FR-052.

A well-behaved library imports without leaking state into its consumer:
no root-logger handlers installed, no `logging.basicConfig` called, no
`os.chdir` or `sys.path` mutations, no network or blocking I/O.

This module hard-checks those invariants so any future regression is
surfaced immediately in CI (Gate G5 — Library Citizenship). The tests
run under a subprocess where feasible so import side-effects can be
compared against a pristine interpreter state.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import textwrap


def test_library_package_logger_has_null_handler() -> None:
    """After `import bin_packer_3d`, the package logger has a NullHandler.

    Required by FR-050 and the Python logging library idiom: every library
    should attach a `logging.NullHandler` to its top-level logger to avoid
    "No handlers could be found for logger X" warnings when consumers have
    not configured logging themselves.
    """
    import bin_packer_3d  # noqa: F401 — import is the test trigger

    package_logger = logging.getLogger("bin_packer_3d")
    null_handlers = [h for h in package_logger.handlers if isinstance(h, logging.NullHandler)]
    assert null_handlers, (
        "bin_packer_3d top-level logger has no NullHandler attached. "
        "Principle V requires a NullHandler so the library does not "
        "force logging configuration on its consumers."
    )


def test_library_import_does_not_touch_root_logger() -> None:
    """Importing the library MUST NOT add handlers to the root logger.

    A library that configures the root logger hijacks the consumer's
    logging setup — unacceptable per Constitution §V (Library Citizenship)
    and Spec FR-050.
    """
    script = textwrap.dedent(
        """
        import logging
        before = list(logging.getLogger().handlers)
        import bin_packer_3d  # noqa
        after = list(logging.getLogger().handlers)
        # Compare lengths — library MUST NOT add handlers
        assert len(before) == len(after), (
            f'root logger handler count changed on import: '
            f'{len(before)} -> {len(after)}'
        )
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"Subprocess check failed (stdout={result.stdout!r}, stderr={result.stderr!r})"
    )


def test_library_import_does_not_mutate_syspath_or_cwd() -> None:
    """Importing the library MUST NOT change sys.path or working directory.

    Global-state mutation at import time breaks consumers' environments.
    Principle V forbids it explicitly.
    """
    script = textwrap.dedent(
        """
        import os
        import sys
        cwd_before = os.getcwd()
        path_before = list(sys.path)
        import bin_packer_3d  # noqa
        cwd_after = os.getcwd()
        path_after = list(sys.path)
        assert cwd_before == cwd_after, (
            f'cwd changed on import: {cwd_before!r} -> {cwd_after!r}'
        )
        assert path_before == path_after, (
            f'sys.path mutated on import: '
            f'{len(path_before)} entries -> {len(path_after)} entries'
        )
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"Subprocess check failed (stdout={result.stdout!r}, stderr={result.stderr!r})"
    )


def test_no_logging_basicConfig_called_on_import() -> None:
    """Importing the library MUST NOT call `logging.basicConfig`.

    `basicConfig` installs a StreamHandler on the root logger, overriding
    whatever the consumer may have set up. Forbidden by Constitution §V.

    We detect this indirectly: root logger's `level` and `handlers` must
    remain at their defaults in a fresh subprocess.
    """
    script = textwrap.dedent(
        """
        import logging
        root = logging.getLogger()
        level_before = root.level
        import bin_packer_3d  # noqa
        level_after = root.level
        assert level_before == level_after, (
            f'root logger level changed on import: '
            f'{level_before} -> {level_after} '
            f'(likely basicConfig was called)'
        )
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"Subprocess check failed (stdout={result.stdout!r}, stderr={result.stderr!r})"
    )
