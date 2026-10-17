"""What every baseline updater and baseline test shares (code conventions, "Baselines").

* ``workspace``: the checkout an ``update_*`` target writes into, which is
  where ``bazel run`` started.
* ``fail_if_changed``: a test's failure when the current code's output no
  longer matches its committed snapshot, with the diff and the command that
  rewrites the snapshot.
"""

from __future__ import annotations

import difflib
import os
import sys
from pathlib import Path

import pytest


def workspace(update_command: str) -> Path | None:
    """The workspace ``bazel run`` started in.

    Args:
        update_command: The updater's ``bazel run`` command, printed when it
            was run some other way.

    Returns:
        The workspace root; None, after printing ``update_command`` to
        stderr, outside ``bazel run``.
    """
    root = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if not root:
        print(f"Run this through Bazel: {update_command}", file=sys.stderr)
        return None
    return Path(root)


def fail_if_changed(
    expected: str,
    actual: str,
    *,
    name: str,
    what: str,
    update_command: str,
    max_lines: int | None = None,
) -> None:
    """Fail the test, with a diff, unless ``actual`` equals the snapshot ``expected``.

    Args:
        expected: The committed snapshot's text.
        actual: The current code's output.
        name: The snapshot's name, for the diff header.
        what: What changed, for the message (``"word_fields.json"``).
        update_command: The command that rewrites the snapshot.
        max_lines: Show at most this many diff lines; None for all.

    Raises:
        pytest.fail.Exception: ``actual`` differs from ``expected``.
    """
    if actual == expected:
        return
    diff = list(
        difflib.unified_diff(
            expected.splitlines(keepends=True),
            actual.splitlines(keepends=True),
            f"{name} (snapshot)",
            f"{name} (current code)",
        )
    )
    shown = "".join(diff if max_lines is None else diff[:max_lines])
    if max_lines is not None and len(diff) > max_lines:
        shown += f"... {len(diff) - max_lines} more diff lines\n"
    pytest.fail(f"{what} changed. If intended, run {update_command}\n{shown}")
