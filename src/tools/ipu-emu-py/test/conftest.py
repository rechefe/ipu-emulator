"""pytest configuration shared by every emulator test."""

from __future__ import annotations

import pytest

from ipu_as.label import reset_labels


@pytest.fixture(autouse=True)
def _reset_assembler_labels():
    """Isolate the assembler's global label table around every test.

    Deleted when E05 removes the global label table.
    """
    reset_labels()
    yield
    reset_labels()
