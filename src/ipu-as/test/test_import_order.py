"""Derived operand widths do not depend on which module is imported first (E03-08).

Before E03-08 the union solver wrote ``LR_INC_DEC_IMM_FIELD_BITS`` and
``RESHAPE_MASK_FIELD_BITS`` into their modules at import time, so a module
that bound them before ``instruction_spec`` was imported saw ``0``.
"""

from __future__ import annotations

import os
import subprocess
import sys


def test_immediates_imported_first_see_the_packed_widths():
    script = (
        "import ipu_as.immediate as imm\n"
        "print(imm.LrIncDecImmediate.bits(), imm.LrOrReshapeMaskImmediate.bits())\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    assert result.stdout.split() == ["8", "5"]
