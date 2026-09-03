"""Kernel-specific check imported by the generic registry test suite."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from ipu_as.lark_tree import assemble_to_bin_file
from ipu_emu.ipu import LANES
from ipu_apps.kernel_registry import resolve


def assert_identity_kernel(app_src: Path) -> None:
    """Resolve, assemble, load, run, and read the identity kernel."""
    rows = 3
    verdict = resolve("identity", shape=(rows, LANES))
    assert verdict.supported
    assert verdict.app_name == "identity"

    asm = next(app_src.rglob(verdict.kernel.asm))
    values = np.arange(rows * LANES, dtype=np.float32) - np.float32(LANES)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        inst_path = tmp_path / "identity.bin"
        input_path = tmp_path / "input.bin"
        output_path = tmp_path / "output.bin"
        assemble_to_bin_file(asm.read_text(), str(inst_path))
        input_path.write_bytes(values.tobytes())

        app = verdict.kernel.app_class(
            inst_path=inst_path,
            input_path=input_path,
            output_path=output_path,
            **verdict.kwargs,
        )
        _, cycles = app.run()

        assert cycles > 0
        assert output_path.read_bytes() == input_path.read_bytes()
