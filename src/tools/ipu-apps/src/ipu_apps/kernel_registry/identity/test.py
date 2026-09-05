"""Identity cases used by the shared runner and registry conformance tests."""
from pathlib import Path

import numpy as np

from ipu_emu.ipu import LANES
from ipu_apps.kernel_registry import resolve
from ipu_apps.kernel_registry.cases import KernelCase, PreparedCase, run_case


def prepare(workspace, *, rows):
    values = np.arange(rows * LANES, dtype=np.float32) - np.float32(LANES)
    inp, out = workspace / "input.bin", workspace / "output.bin"
    inp.write_bytes(values.tobytes())

    def check():
        assert out.read_bytes() == inp.read_bytes()

    return PreparedCase({"shape": (rows, LANES)},
                        {"input_path": inp, "output_path": out}, check)


CASES = {
    "default": KernelCase(prepare, {"rows": 3}),
    "single_row": KernelCase(prepare, {"rows": 1}),
}


def assert_identity_kernel(app_src: Path) -> None:
    verdict = resolve("identity", shape=(3, LANES))
    assert verdict.supported and verdict.app_name == "identity"
    _, cycles = run_case("identity", CASES["default"])
    assert cycles > 0


def test_identity():
    for case in CASES.values():
        _, cycles = run_case("identity", case)
        assert cycles > 0
