"""Identity inputs and output checks, independent of pytest."""
import numpy as np

from ipu_emu.ipu import LANES
from ipu_apps.kernel_registry.cases import KernelCase, PreparedCase, check_output_bytes


def prepare(workspace, *, rows):
    values = np.arange(rows * LANES, dtype=np.float32) - np.float32(LANES)
    inp, out = workspace / "input.bin", workspace / "output.bin"
    inp.write_bytes(values.tobytes())

    def check():
        check_output_bytes(out, inp)

    return PreparedCase({"shape": (rows, LANES)},
                        {"input_path": inp, "output_path": out}, check)


CASES = {
    "default": KernelCase(prepare, {"rows": 3}),
    "single_row": KernelCase(prepare, {"rows": 1}),
}
