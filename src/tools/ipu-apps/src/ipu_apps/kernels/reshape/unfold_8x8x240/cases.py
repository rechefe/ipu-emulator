"""Runtime cases for unfold_8x8x240.

The input is staged through :func:`~ipu_apps.kernels.reshape.unfold_8x8x240.app.pack_input_rows`,
the single implementation of this kernel's row-permuting input contract.
``pad_value`` fills the source rows' padding lanes (64..127): the
``garbage_padding`` case proves they never reach the valid output.
"""
import numpy as np

from ipu_apps.kernel_registry.cases import KernelCase, PreparedCase
from ipu_apps.kernels.reshape.unfold_8x8x240.app import C, H, W, N_TOK, pack_input_rows
from ipu_apps.kernels.reshape.cases import MAX_CYCLES, check_streams, spatial_tensor


def prepare(workspace, *, seed, pad_value):
    x = spatial_tensor(seed, C, H, W)
    rows = pack_input_rows(x)
    rows[:, H * W:] = pad_value
    inp, out = workspace / "input.bin", workspace / "output.bin"
    inp.write_bytes(rows.tobytes())
    # teardown() crops each output row to its N_TOK valid tokens.
    return PreparedCase({"shape": (H, W, C)}, {"input_path": inp, "output_path": out},
                        lambda: check_streams(out, x, row_width=N_TOK, rtol=1e-4, atol=1e-3))


CASES = {
    "default": KernelCase(prepare, {"seed": 0x058, "pad_value": 0.0}, MAX_CYCLES),
    "garbage_padding": KernelCase(prepare, {"seed": 0x058, "pad_value": 1e3}, MAX_CYCLES),
}
