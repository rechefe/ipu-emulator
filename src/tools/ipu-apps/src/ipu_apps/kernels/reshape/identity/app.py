"""Minimal registered kernel: copies an ``(rows, 128)`` FP32 matrix unchanged.

The smallest complete example of the memory-only harness contract. The input
file is the preformatted XMEM image (here, just the matrix rows); the output
rows follow it in XMEM, and :class:`MemoryApp` loads, configures CRs and dumps
them. The only kernel-specific code is the CR map in ``memory_layout``:

  CR2 = input base row, CR3 = output base row, CR4 = row count.
"""
from ipu_emu.ipu import LANES

from ipu_apps.kernel_registry import OUTPUT, ShapeBundle
from ipu_apps.kernel_registry.memory import MemoryApp, MemoryLayout, memory_spec


class IdentityApp(MemoryApp):
    @classmethod
    def memory_layout(cls, *, shape):
        shape = tuple(shape)
        if len(shape) != 2:
            raise ValueError(f"expects a rank-2 FP32 matrix shaped (rows, {LANES}); got {shape}")
        rows, columns = shape
        if type(rows) is not int or rows < 1:
            raise ValueError(f"needs at least one matrix row; got {rows}")
        if columns != LANES:
            raise ValueError(f"expects {LANES} FP32 columns per XMEM row; got {columns}")
        return MemoryLayout(rows, rows, {2: 0, 3: rows, 4: rows})


SPEC = memory_spec(
    "identity", IdentityApp,
    variant="fp32_matrix",
    explain=lambda **params: (
        f"a {tuple(params['shape'])[0]} x {LANES} FP32 matrix is copied without changing layout"
    ),
    caveats=lambda **params: (),
    bundle=lambda **params: ShapeBundle.of(input=tuple(params["shape"])).with_shapes(
        derived={OUTPUT: tuple(params["shape"])}),
    tags=("example", "fp32-wide"),
)
