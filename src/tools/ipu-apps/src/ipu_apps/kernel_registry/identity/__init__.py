"""Minimal registered kernel and memory-only Python harness.

The harness deliberately does not calculate an identity result, reshape data,
or prepare a kernel-specific layout. It only:

1. loads a preformatted row-major matrix from a binary file;
2. configures the kernel's CRs;
3. runs the assembled kernel through :class:`~ipu_apps.base.IpuApp`; and
4. reads the output matrix into a binary file.

The input and output files are little-endian FP32 matrices with shape
``(rows, 128)``. Each matrix row maps directly to one XMEM row.
"""

from __future__ import annotations

from pathlib import Path

from ipu_emu.emulator import dump_xmem_to_binary, load_binary_to_xmem
from ipu_emu.ipu import LANES, R_ACC_SIZE
from ipu_emu.ipu_state import IpuState
from ipu_emu.xmem import XMEM_SIZE_BYTES

from ipu_apps.base import IpuApp
from ipu_apps.kernel_registry import OUTPUT, ExecutionConfig, KernelSpec, ShapeBundle, no, yes

INPUT_BASE_ADDR = 0
OUTPUT_BASE_ADDR = XMEM_SIZE_BYTES // 2


class IdentityApp(IpuApp):
    """Load, run, and read an ``(rows, 128)`` FP32 identity operation."""

    def __init__(
        self,
        *,
        input_path: str | Path,
        shape=(1, LANES),
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(input_path)
        self.shape = tuple(int(dim) for dim in shape)
        SPEC.guard(shape=self.shape)
        self.rows = self.shape[0]

        size = self.input_path.stat().st_size
        expected_size = self.rows * R_ACC_SIZE
        if size != expected_size:
            raise ValueError(
                f"input file for shape {self.shape} must contain {expected_size} "
                f"bytes; got {size} bytes"
            )


    def setup(self, state: IpuState) -> None:
        load_binary_to_xmem(
            state,
            self.input_path,
            INPUT_BASE_ADDR,
            R_ACC_SIZE,
            max_chunks=self.rows,
        )
        state.regfile.set_cr(2, INPUT_BASE_ADDR // R_ACC_SIZE)
        state.regfile.set_cr(3, OUTPUT_BASE_ADDR // R_ACC_SIZE)
        state.regfile.set_cr(4, self.rows)
        state.set_cr_dstructure(valid_elements=LANES)

    def teardown(self, state: IpuState) -> None:
        if self.output_path is not None:
            dump_xmem_to_binary(
                state,
                self.output_path,
                OUTPUT_BASE_ADDR,
                R_ACC_SIZE,
                num_chunks=self.rows,
            )



def _shape(params) -> tuple[int, ...]:
    return tuple(int(dim) for dim in params["shape"])


def _supports(**params):
    shape = _shape(params)
    if len(shape) != 2:
        return no(f"expects a rank-2 FP32 matrix shaped (rows, {LANES}); got {shape}")
    rows, columns = shape
    if rows < 1:
        return no(f"needs at least one matrix row; got {rows}")
    if columns != LANES:
        return no(f"expects {LANES} FP32 columns per XMEM row; got {columns}")
    if rows * R_ACC_SIZE > OUTPUT_BASE_ADDR:
        return no(
            f"input matrix needs {rows * R_ACC_SIZE} bytes, exceeding its "
            f"{OUTPUT_BASE_ADDR}-byte half of XMEM"
        )
    return yes()


def _bundle(**params):
    shape = _shape(params)
    return ShapeBundle.of(input=shape).with_shapes(derived={OUTPUT: shape})


SPEC = KernelSpec(
    execution=ExecutionConfig(mode="fp32"),
    name="identity",
    op="identity",
    variant="fp32_matrix",
    app_class=IdentityApp,
    asm="identity.asm",
    requires=("shape",),
    supports=_supports,
    build=lambda **params: {"shape": _shape(params)},
    explain=lambda **params: (
        f"a {_shape(params)[0]} x {LANES} FP32 matrix is copied without changing layout"
    ),
    bundle=_bundle,
    tags=("example", "fp32-wide"),
)
