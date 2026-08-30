"""MLP-residual test harness — the second residual of ``Block.forward``.

Implements::

    x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))

``skip`` is the tensor the mixer residual produced, ``branch`` is the timm
``Mlp`` output (``fc1`` -> GELU -> ``fc2``, hidden width ``4 * d_model``);
both are token view ``(L, d_model)``. The MLP itself is not part of this
kernel: fc1 and fc2 are the fully_connected app and the GELU is an
``ACTIVATE.QUANTIZE`` activation code, so only the closing residual needed
its own program.

The harness:

1. Loads the skip and branch buffers into XMEM
2. Sets CR registers for base addresses, the row count, gamma and dtype
3. Runs the assembly program
4. Dumps the summed buffer from XMEM

Usage::

    from ipu_apps.residual_mlp import ResidualMlpApp

    app = ResidualMlpApp(
        inst_path="residual_mlp.bin",
        inputs_path="skip_in_int8.bin",
        branch_path="branch_in_int8.bin",
        output_path="out.bin",
        stage="stage3",
    )
    state, cycles = app.run()
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ipu_emu.ipu_math import DType
from ipu_emu.emulator import load_binary_to_xmem

from ipu_apps.base import IpuApp

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

# -- Constants --------------------------------------------------------------

ROW_ELEMS = 128
ROW_BYTES = 128

#: ``(d_model, seq_len)`` for the two ``mamba_vision_T`` stages that contain
#: MambaVisionMixer blocks.
STAGES = {
    "stage3": (320, 196),
    "stage4": (640, 49),
}

#: XMEM row numbers (one row = 128 bytes). Narrow (INT8) mode may only
#: address the first 16384 rows.
SKIP_BASE_ROW = 1024
BRANCH_BASE_ROW = 3072
OUT_BASE_ROW = 5120

STORE_SLACK_ROWS = 3


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def parse_stage(stage: str) -> tuple[int, int]:
    """Parse a stage name into ``(d_model, seq_len)``."""
    try:
        return STAGES[stage]
    except KeyError:
        raise ValueError(
            f"Invalid stage '{stage}'. Supported: {', '.join(STAGES)}"
        ) from None


class ResidualMlpApp(IpuApp):
    """MLP-residual application harness.

    Args:
        inst_path:   Path to assembled instruction binary.
        inputs_path: Path to the skip tensor (the mixer residual's output).
        branch_path: Path to the MLP output being added in, token view.
        output_path: Optional path to write output.
        stage:       ``'stage3'`` or ``'stage4'`` of ``mamba_vision_T``.
        gamma:       ``layer_scale`` coefficient ``gamma_2``. ``mamba_vision_T``
            sets no layer_scale, so ``Block.gamma_2`` is the int 1; pass
            something else only for a layer-scaled variant.
        channels:    Channel count; defaults to ``d_model``.
    """

    def __init__(
        self,
        *,
        stage: str = "stage3",
        branch_path=None,
        gamma: int = 1,
        channels: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if branch_path is None:
            raise ValueError("residual_mlp needs both addends: pass branch_path=...")
        self.inputs_path = Path(self.inputs_path)
        self.branch_path = Path(branch_path)
        self.stage = stage
        self.gamma = gamma

        d_model, seq_len = parse_stage(stage)
        self.d_model = d_model
        self.seq_len = seq_len
        self.channels = d_model if channels is None else channels

        self.rows_per_token = ceil_div(self.channels, ROW_ELEMS)
        self.n_rows = seq_len * self.rows_per_token

    def setup(self, state: "IpuState") -> None:
        state.dtype = DType.INT8
        state.set_cr_dstructure(valid_elements=ROW_ELEMS, partition=0)

        load_binary_to_xmem(
            state, self.inputs_path, SKIP_BASE_ROW * ROW_BYTES, ROW_BYTES, self.n_rows
        )
        load_binary_to_xmem(
            state, self.branch_path, BRANCH_BASE_ROW * ROW_BYTES, ROW_BYTES, self.n_rows
        )
        # The loop prefetches one skip row past the last it uses.
        state.xmem.write_address((SKIP_BASE_ROW + self.n_rows) * ROW_BYTES,
                                 bytearray(ROW_BYTES))
        state.xmem.write_address(
            OUT_BASE_ROW * ROW_BYTES,
            bytearray((self.n_rows + STORE_SLACK_ROWS) * ROW_BYTES),
        )

        # CR0=0 and CR1=1 permanently; CR1 is the multiply-by-one scalar for
        # the skip term, CR6 carries gamma_2 for the branch term, so the
        # kernel needs no constants row in XMEM.
        state.regfile.set_cr(2, SKIP_BASE_ROW)
        state.regfile.set_cr(3, OUT_BASE_ROW)
        state.regfile.set_cr(4, BRANCH_BASE_ROW)
        state.regfile.set_cr(5, self.n_rows)
        state.regfile.set_cr(6, self.gamma)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            data = state.xmem.read_address(
                OUT_BASE_ROW * ROW_BYTES, self.n_rows * ROW_BYTES
            )
            Path(self.output_path).write_bytes(bytes(data))
