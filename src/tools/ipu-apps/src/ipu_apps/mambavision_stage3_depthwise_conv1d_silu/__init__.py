"""MambaVision Stage 3 depthwise Conv1D + SiLU (wide-vector FP32 mode).

Validates rows 27/28 from the MambaVision Excel mapping: depthwise 1D conv
(kernel size 3, non-causal, padding='same') + SiLU, over 196 tokens x 160
channels, tiled 128+32. Same kernel run once per branch (x, z) -- pass
different weight/input file paths, everything else is identical.

Runs in wide-vector FP32 mode (switched from an initial INT32 attempt,
2026-07-10): the conv accumulation itself is unaffected (inputs/weights are
integer-valued and exactly representable in float32, well under the 2^24
exact-integer range, so the sum stays bit-exact regardless of arithmetic
mode) -- but SiLU's output is genuinely fractional for inputs near zero
(e.g. SiLU(1) =~ 0.73), and this op sits right after Layer Norm, which
normalizes activations into roughly that same near-zero range. INT32 would
round that fraction away every time, the same class of precision loss Layer
Norm's rsqrt had, just narrower in scope. FP32 preserves it for free.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ipu_apps.base import IpuApp
from ipu_emu.emulator import dump_xmem_to_binary, load_binary_to_xmem
from ipu_emu.ipu_config import encode_dstructure
from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState as IpuStateType


TOKENS = 196
CHANNELS = 160
ELEM_BYTES = 4  # FP32

ROW_BYTES = CHANNELS * ELEM_BYTES  # 640
PADDED_ROWS = TOKENS + 2  # +1 zero row before, +1 zero row after
LOOP_START_BYTES = ROW_BYTES  # first "center" row = padded row 1 = token 0
LOOP_END_BYTES = ROW_BYTES * (TOKENS + 1)  # padded row 197 = one past token 195
TILE1_SUBOFFSET = 128 * ELEM_BYTES  # 512
TAPZERO_SUBOFFSET = ROW_BYTES  # 640
TAPPLUS1_SUBOFFSET = ROW_BYTES * 2  # 1280

# NOTE: CR/LR registers are only 20 bits wide -- keep every base address well
# under 0x100000 (see ipu_isa_gotchas memory note / mambavision_stage3_layer_norm).
INPUT_BASE = 0x000000  # 198 rows x 640 bytes = 126720 bytes (padded)
OUTPUT_BASE = 0x020000  # 131072, past input's end
TAPS_BASE = 0x040000  # 262144, past output's end (3 x 640 = 1920 bytes)


class MambavisionStage3DepthwiseConv1dSiluApp(IpuApp):
    """Depthwise Conv1D(k=3) + SiLU over one branch (196 x 160), wide-vector FP32."""

    def __init__(
        self,
        *,
        inputs_padded_path: str | Path,
        tap_minus1_path: str | Path,
        tap_zero_path: str | Path,
        tap_plus1_path: str | Path,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.inputs_padded_path = Path(inputs_padded_path)
        self.tap_minus1_path = Path(tap_minus1_path)
        self.tap_zero_path = Path(tap_zero_path)
        self.tap_plus1_path = Path(tap_plus1_path)
        self.output_path = Path(self.output_path) if self.output_path is not None else None

    def setup(self, state: "IpuStateType") -> None:
        load_binary_to_xmem(state, self.inputs_padded_path, INPUT_BASE, ROW_BYTES, PADDED_ROWS)

        state.xmem.write_address(TAPS_BASE, Path(self.tap_minus1_path).read_bytes())
        state.xmem.write_address(TAPS_BASE + TAPZERO_SUBOFFSET, Path(self.tap_zero_path).read_bytes())
        state.xmem.write_address(TAPS_BASE + TAPPLUS1_SUBOFFSET, Path(self.tap_plus1_path).read_bytes())

        # CR0 (permanently 0) used directly for ZERO / INPUT_BASE -- not set explicitly.
        state.regfile.set_cr(2, OUTPUT_BASE)
        state.regfile.set_cr(3, TAPS_BASE)
        state.regfile.set_cr(4, ROW_BYTES)
        state.regfile.set_cr(5, LOOP_START_BYTES)
        state.regfile.set_cr(6, LOOP_END_BYTES)
        state.regfile.set_cr(7, TILE1_SUBOFFSET)
        state.regfile.set_cr(8, TAPZERO_SUBOFFSET)
        state.regfile.set_cr(9, TAPPLUS1_SUBOFFSET)
        state.regfile.set_cr(10, encode_dstructure(valid_elements=32, partition=0))
        state.regfile.set_cr(15, encode_dstructure(valid_elements=128, partition=0))

    def teardown(self, state: "IpuStateType") -> None:
        if self.output_path is not None:
            dump_xmem_to_binary(state, self.output_path, OUTPUT_BASE, ROW_BYTES, TOKENS)

    def run(self, *, max_cycles: int = 5_000_000, **kwargs):
        state = kwargs.pop("state", None)
        if state is None:
            state = IpuState(
                wide_vector_debug=True,
                wide_vector_arithmetic=WideVectorArithmetic.FP32,
                wide_vector_quantize_output=False,
            )
        return super().run(max_cycles=max_cycles, state=state, **kwargs)
