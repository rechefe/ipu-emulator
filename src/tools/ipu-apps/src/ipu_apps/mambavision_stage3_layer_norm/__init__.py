"""MambaVision Stage 3 Layer Norm (wide-vector FP32 mode).

Validates one real row from the MambaVision Excel mapping (rows 24 and 36,
same shape): Layer Norm over 196 tokens x 320 channels, tiled 128+128+64.

Runs entirely in wide-vector FP32 mode (no INT8 quantization) per team
guidance -- rsqrt(var+eps) is a small fraction that would round to 0 under
normal INT8 ACTIVATE.QUANTIZE.
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
CHANNELS = 320
ELEM_BYTES = 4  # FP32

TOKEN_ROW_BYTES = CHANNELS * ELEM_BYTES  # 1280
TOKEN_LOOP_END_BYTES = TOKENS * TOKEN_ROW_BYTES  # 250880
TILE_BYTES = 128 * ELEM_BYTES  # 512
BETA_SUBOFFSET_BYTES = CHANNELS * ELEM_BYTES  # 1280, gamma/beta both 320 floats

# NOTE: CR/LR registers are only 20 bits wide (max 0xFFFFF) -- addresses must
# stay below 0x100000 or they silently truncate/wrap. (Confirmed the hard way:
# 0x100000 wrapped to 0x000000, colliding OUTPUT_BASE with INPUT_BASE.)
INPUT_BASE = 0x000000  # 250880 bytes (0x3D480)
OUTPUT_BASE = 0x040000  # 262144, comfortably past input's end
PARAMS_BASE = 0x080000  # 524288, comfortably past output's end
CONST_SCRATCH_BASE = 0x082000  # 528384, past params' end (2560 bytes)

MEAN_SCRATCH_OFFSET = 512
XMINUSMEAN_SCRATCH_OFFSET = 1024
VAR_SCRATCH_OFFSET = 1536


class MambavisionStage3LayerNormApp(IpuApp):
    """Layer Norm over MambaVision Stage 3 tokens (196 x 320), wide-vector FP32."""

    def __init__(
        self,
        *,
        inputs_path: str | Path,
        gamma_path: str | Path,
        beta_path: str | Path,
        const_path: str | Path,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.inputs_path = Path(inputs_path)
        self.gamma_path = Path(gamma_path)
        self.beta_path = Path(beta_path)
        self.const_path = Path(const_path)
        self.output_path = Path(self.output_path) if self.output_path is not None else None

    def setup(self, state: "IpuStateType") -> None:
        load_binary_to_xmem(state, self.inputs_path, INPUT_BASE, TOKEN_ROW_BYTES, TOKENS)

        gamma_bytes = Path(self.gamma_path).read_bytes()
        beta_bytes = Path(self.beta_path).read_bytes()
        state.xmem.write_address(PARAMS_BASE, gamma_bytes)
        state.xmem.write_address(PARAMS_BASE + BETA_SUBOFFSET_BYTES, beta_bytes)

        const_bytes = Path(self.const_path).read_bytes()
        state.xmem.write_address(CONST_SCRATCH_BASE, const_bytes)

        # CR0 (permanently 0) and CR1 (permanently 1) are hardware-reserved --
        # not set explicitly; used directly as ZERO / LITERAL_ONE in the asm.
        state.regfile.set_cr(2, OUTPUT_BASE)
        state.regfile.set_cr(3, PARAMS_BASE)
        state.regfile.set_cr(4, CONST_SCRATCH_BASE)
        state.regfile.set_cr(5, TOKEN_ROW_BYTES)
        state.regfile.set_cr(6, TOKEN_LOOP_END_BYTES)
        state.regfile.set_cr(7, TILE_BYTES)
        state.regfile.set_cr(8, 128)  # combined R0++R1 index of R1[0] = 1.0
        state.regfile.set_cr(9, 129)  # combined index of R1[1] = 1/320
        state.regfile.set_cr(10, 130)  # combined index of R1[2] = eps
        state.regfile.set_cr(11, encode_dstructure(valid_elements=1, partition=0))
        state.regfile.set_cr(12, encode_dstructure(valid_elements=64, partition=0))
        state.regfile.set_cr(13, 4)
        state.regfile.set_cr(15, encode_dstructure(valid_elements=128, partition=0))

    def teardown(self, state: "IpuStateType") -> None:
        if self.output_path is not None:
            dump_xmem_to_binary(
                state, self.output_path, OUTPUT_BASE, TOKEN_ROW_BYTES, TOKENS
            )

    def run(self, *, max_cycles: int = 5_000_000, **kwargs):
        state = kwargs.pop("state", None)
        if state is None:
            state = IpuState(
                wide_vector_debug=True,
                wide_vector_arithmetic=WideVectorArithmetic.FP32,
                wide_vector_quantize_output=False,
            )
        return super().run(max_cycles=max_cycles, state=state, **kwargs)
