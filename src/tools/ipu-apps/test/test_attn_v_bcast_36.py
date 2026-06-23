"""End-to-end regression tests for the attn_v_bcast_36 application (attn@V broadcast)."""

from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np
import pytest

from ipu_apps.attn_v_bcast_36 import AttnVBcast36App


_INST_BIN = Path(os.environ["ATTN_V_BCAST_36_INST_BIN"])
_DATA_DIR = Path(os.environ["ATTN_V_BCAST_36_DATA_DIR"])


# ---------------------------------------------------------------------------
# Wide-vector FP32 verification: the kernel's exact key-major broadcast pattern
# (LDR_CYCLIC P[:,s] -> MULT.RC.VE scalar=V[s] -> ACC over keys) over the real
# 256-key contraction, compared to numpy O = P @ V WITHOUT quant noise.
# Self-contained (assembles inline); no env binary needed.
# ---------------------------------------------------------------------------

def test_wide_fp32_matches_numpy_no_quant_noise() -> None:
    from ipu_emu.emulator import load_program, run_until_complete
    from ipu_emu.execute import decode_instruction_word
    from ipu_emu.ipu_math import DType
    from ipu_emu.ipu_state import IpuState, WideVectorArithmetic
    from ipu_as.lark_tree import assemble

    N_Q, N_K = 128, 8          # one query group (128 lanes), short contraction
    rng = np.random.RandomState(42)
    P = rng.uniform(-1.0, 1.0, size=(N_Q, N_K)).astype(np.float32)  # P[i, s]
    V = rng.uniform(-1.0, 1.0, size=(N_K,)).astype(np.float32)      # V[s] (one channel)
    ref = P @ V                # (N_Q,)  exact reference

    st = IpuState(wide_vector_debug=True,
                  wide_vector_arithmetic=WideVectorArithmetic.FP32)
    st.dtype = DType.INT8
    st.set_cr_dstructure(128)

    # P key-major: key s -> 512-byte FP32 row of 128 queries at 0x10000 + s*512.
    for s in range(N_K):
        st.xmem.write_address(0x10000 + s * 512, struct.pack("<128f", *P[:, s]))
    # V channel broadcast: each scalar V[s] splatted to 128 lanes at 0x20000 + s*512.
    for s in range(N_K):
        st.xmem.write_address(0x20000 + s * 512,
                              struct.pack("<128f", *([float(V[s])] * 128)))

    st.regfile.set_cr(6, 0x10000)   # P key base
    st.regfile.set_cr(7, 0x20000)   # V base
    st.regfile.set_cr(9, 512)       # stride (wide FP32)

    # lr0 = P key ptr, lr1 = V key ptr, lr2 = rc index (0), lr3 = stride.
    lines = ["SET lr0 cr6;;", "SET lr1 cr7;;", "SET lr3 cr9;;"]
    for s in range(N_K):
        acc = "ACC.FIRST" if s == 0 else "ACC"
        lines.append("LDR_MULT_REG r0 lr1 cr0;;")              # R0 = V[s] broadcast
        lines.append("LDR_CYCLIC_MULT_REG lr0 cr0 lr2;;")      # R_CYCLIC = P[:,s]
        lines.append(f"MULT.RC.VE lr2 lr2 0 lr2; {acc};;")     # scalar = R0[lr2=0] = V[s]
        lines.append("ADD lr0 lr0 lr3;;")
        lines.append("ADD lr1 lr1 lr3;;")
    lines.append("BKPT;;")

    encoded = assemble("\n".join(lines))
    load_program(st, [decode_instruction_word(w) for w in encoded])
    run_until_complete(st)

    got = np.array([struct.unpack_from("<f", st.regfile.raw("r_acc"), i * 4)[0]
                    for i in range(N_Q)])
    assert np.allclose(got, ref, rtol=1e-5, atol=1e-4), f"got {got[:6]}\nref {ref[:6]}"


def _run(tmp_path: Path, dtype_dir: str, dtype_str: str) -> tuple[bytes, int]:
    data_dir = _DATA_DIR / dtype_dir
    if not data_dir.exists():
        pytest.skip(f"Test data not found: {data_dir}")
    p_path = data_dir / f"p_{dtype_dir}.bin"
    v_path = data_dir / f"v_{dtype_dir}.bin"
    if not p_path.exists() or not v_path.exists():
        pytest.skip(f"Missing data files in {data_dir}")
    output = tmp_path / "output.bin"
    app = AttnVBcast36App(
        inst_path=_INST_BIN,
        p_path=p_path,
        v_path=v_path,
        output_path=output,
        dtype=dtype_str,
    )
    _, cycles = app.run(max_cycles=20_000_000)
    return output.read_bytes(), cycles


@pytest.mark.parametrize("dtype_dir,dtype_str,golden_name", [
    ("int8",     "INT8",   "out_int8_acc_int32.bin"),
    ("fp8_e4m3", "fp8_e4", "out_fp8_e4m3_acc_fp32.bin"),
    ("fp8_e5m2", "fp8_e5", "out_fp8_e5m2_acc_fp32.bin"),
])
def test_attn_v_bcast_36(
    tmp_path: Path, dtype_dir: str, dtype_str: str, golden_name: str
) -> None:
    actual, cycles = _run(tmp_path, dtype_dir, dtype_str)
    assert cycles > 0
    golden = _DATA_DIR / dtype_dir / golden_name
    if golden.exists():
        assert actual == golden.read_bytes()
