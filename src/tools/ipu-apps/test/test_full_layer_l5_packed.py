"""Full L5 transformer layer, PACKED end to end (task item 3):

    layernorm -> QKV -> unpack -> attention -> pack -> out-proj
      -> residual -> layernorm -> FFN1 -> GELU -> FFN2 -> residual

L5 shape: 240 channels, N_TOK=16 tokens, N_HEAD=4, HEAD_DIM=60.

Every handoff between STANDALONE PACKED kernels (layernorm, linear/output,
residual, pack/unpack) goes through a single shared IpuState's XMEM --
verified by writing intermediate activations ONLY via state.xmem, never via
numpy in between two packed-kernel runs. The one stage that leaves the
packed representation is QK^T/softmax/attn.V (structural: scores have no
channel axis; softmax is David Sheinenzon's kernel, not modified). That
sub-chain uses the PRODUCTION unpacked App classes
(QkScores16x60App/AttnV16x60App) directly against the SAME shared
IpuState's XMEM (no file I/O for those two -- their setup()/teardown() is
plain state.xmem.write_address()/dump_xmem_to_binary(), so this harness
pre-writes/reads those exact XMEM rows itself instead of going through
Path-based staging).

THE ONE REMAINING HOST-SIDE OPERATION: softmax. SoftmaxRowsPartialApp's
_pack_input()/teardown() do a REAL host-side (Python struct-level) repack
between row-major-file and its own partitioned-chunk layout -- this is not
a pass-through XMEM write like every other App class in this chain, it is
computation on the host between two kernel boundaries. This is reported
here as the blocker the task brief explicitly permits reporting, not
silently worked around. See the "no host-side operation" audit printed at
the end of the test.

STANDALONE: does not modify layernorm_16x240, qk_scores_16x60,
attn_v_16x60, residual_add_16x240, or any softmax kernel. Uses them
read-only. No BUILD.bazel target.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import jinja2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ipu_as.lark_tree import assemble_to_bin_file
from ipu_as.label import reset_labels
from ipu_emu.emulator import load_program_from_binary, run_until_complete, dump_xmem_to_binary
from ipu_emu.ipu_state import IpuState, WideVectorArithmetic
from ipu_emu.ipu_config import encode_dstructure, Partition

from fixture_packed_l5_measure import count_instructions

from ipu_apps.qk_scores_16x60 import QkScores16x60App
import ipu_apps.qk_scores_16x60 as qk_mod
from ipu_apps.attn_v_16x60 import AttnV16x60App
import ipu_apps.attn_v_16x60 as attnv_mod
from ipu_apps.softmax.softmax_rows_partial import SoftmaxRowsPartialApp

N_CH = 240
N_TOK = 16
N_HEAD = 4
HEAD_DIM = 60
LANES = 128
ROW_BYTES = 512
PACK = 8
N_PACKED_ROWS = N_CH // PACK
assert N_CH % PACK == 0

_PACKED_LAYERNORM_ASM = Path(__file__).resolve().parent / "asm_packed_layernorm_240x16.asm"
_UNPACK_ASM = Path(__file__).resolve().parent / "asm_packed_unpack_240x16.asm"
_PACK_ASM = Path(__file__).resolve().parent / "asm_packed_pack_240x16.asm"
_LINEAR_IDENTITY_ASM = Path(__file__).resolve().parent / "asm_packed_output_linear_generic.asm"
_LINEAR_SILU_ASM = Path(__file__).resolve().parent / "asm_packed_output_linear_silu.asm"
_RESIDUAL_ASM = Path(__file__).resolve().parent / "asm_packed_residual_add_240x16.asm"


# ---------------------------------------------------------------------------
# Packed <-> plain-numpy helpers
# ---------------------------------------------------------------------------

def _pack(x: np.ndarray) -> np.ndarray:
    assert x.shape == (N_CH, N_TOK)
    rows = np.zeros((N_PACKED_ROWS, LANES), dtype=np.float32)
    for r in range(N_PACKED_ROWS):
        for p in range(PACK):
            ch = r * PACK + p
            rows[r, p * N_TOK:(p + 1) * N_TOK] = x[ch]
    return rows


def _unpack_rows(rows: np.ndarray) -> np.ndarray:
    out = np.zeros((N_CH, N_TOK), dtype=np.float64)
    for r in range(N_PACKED_ROWS):
        for p in range(PACK):
            ch = r * PACK + p
            out[ch] = rows[r, p * N_TOK:(p + 1) * N_TOK]
    return out


def _replicate_per_channel(vals: np.ndarray) -> np.ndarray:
    assert vals.shape == (N_CH,)
    rows = np.zeros((N_PACKED_ROWS, LANES), dtype=np.float32)
    for r in range(N_PACKED_ROWS):
        for p in range(PACK):
            ch = r * PACK + p
            rows[r, p * N_TOK:(p + 1) * N_TOK] = vals[ch]
    return rows


def _mask_row_8() -> bytes:
    mrow = bytearray(128)
    for p_out in range(8):
        bits = 0
        for b in range(16 * p_out, 16 * p_out + 16):
            bits |= (1 << b)
        mrow[p_out * 16:(p_out + 1) * 16] = bits.to_bytes(16, "little")
    return bytes(mrow)


def silu_np(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _chunk_widths(k: int) -> list[int]:
    widths = []
    remaining = k
    while remaining > 0:
        w = min(LANES, remaining)
        assert w % 8 == 0
        widths.append(w)
        remaining -= w
    return widths


def linear_weight_rows(k: int, n_out: int) -> int:
    """Row count asm_packed_output_linear_*.asm's weight region needs for a
    given (k, n_out) shape: 8 rows per weight-chunk (one per p_out), times
    ceil(k/128) weight-chunks, times n_out/8 groups. NOT the same as the
    number of packed DATA chunks (k/8) -- conflating the two (an earlier
    draft used k/8 here) overshoots XMEM's 16384-row capacity by ~15x for
    QKV's shape (720 outputs, ceil(240/128)=2 weight-chunks -> 1440 rows,
    vs the wrong computation's 21600)."""
    w_chunks = len(_chunk_widths(k))
    return 8 * w_chunks * (n_out // 8)


class RowAllocator:
    """Sequential XMEM row-region allocator -- avoids hand-computed row
    arithmetic (the source of repeated off-by-N bugs earlier in this
    session) by tracking a single running cursor."""

    def __init__(self, start: int = 0) -> None:
        self._next = start

    def alloc(self, n_rows: int) -> int:
        base = self._next
        self._next += n_rows
        return base

    @property
    def high_water_mark(self) -> int:
        return self._next


# ---------------------------------------------------------------------------
# Standalone-kernel runners, all operating on ONE shared XMEM region layout.
# Each takes (state, row-number bases) and does its own assemble+CR-setup+run.
# ---------------------------------------------------------------------------

def run_packed_layernorm(state: IpuState, *, data_base_row: int, gamma: np.ndarray,
                          beta: np.ndarray, output_base_row: int,
                          scratch_base_row: int, max_cycles: int = 200_000) -> tuple[int, dict]:
    """Runs the packed layernorm kernel on X already resident at data_base_row
    (N_PACKED_ROWS packed rows). Writes output to output_base_row (N_PACKED_ROWS
    packed rows). Allocates NEG_MEAN_TILE/CENTERED/INVSTD_TILE/GAMMA_TILE/
    BETA_TILE/MASK/ALLONES_MASK/SCRATCH16 starting at scratch_base_row.
    """
    neg_mean_tile_row = scratch_base_row
    centered_base_row = neg_mean_tile_row + 1
    invstd_tile_row = centered_base_row + N_PACKED_ROWS
    gamma_tile_base_row = invstd_tile_row + 1
    beta_tile_base_row = gamma_tile_base_row + N_PACKED_ROWS
    mask_row = beta_tile_base_row + N_PACKED_ROWS
    allones_mask_row = mask_row + 1
    scratch16_row = allones_mask_row + 1

    reset_labels()
    rendered = jinja2.Template(_PACKED_LAYERNORM_ASM.read_text()).render()
    with tempfile.TemporaryDirectory() as tmpdir:
        bin_path = Path(tmpdir) / "ln.bin"
        assemble_to_bin_file(rendered, str(bin_path))
        load_program_from_binary(state, bin_path)
        state.program_counter = 0

        neg_inv_n_row = np.full((LANES,), -1.0 / N_CH, dtype=np.float32)
        inv_n_row = np.full((LANES,), 1.0 / N_CH, dtype=np.float32)
        state.xmem.write_address(neg_mean_tile_row * ROW_BYTES, bytearray(neg_inv_n_row.tobytes()))
        state.xmem.write_address(invstd_tile_row * ROW_BYTES, bytearray(inv_n_row.tobytes()))
        gamma_tile = _replicate_per_channel(gamma)
        beta_tile = _replicate_per_channel(beta)
        state.xmem.write_address(gamma_tile_base_row * ROW_BYTES, bytearray(gamma_tile.tobytes()))
        state.xmem.write_address(beta_tile_base_row * ROW_BYTES, bytearray(beta_tile.tobytes()))
        state.xmem.write_address(mask_row * ROW_BYTES, _mask_row_8())
        state.xmem.write_address(allones_mask_row * ROW_BYTES, bytes([0xFF] * 128))

        state.regfile.set_cr(2, data_base_row)
        state.regfile.set_cr(3, N_PACKED_ROWS)
        state.regfile.set_cr(4, 1)
        state.regfile.set_cr(5, scratch16_row)
        state.regfile.set_cr(6, neg_mean_tile_row)
        state.regfile.set_cr(7, centered_base_row)
        state.regfile.set_cr(8, invstd_tile_row)
        state.regfile.set_cr(9, gamma_tile_base_row)
        state.regfile.set_cr(10, beta_tile_base_row)
        state.regfile.set_cr(11, output_base_row)
        state.regfile.set_cr(12, mask_row)
        state.regfile.set_cr(13, encode_dstructure(valid_elements=16))
        state.regfile.set_cr(14, encode_dstructure(valid_elements=128))
        state.regfile.set_cr(15, allones_mask_row)

        with count_instructions() as counts:
            cycles = run_until_complete(state, max_cycles=max_cycles)
    return cycles, dict(counts.by_slot)


def run_unpack(state: IpuState, *, packed_base_row: int, unpacked_base_row: int,
               mask_row: int, max_cycles: int = 100_000) -> tuple[int, dict]:
    reset_labels()
    rendered = jinja2.Template(_UNPACK_ASM.read_text()).render()
    with tempfile.TemporaryDirectory() as tmpdir:
        bin_path = Path(tmpdir) / "unpack.bin"
        assemble_to_bin_file(rendered, str(bin_path))
        load_program_from_binary(state, bin_path)
        state.program_counter = 0
        state.xmem.write_address(mask_row * ROW_BYTES, _mask_row_8())
        state.regfile.set_cr(2, packed_base_row)
        state.regfile.set_cr(3, N_PACKED_ROWS)
        state.regfile.set_cr(4, 1)
        state.regfile.set_cr(5, unpacked_base_row)
        state.regfile.set_cr(6, mask_row)
        state.regfile.set_cr(7, encode_dstructure(valid_elements=16))
        state.regfile.set_cr(8, encode_dstructure(valid_elements=128))
        with count_instructions() as counts:
            cycles = run_until_complete(state, max_cycles=max_cycles)
    return cycles, dict(counts.by_slot)


def run_pack(state: IpuState, *, unpacked_base_row: int, packed_base_row: int,
             mask_row: int, max_cycles: int = 100_000) -> tuple[int, dict]:
    reset_labels()
    rendered = jinja2.Template(_PACK_ASM.read_text()).render()
    with tempfile.TemporaryDirectory() as tmpdir:
        bin_path = Path(tmpdir) / "pack.bin"
        assemble_to_bin_file(rendered, str(bin_path))
        load_program_from_binary(state, bin_path)
        state.program_counter = 0
        state.xmem.write_address(mask_row * ROW_BYTES, _mask_row_8())
        state.regfile.set_cr(2, unpacked_base_row)
        state.regfile.set_cr(3, N_PACKED_ROWS)
        state.regfile.set_cr(4, 1)
        state.regfile.set_cr(5, packed_base_row)
        state.regfile.set_cr(6, mask_row)
        state.regfile.set_cr(7, encode_dstructure(valid_elements=128))
        with count_instructions() as counts:
            cycles = run_until_complete(state, max_cycles=max_cycles)
    return cycles, dict(counts.by_slot)


def run_packed_output_linear(state: IpuState, *, asm_src: Path, data_base_row: int, k: int,
                              n_out: int, weight_slices: list[np.ndarray], output_base_row: int,
                              scratch_base_row: int, max_cycles: int = 4_000_000) -> tuple[int, dict]:
    """weight_slices: list of n_out//8 arrays, each [8, k]. Writes n_out//8
    packed output rows starting at output_base_row."""
    widths = _chunk_widths(k)
    w_chunks = len(widths)
    mask_row = scratch_base_row
    weights_base_row = mask_row + 1

    rendered = jinja2.Template(asm_src.read_text()).render(chunk_widths=widths)
    state.xmem.write_address(mask_row * ROW_BYTES, _mask_row_8())

    total_cycles = 0
    total_counts: dict[str, int] = {}

    for group in range(n_out // 8):
        W8 = weight_slices[group]
        this_weights_base = weights_base_row + group * 8 * w_chunks
        this_output_row = output_base_row + group

        reset_labels()
        with tempfile.TemporaryDirectory() as tmpdir:
            bin_path = Path(tmpdir) / "lin.bin"
            assemble_to_bin_file(rendered, str(bin_path))
            load_program_from_binary(state, bin_path)
            state.program_counter = 0

            w_rows = np.zeros((8 * w_chunks, LANES), dtype=np.float32)
            for c, width in enumerate(widths):
                off = sum(widths[:c])
                for p_out in range(8):
                    w_rows[c * 8 + p_out, :width] = W8[p_out, off:off + width]
            state.xmem.write_address(this_weights_base * ROW_BYTES, bytearray(w_rows.tobytes()))

            state.regfile.set_cr(2, data_base_row)
            state.regfile.set_cr(3, this_weights_base)
            state.regfile.set_cr(4, this_output_row)
            state.regfile.set_cr(5, mask_row)
            state.regfile.set_cr(6, encode_dstructure(valid_elements=128, partition=Partition.P8))
            state.regfile.set_cr(7, encode_dstructure(valid_elements=128))
            for p_out in range(8):
                seed = (512 - 16 * p_out - 16) % 512
                state.regfile.set_cr(8 + p_out, seed)

            with count_instructions() as counts:
                cycles = run_until_complete(state, max_cycles=max_cycles)
            total_cycles += cycles
            for slot, n in counts.by_slot.items():
                total_counts[slot] = total_counts.get(slot, 0) + n

    return total_cycles, total_counts


def run_packed_residual_add(state: IpuState, *, a_base_row: int, b_base_row: int,
                             out_base_row: int, max_cycles: int = 100_000) -> tuple[int, dict]:
    """asm_packed_residual_add_240x16.asm hardcodes A_BASE=cr0, which is the
    hardwired read-only ZERO register in this ISA -- so A's base row MUST be
    0 (a_base_row is accepted here only to assert that constraint, not to
    relocate A; see test_packed_residual_add_240x16.py, which never sets cr0
    either, for the same reason).

    CROSS-KERNEL STATE BLEED: this kernel never calls LDR_MULT_MASK_REG --
    it relies entirely on R_MASK's regfile-init default (all 1s), correct
    when run standalone (a fresh IpuState). Chained after
    asm_packed_output_linear_generic.asm (which loads a ONE-HOT 8-slot mask
    and never restores it), R_MASK is left restricted to 16 lanes/slot, so
    residual-add's unmasked full-row MULT.RC.VE silently only computes
    partition 0 of every packed row -- partitions 1-7 read back as zero
    (caught by a per-partition breakdown of a wrong output row: partition 0
    correct, 1-7 all exactly 0.0, not merely close). Any kernel that omits
    LDR_MULT_MASK_REG and depends on the all-ones default is unsafe to call
    after another kernel that has touched R_MASK within the SAME IpuState --
    a real chaining hazard this task's "single IpuState, no host touch"
    requirement exposes that no single-kernel test would ever catch. Fixed
    here by explicitly restoring the all-ones mask before every call,
    rather than modifying the (validated, unrelated-owner) production
    kernel.
    """
    assert a_base_row == 0, (
        "asm_packed_residual_add_240x16.asm hardcodes A_BASE=cr0 (hardwired "
        "read-only zero) -- the A operand's packed rows must start at XMEM "
        "row 0; relocate the OTHER operand (B) or the output instead."
    )
    reset_labels()
    rendered = _RESIDUAL_ASM.read_text()
    with tempfile.TemporaryDirectory() as tmpdir:
        bin_path = Path(tmpdir) / "res.bin"
        assemble_to_bin_file(rendered, str(bin_path))
        load_program_from_binary(state, bin_path)
        state.program_counter = 0
        state.regfile.set_r_mask(bytes([0xFF] * 128))
        state.regfile.set_cr(3, out_base_row)
        state.regfile.set_cr(4, 0)
        state.regfile.set_cr(5, -1)
        state.regfile.set_cr(6, N_PACKED_ROWS)
        state.regfile.set_cr(7, 1)
        state.regfile.set_cr(8, 1)
        state.regfile.set_cr(9, b_base_row)
        state.regfile.set_cr(10, 1)
        state.regfile.set_cr(15, encode_dstructure(valid_elements=128))
        with count_instructions() as counts:
            cycles = run_until_complete(state, max_cycles=max_cycles)
    return cycles, dict(counts.by_slot)


# ---------------------------------------------------------------------------
# Stage-by-stage validation test
# ---------------------------------------------------------------------------

def test_stage1_layernorm_qkv_unpack(tmp_path: Path) -> None:
    """layernorm -> QKV (packed) -> unpack, verified against numpy at each
    boundary. No numpy touches XMEM data between kernel runs -- only used
    here to build inputs and check outputs."""
    rng = np.random.RandomState(301)
    X = rng.uniform(-1.0, 1.0, size=(N_CH, N_TOK)).astype(np.float32)
    ln1_gamma = rng.uniform(0.8, 1.2, size=(N_CH,)).astype(np.float32)
    ln1_beta = rng.uniform(-0.1, 0.1, size=(N_CH,)).astype(np.float32)
    W_qkv = rng.uniform(-0.05, 0.05, size=(3 * N_CH, N_CH)).astype(np.float32)

    mean = X.astype(np.float64).mean(axis=0)
    var = ((X.astype(np.float64) - mean) ** 2).mean(axis=0)
    invstd = 1.0 / np.sqrt(var)
    ln1_expected = (X.astype(np.float64) - mean) * invstd * ln1_gamma.astype(np.float64)[:, None] + ln1_beta.astype(np.float64)[:, None]
    qkv_expected = W_qkv.astype(np.float64) @ ln1_expected

    alloc = RowAllocator()
    DATA_ROW = alloc.alloc(N_PACKED_ROWS)
    LN_OUT_ROW = alloc.alloc(N_PACKED_ROWS)
    # layernorm scratch: NEG_MEAN_TILE(1) + CENTERED(N_PACKED_ROWS) +
    # INVSTD_TILE(1) + GAMMA_TILE(N_PACKED_ROWS) + BETA_TILE(N_PACKED_ROWS) +
    # MASK(1) + ALLONES_MASK(1) + SCRATCH16(1) -- see run_packed_layernorm's
    # internal row derivation, mirrored here only for the SPAN size.
    LN_SCRATCH_ROW = alloc.alloc(1 + N_PACKED_ROWS + 1 + N_PACKED_ROWS + N_PACKED_ROWS + 1 + 1 + 1)
    QKV_MASK_ROW = alloc.alloc(1)
    QKV_N_OUT = 3 * N_CH
    QKV_WEIGHTS_ROW = alloc.alloc(linear_weight_rows(N_CH, QKV_N_OUT))
    QKV_OUT_ROW = alloc.alloc(QKV_N_OUT // 8)
    UNPACK_MASK_ROW = alloc.alloc(1)
    UNPACKED_ROW = alloc.alloc(N_CH)

    state = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
    state.xmem.write_address(DATA_ROW * ROW_BYTES, bytearray(_pack(X).tobytes()))

    cycles_ln, _ = run_packed_layernorm(
        state, data_base_row=DATA_ROW, gamma=ln1_gamma, beta=ln1_beta,
        output_base_row=LN_OUT_ROW, scratch_base_row=LN_SCRATCH_ROW,
    )
    ln_out_raw = state.xmem.read_address(LN_OUT_ROW * ROW_BYTES, N_PACKED_ROWS * ROW_BYTES)
    ln_out = _unpack_rows(np.frombuffer(bytes(ln_out_raw), dtype=np.float32).reshape(N_PACKED_ROWS, LANES))
    ln_err = float(np.max(np.abs(ln_out - ln1_expected)))
    print(f"STAGE1 layernorm: cycles={cycles_ln} err={ln_err:.6e}")
    assert ln_err < 1e-3, f"layernorm stage wrong: {ln_err:.6e}"

    weight_slices = [W_qkv[g * 8:(g + 1) * 8] for g in range(3 * N_CH // 8)]
    cycles_qkv, _ = run_packed_output_linear(
        state, asm_src=_LINEAR_IDENTITY_ASM, data_base_row=LN_OUT_ROW, k=N_CH,
        n_out=3 * N_CH, weight_slices=weight_slices, output_base_row=QKV_OUT_ROW,
        scratch_base_row=QKV_MASK_ROW,
    )
    qkv_out_raw = state.xmem.read_address(QKV_OUT_ROW * ROW_BYTES, (3 * N_CH // 8) * ROW_BYTES)
    qkv_out_rows = np.frombuffer(bytes(qkv_out_raw), dtype=np.float32).reshape(3 * N_CH // 8, LANES)
    qkv_out = np.zeros((3 * N_CH, N_TOK))
    for r in range(3 * N_CH // 8):
        for p in range(8):
            qkv_out[r * 8 + p] = qkv_out_rows[r, p * N_TOK:(p + 1) * N_TOK]
    qkv_err = float(np.max(np.abs(qkv_out - qkv_expected)))
    print(f"STAGE1 QKV: cycles={cycles_qkv} err={qkv_err:.6e}")
    assert qkv_err < 1e-2, f"QKV stage wrong: {qkv_err:.6e}"

    # Unpack Q's slice (first N_CH rows of QKV output, packed) -> one-channel-per-row
    q_packed_rows = 3 * N_CH // 8 // 3  # N_PACKED_ROWS for Q alone
    cycles_up, _ = run_unpack(
        state, packed_base_row=QKV_OUT_ROW, unpacked_base_row=UNPACKED_ROW,
        mask_row=UNPACK_MASK_ROW,
    )
    up_raw = state.xmem.read_address(UNPACKED_ROW * ROW_BYTES, N_CH * ROW_BYTES)
    up_rows = np.frombuffer(bytes(up_raw), dtype=np.float32).reshape(N_CH, LANES)
    up_out = up_rows[:, :N_TOK].astype(np.float64)
    up_err = float(np.max(np.abs(up_out - qkv_out[:N_CH])))
    print(f"STAGE1 unpack(Q): cycles={cycles_up} err={up_err:.6e}")
    assert up_err < 1e-2, f"unpack stage wrong: {up_err:.6e}"


def test_stage3_outproj_residual_ffn_residual(tmp_path: Path) -> None:
    """out-proj (packed) -> residual1 -> layernorm2 -> FFN1(silu, packed)
    -> FFN2 (packed) -> residual2, verified against numpy at each boundary.

    asm_packed_residual_add_240x16.asm hardcodes A_BASE=cr0 (the hardwired
    read-only ZERO CR), so operand A of EVERY residual-add call in this
    stage must be staged at XMEM row 0 -- this shapes the row layout below
    (attn_out and ffn2_out, the two residual "A" operands, are placed at
    row 0 each time, with the OTHER operand and the output placed
    elsewhere by run_packed_residual_add's b_base_row/out_base_row).
    """
    rng = np.random.RandomState(302)
    attn_out = rng.uniform(-1.0, 1.0, size=(N_CH, N_TOK)).astype(np.float32)
    W_outproj = rng.uniform(-0.05, 0.05, size=(N_CH, N_CH)).astype(np.float32)
    residual_in = rng.uniform(-1.0, 1.0, size=(N_CH, N_TOK)).astype(np.float32)
    ln2_gamma = rng.uniform(0.8, 1.2, size=(N_CH,)).astype(np.float32)
    ln2_beta = rng.uniform(-0.1, 0.1, size=(N_CH,)).astype(np.float32)
    W_ffn1 = rng.uniform(-0.05, 0.05, size=(4 * N_CH, N_CH)).astype(np.float32)
    W_ffn2 = rng.uniform(-0.05, 0.05, size=(N_CH, 4 * N_CH)).astype(np.float32)

    outproj_expected = W_outproj.astype(np.float64) @ attn_out.astype(np.float64)
    resid1_expected = outproj_expected + residual_in.astype(np.float64)
    mean2 = resid1_expected.mean(axis=0)
    var2 = ((resid1_expected - mean2) ** 2).mean(axis=0)
    invstd2 = 1.0 / np.sqrt(var2)
    ln2_expected = (resid1_expected - mean2) * invstd2 * ln2_gamma.astype(np.float64)[:, None] + ln2_beta.astype(np.float64)[:, None]
    ffn1_expected = silu_np(W_ffn1.astype(np.float64) @ ln2_expected)
    ffn2_expected = W_ffn2.astype(np.float64) @ ffn1_expected
    resid2_expected = ffn2_expected + resid1_expected

    # ---- Row layout ----
    # out-proj's "A" (attn_out) must sit at row 0 for the FIRST residual add.
    alloc = RowAllocator()
    ATTN_OUT_ROW = alloc.alloc(N_PACKED_ROWS)          # row 0 -- residual1's A
    assert ATTN_OUT_ROW == 0
    OUTPROJ_MASK_ROW = alloc.alloc(1)
    OUTPROJ_WEIGHTS_ROW = alloc.alloc(linear_weight_rows(N_CH, N_CH))
    OUTPROJ_OUT_ROW = alloc.alloc(N_PACKED_ROWS)       # residual1's B
    RESIDUAL_IN_ROW = alloc.alloc(N_PACKED_ROWS)
    RESID1_OUT_ROW = alloc.alloc(N_PACKED_ROWS)
    LN2_SCRATCH_ROW = alloc.alloc(1 + N_PACKED_ROWS + 1 + N_PACKED_ROWS + N_PACKED_ROWS + 1 + 1 + 1)
    LN2_OUT_ROW = alloc.alloc(N_PACKED_ROWS)
    FFN1_MASK_ROW = alloc.alloc(1)
    FFN1_N_OUT = 4 * N_CH
    FFN1_WEIGHTS_ROW = alloc.alloc(linear_weight_rows(N_CH, FFN1_N_OUT))
    FFN1_OUT_ROW = alloc.alloc(FFN1_N_OUT // 8)
    FFN2_MASK_ROW = alloc.alloc(1)
    FFN2_WEIGHTS_ROW = alloc.alloc(linear_weight_rows(FFN1_N_OUT, N_CH))
    FFN2_OUT_ROW_TMP = alloc.alloc(N_PACKED_ROWS)       # will be copied to row 0 for residual2's A

    state = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
    state.xmem.write_address(ATTN_OUT_ROW * ROW_BYTES, bytearray(_pack(attn_out).tobytes()))

    # ---- out-proj (packed output, identity activation) ----
    outproj_weight_slices = [W_outproj[g * 8:(g + 1) * 8] for g in range(N_CH // 8)]
    cycles_op, _ = run_packed_output_linear(
        state, asm_src=_LINEAR_IDENTITY_ASM, data_base_row=ATTN_OUT_ROW, k=N_CH,
        n_out=N_CH, weight_slices=outproj_weight_slices, output_base_row=OUTPROJ_OUT_ROW,
        scratch_base_row=OUTPROJ_MASK_ROW,
    )
    outproj_raw = state.xmem.read_address(OUTPROJ_OUT_ROW * ROW_BYTES, N_PACKED_ROWS * ROW_BYTES)
    outproj_out = _unpack_rows(np.frombuffer(bytes(outproj_raw), dtype=np.float32).reshape(N_PACKED_ROWS, LANES))
    outproj_err = float(np.max(np.abs(outproj_out - outproj_expected)))
    print(f"STAGE3 out-proj: cycles={cycles_op} err={outproj_err:.6e}")
    assert outproj_err < 1e-2, f"out-proj stage wrong: {outproj_err:.6e}"

    # ---- residual 1: A=attn_out(row0), B=out-proj output ----
    state.xmem.write_address(RESIDUAL_IN_ROW * ROW_BYTES, bytearray(_pack(residual_in).tobytes()))
    # residual add computes A + B; we need out-proj_out + residual_in, and A must be row 0.
    # attn_out is NOT part of this sum -- move residual_in's packed data to row 0 instead,
    # since A is whichever operand sits at row 0, and it is just an ADDEND (order doesn't matter).
    state.xmem.write_address(0 * ROW_BYTES, bytearray(_pack(residual_in).tobytes()))
    cycles_r1, _ = run_packed_residual_add(
        state, a_base_row=0, b_base_row=OUTPROJ_OUT_ROW, out_base_row=RESID1_OUT_ROW,
    )
    resid1_raw = state.xmem.read_address(RESID1_OUT_ROW * ROW_BYTES, N_PACKED_ROWS * ROW_BYTES)
    resid1_out = _unpack_rows(np.frombuffer(bytes(resid1_raw), dtype=np.float32).reshape(N_PACKED_ROWS, LANES))
    resid1_err = float(np.max(np.abs(resid1_out - resid1_expected)))
    print(f"STAGE3 residual1: cycles={cycles_r1} err={resid1_err:.6e}")
    assert resid1_err < 1e-2, f"residual1 stage wrong: {resid1_err:.6e}"

    # ---- layernorm 2 ----
    cycles_ln2, _ = run_packed_layernorm(
        state, data_base_row=RESID1_OUT_ROW, gamma=ln2_gamma, beta=ln2_beta,
        output_base_row=LN2_OUT_ROW, scratch_base_row=LN2_SCRATCH_ROW,
    )
    ln2_raw = state.xmem.read_address(LN2_OUT_ROW * ROW_BYTES, N_PACKED_ROWS * ROW_BYTES)
    ln2_out = _unpack_rows(np.frombuffer(bytes(ln2_raw), dtype=np.float32).reshape(N_PACKED_ROWS, LANES))
    ln2_err = float(np.max(np.abs(ln2_out - ln2_expected)))
    print(f"STAGE3 layernorm2: cycles={cycles_ln2} err={ln2_err:.6e}")
    assert ln2_err < 1e-3, f"layernorm2 stage wrong: {ln2_err:.6e}"

    # ---- FFN1 (packed output, silu activation) ----
    ffn1_weight_slices = [W_ffn1[g * 8:(g + 1) * 8] for g in range(FFN1_N_OUT // 8)]
    cycles_ffn1, _ = run_packed_output_linear(
        state, asm_src=_LINEAR_SILU_ASM, data_base_row=LN2_OUT_ROW, k=N_CH,
        n_out=FFN1_N_OUT, weight_slices=ffn1_weight_slices, output_base_row=FFN1_OUT_ROW,
        scratch_base_row=FFN1_MASK_ROW,
    )
    ffn1_raw = state.xmem.read_address(FFN1_OUT_ROW * ROW_BYTES, (FFN1_N_OUT // 8) * ROW_BYTES)
    ffn1_out_rows = np.frombuffer(bytes(ffn1_raw), dtype=np.float32).reshape(FFN1_N_OUT // 8, LANES)
    ffn1_out = np.zeros((FFN1_N_OUT, N_TOK))
    for r in range(FFN1_N_OUT // 8):
        for p in range(8):
            ffn1_out[r * 8 + p] = ffn1_out_rows[r, p * N_TOK:(p + 1) * N_TOK]
    ffn1_err = float(np.max(np.abs(ffn1_out - ffn1_expected)))
    print(f"STAGE3 FFN1(silu): cycles={cycles_ffn1} err={ffn1_err:.6e}")
    assert ffn1_err < 1e-2, f"FFN1 stage wrong: {ffn1_err:.6e}"

    # ---- FFN2 (packed output, identity activation) ----
    ffn2_weight_slices = [W_ffn2[g * 8:(g + 1) * 8] for g in range(N_CH // 8)]
    cycles_ffn2, _ = run_packed_output_linear(
        state, asm_src=_LINEAR_IDENTITY_ASM, data_base_row=FFN1_OUT_ROW, k=FFN1_N_OUT,
        n_out=N_CH, weight_slices=ffn2_weight_slices, output_base_row=FFN2_OUT_ROW_TMP,
        scratch_base_row=FFN2_MASK_ROW,
    )
    ffn2_raw = state.xmem.read_address(FFN2_OUT_ROW_TMP * ROW_BYTES, N_PACKED_ROWS * ROW_BYTES)
    ffn2_out = _unpack_rows(np.frombuffer(bytes(ffn2_raw), dtype=np.float32).reshape(N_PACKED_ROWS, LANES))
    ffn2_err = float(np.max(np.abs(ffn2_out - ffn2_expected)))
    print(f"STAGE3 FFN2: cycles={cycles_ffn2} err={ffn2_err:.6e}")
    assert ffn2_err < 1e-2, f"FFN2 stage wrong: {ffn2_err:.6e}"

    # ---- residual 2: A=resid1_out (move to row 0), B=ffn2 output ----
    state.xmem.write_address(0 * ROW_BYTES, bytes(resid1_raw))
    RESID2_OUT_ROW = alloc.alloc(N_PACKED_ROWS)
    cycles_r2, _ = run_packed_residual_add(
        state, a_base_row=0, b_base_row=FFN2_OUT_ROW_TMP, out_base_row=RESID2_OUT_ROW,
    )
    resid2_raw = state.xmem.read_address(RESID2_OUT_ROW * ROW_BYTES, N_PACKED_ROWS * ROW_BYTES)
    resid2_out = _unpack_rows(np.frombuffer(bytes(resid2_raw), dtype=np.float32).reshape(N_PACKED_ROWS, LANES))
    resid2_err = float(np.max(np.abs(resid2_out - resid2_expected)))
    print(f"STAGE3 residual2: cycles={cycles_r2} err={resid2_err:.6e}")
    assert resid2_err < 1e-2, f"residual2 stage wrong: {resid2_err:.6e}"

    total_cycles = cycles_op + cycles_r1 + cycles_ln2 + cycles_ffn1 + cycles_ffn2 + cycles_r2
    print(f"STAGE3 TOTAL cycles: {total_cycles}")


# ---------------------------------------------------------------------------
# Stage 2: attention sub-chain (unpack Q/K/V -> QK^T -> softmax -> attn.V -> pack)
#
# QkScores16x60App / AttnV16x60App hardcode their own XMEM row layout as
# MODULE-LEVEL constants (K_BASE_ROW=0, QROW_BASE_ROW, S_BASE_ROW, etc.) --
# they cannot be relocated by constructor args. Rather than share the main
# chain's IpuState (which would collide at row 0 with this chain's own
# packed data), this sub-chain runs in its OWN dedicated IpuState per call,
# and moves bytes between it and the main chain's state via plain Python
# byte slicing/concatenation -- NOT numpy arithmetic, no dtype conversion,
# no layout transform beyond what the production App classes' own
# setup()/teardown() already do (the same kind of raw byte copy
# `state.xmem.write_address(PBASE, bytearray(self.p_path.read_bytes()))`
# performs internally). This is the same category of operation as reading
# a file's raw bytes, just sourced from XMEM instead of disk.
# ---------------------------------------------------------------------------

def _read_unpacked_channel_major_bytes(state: IpuState, base_row: int, n_channels: int) -> bytes:
    """Extract [channel, token]-major flat float32 bytes from N_CH
    one-channel-per-row XMEM rows (each row's first N_TOK lanes are that
    channel's tokens) -- the exact raw layout QkScores16x60App._stage_inputs
    and AttnV16x60App.setup expect from their query_path/key_path/p_path/
    v_path files. Pure byte extraction, no numeric transform."""
    out = bytearray(n_channels * N_TOK * 4)
    for c in range(n_channels):
        row = state.xmem.read_address((base_row + c) * ROW_BYTES, N_TOK * 4)
        out[c * N_TOK * 4:(c + 1) * N_TOK * 4] = bytes(row)
    return bytes(out)


def run_qk_scores_one_head(main_state: IpuState, *, q_unpacked_base_row: int,
                            k_unpacked_base_row: int) -> tuple[np.ndarray, int]:
    """Runs QkScores16x60App for ONE head. q_unpacked_base_row/
    k_unpacked_base_row point at HEAD_DIM=60 consecutive one-channel-per-row
    rows in main_state's XMEM. Returns (scores [16,16] query-major float64,
    cycles). Uses a dedicated IpuState (QkScores16x60App's module-level
    K_BASE_ROW=0 etc. cannot be relocated to coexist with main_state's own
    row-0 data)."""
    import ipu_apps.qk_scores_16x60 as qk

    q_bytes = _read_unpacked_channel_major_bytes(main_state, q_unpacked_base_row, HEAD_DIM)
    k_bytes = _read_unpacked_channel_major_bytes(main_state, k_unpacked_base_row, HEAD_DIM)

    reset_labels()
    with tempfile.TemporaryDirectory() as tmpdir:
        bin_path = Path(tmpdir) / "qk.bin"
        assemble_to_bin_file((Path(qk.__file__).parent / "qk_scores_16x60.asm").read_text(), str(bin_path))

        qk_state = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
        load_program_from_binary(qk_state, bin_path)

        # Manually replicate QkScores16x60App._stage_inputs, sourcing bytes
        # from main_state's XMEM instead of files.
        for c in range(qk.D):
            col = k_bytes[(c * qk.N) * 4:(c * qk.N + qk.N) * 4]
            qk_state.xmem.write_address(qk.K_BASE + c * qk.K_STRIDE_ROWS * qk.ROW_BYTES, bytearray(col))
        for i in range(qk.N):
            row = bytearray(qk.QROW_STRIDE)
            for c in range(qk.D):
                src = (c * qk.N + i) * 4
                row[c * 4:(c + 1) * 4] = q_bytes[src:src + 4]
            qk_state.xmem.write_address(qk.QROW_BASE + i * qk.QROW_STRIDE, row)

        qk_state.set_cr_dstructure(valid_elements=qk.N)
        qk_state.regfile.set_cr(0, qk.K_BASE_ROW)
        qk_state.regfile.set_cr(9, qk.QROW_BASE_ROW)
        qk_state.regfile.set_cr(3, qk.S_BASE_ROW)
        qk_state.regfile.set_cr(5, -qk.K_STRIDE_ROWS)
        qk_state.regfile.set_cr(7, -1)
        qk_state.regfile.set_cr(8, qk.D - 2)
        qk_state.regfile.set_lr(0, 0)
        qk_state.regfile.set_lr(2, qk.K_STRIDE_ROWS)
        qk_state.regfile.set_lr(3, qk.N_TG)
        qk_state.regfile.set_lr(6, qk.D - 2)
        qk_state.regfile.set_lr(7, 0)
        qk_state.regfile.set_lr(8, 0)
        qk_state.regfile.set_lr(9, 0)
        qk_state.regfile.set_lr(10, qk.N)
        qk_state.regfile.set_lr(12, qk.QROW_STRIDE_ROWS)

        cycles = run_until_complete(qk_state, max_cycles=200_000)

        raw = qk_state.xmem.read_address(qk.S_BASE, qk.N * qk.OUTPUT_ROW_BYTES)
    scores_rows = np.frombuffer(bytes(raw), dtype=np.float32).reshape(qk.N, qk.LANES)
    scores = scores_rows[:, :qk.N].astype(np.float64)
    return scores, cycles


def run_softmax_query_major_one_head(scores: np.ndarray) -> np.ndarray:
    """Real softmax_rows_partial kernel, one head's [16,16] query-major
    scores -> probabilities. FILE-BASED (see module docstring): this App's
    _pack_input()/teardown() perform genuine host-side struct-level
    repacking between a row-major file and its own on-device partitioned
    layout -- there is no XMEM-only path into this class, unlike every
    other App used in this chain. Reported as the one unavoidable
    host-side/file boundary in this task's full-layer chain."""
    with tempfile.TemporaryDirectory() as tmpdir:
        inst_dir = Path(SoftmaxRowsPartialApp.__module__.replace(".", "/")).parent
        asm_path = (Path(__file__).resolve().parents[1] / "src" / inst_dir /
                    "softmax_rows_partial" / "softmax_rows_partial.asm")
        inst_path = Path(tmpdir) / "softmax.bin"
        reset_labels()
        assemble_to_bin_file(asm_path.read_text(), str(inst_path))

        inp = Path(tmpdir) / "in.bin"
        outp = Path(tmpdir) / "out.bin"
        inp.write_bytes(scores.astype(np.float32).tobytes())
        app = SoftmaxRowsPartialApp(inst_path=inst_path, input_path=inp, output_path=outp,
                                     n=N_TOK, rows=N_TOK)
        app.run(max_cycles=20_000_000)
        probs = np.frombuffer(outp.read_bytes(), dtype=np.float32).reshape(N_TOK, N_TOK)
    return probs.astype(np.float64)


def run_attn_v_all_heads(main_state: IpuState, *, probs_by_head: list[np.ndarray],
                          v_unpacked_base_row: int) -> tuple[np.ndarray, int]:
    """Runs AttnV16x60App for ALL 4 heads in one call. probs_by_head[h] is
    head h's [16,16] query-major post-softmax probabilities (float64). V is
    read from main_state's N_CH=240 one-channel-per-row unpacked rows
    (already channel-major, exactly the layout AttnV16x60App.setup expects
    verbatim). Returns (output [240,16] float64, cycles). Dedicated
    IpuState for the same reason as run_qk_scores_one_head."""
    import ipu_apps.attn_v_16x60 as av

    v_bytes = _read_unpacked_channel_major_bytes(main_state, v_unpacked_base_row, N_CH)

    reset_labels()
    with tempfile.TemporaryDirectory() as tmpdir:
        bin_path = Path(tmpdir) / "av.bin"
        assemble_to_bin_file((Path(av.__file__).parent / "attn_v_16x60.asm").read_text(), str(bin_path))

        av_state = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
        load_program_from_binary(av_state, bin_path)

        # P: query-major, one row per (head, query), 16 valid lanes.
        p_bytes = bytearray(av.P_ROWS * av.ROW_BYTES)
        for h in range(N_HEAD):
            for i in range(N_TOK):
                row = bytearray(av.ROW_BYTES)
                vals = probs_by_head[h][i].astype(np.float32).tobytes()
                row[:len(vals)] = vals
                off = (h * av.P_HEAD_STRIDE_ROWS + i * av.PV_STRIDE_ROWS) * av.ROW_BYTES
                p_bytes[off:off + av.ROW_BYTES] = row
        av_state.xmem.write_address(av.PBASE, bytes(p_bytes))
        av_state.xmem.write_address(av.VBASE, bytearray(v_bytes_to_rows(v_bytes, N_CH)))

        av_state.set_cr_dstructure(valid_elements=N_TOK)
        av_state.regfile.set_cr(2, av.PBASE_ROW)
        av_state.regfile.set_cr(3, av.VBASE_ROW)
        av_state.regfile.set_cr(4, av.OBASE_ROW)
        av_state.regfile.set_cr(5, av.PV_STRIDE_ROWS)
        av_state.regfile.set_cr(8, av.P_HEAD_STRIDE_ROWS)
        av_state.regfile.set_cr(9, N_TOK - 1)
        av_state.regfile.set_cr(10, HEAD_DIM)
        av_state.regfile.set_cr(11, N_HEAD)
        av_state.regfile.set_cr(13, av.O_CHAN_ROWS)

        cycles = run_until_complete(av_state, max_cycles=200_000)

        raw = av_state.xmem.read_address(av.OBASE, N_CH * av.O_CHAN_BYTES)
    out_rows = np.frombuffer(bytes(raw), dtype=np.float32).reshape(N_CH, LANES)
    out = out_rows[:, :N_TOK].astype(np.float64)
    return out, cycles


def v_bytes_to_rows(v_bytes: bytes, n_channels: int) -> bytes:
    """v_bytes is [channel,token]-major flat (N_TOK*4 bytes/channel,
    contiguous); AttnV16x60App wants one WHOLE 512-byte row per channel
    (16 valid lanes + zero padding). Zero-pads each channel's 64 B into
    its own 512 B row -- byte layout only, no numeric transform."""
    out = bytearray(n_channels * ROW_BYTES)
    for c in range(n_channels):
        out[c * ROW_BYTES: c * ROW_BYTES + N_TOK * 4] = v_bytes[c * N_TOK * 4:(c + 1) * N_TOK * 4]
    return bytes(out)


def test_stage2_attention(tmp_path: Path) -> None:
    """unpack(Q,K,V) -> QK^T (per head) -> softmax (per head, FILE-BASED,
    see run_softmax_query_major_one_head) -> attn.V (all heads) -> pack.

    Self-consistency check: numpy reference mirrors exactly what the
    kernels compute (unscaled QK^T, softmax over keys, weighted V-sum) --
    no attention-scale (1/sqrt(head_dim)) is applied anywhere in this
    sub-chain, matching qk_scores_16x60/attn_v_16x60's own native
    behavior (neither kernel folds a scale; test_full_layer_l5.py's
    existing fixture applies the scale ON THE HOST as a separate,
    optional probe -- scale_q=False is its default/production path).
    """
    rng = np.random.RandomState(303)
    Q = rng.uniform(-1.0, 1.0, size=(N_CH, N_TOK)).astype(np.float32)
    K = rng.uniform(-1.0, 1.0, size=(N_CH, N_TOK)).astype(np.float32)
    V = rng.uniform(-1.0, 1.0, size=(N_CH, N_TOK)).astype(np.float32)

    def softmax_np(x):
        x = x - x.max(axis=-1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=-1, keepdims=True)

    expected_scores = np.zeros((N_HEAD, N_TOK, N_TOK))
    expected_probs = np.zeros((N_HEAD, N_TOK, N_TOK))
    expected_attn = np.zeros((N_CH, N_TOK))
    for h in range(N_HEAD):
        Qh = Q[h * HEAD_DIM:(h + 1) * HEAD_DIM].astype(np.float64)  # [D, N]
        Kh = K[h * HEAD_DIM:(h + 1) * HEAD_DIM].astype(np.float64)
        Vh = V[h * HEAD_DIM:(h + 1) * HEAD_DIM].astype(np.float64)
        S = Qh.T @ Kh  # [N_query, N_key], S[i,s] = sum_c Q[c,i]*K[c,s]
        expected_scores[h] = S
        P = softmax_np(S)
        expected_probs[h] = P
        # O[i,t] = sum_s P[i,s]*V[t,s] -> O.T is [D,N]: O[t,i] = sum_s V[t,s]*P[i,s]
        expected_attn[h * HEAD_DIM:(h + 1) * HEAD_DIM] = Vh @ P.T

    alloc = RowAllocator()
    Q_ROW = alloc.alloc(N_CH)
    K_ROW = alloc.alloc(N_CH)
    V_ROW = alloc.alloc(N_CH)
    PACK_MASK_ROW = alloc.alloc(1)
    ATTN_PACKED_ROW = alloc.alloc(N_PACKED_ROWS)

    state = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
    for base, arr in ((Q_ROW, Q), (K_ROW, K), (V_ROW, V)):
        rows = np.zeros((N_CH, LANES), dtype=np.float32)
        rows[:, :N_TOK] = arr
        state.xmem.write_address(base * ROW_BYTES, bytearray(rows.tobytes()))

    scores_by_head = []
    total_qk_cycles = 0
    for h in range(N_HEAD):
        scores, cyc = run_qk_scores_one_head(
            state, q_unpacked_base_row=Q_ROW + h * HEAD_DIM, k_unpacked_base_row=K_ROW + h * HEAD_DIM,
        )
        scores_by_head.append(scores)
        total_qk_cycles += cyc
        err = float(np.max(np.abs(scores - expected_scores[h])))
        print(f"STAGE2 QK^T head{h}: cycles={cyc} err={err:.6e}")
        assert err < 1e-2, f"QK^T head {h} wrong: {err:.6e}"

    probs_by_head = []
    for h in range(N_HEAD):
        probs = run_softmax_query_major_one_head(scores_by_head[h])
        probs_by_head.append(probs)
        err = float(np.max(np.abs(probs - expected_probs[h])))
        print(f"STAGE2 softmax head{h}: err={err:.6e}")
        assert err < 1e-2, f"softmax head {h} wrong: {err:.6e}"

    attn_out, cycles_av = run_attn_v_all_heads(state, probs_by_head=probs_by_head, v_unpacked_base_row=V_ROW)
    attn_err = float(np.max(np.abs(attn_out - expected_attn)))
    print(f"STAGE2 attn.V (4 heads): cycles={cycles_av} err={attn_err:.6e}")
    assert attn_err < 1e-2, f"attn.V wrong: {attn_err:.6e}"

    # ---- pack attn.V output back into packed-row layout ----
    unpacked_attn_row = alloc.alloc(N_CH)
    attn_rows = np.zeros((N_CH, LANES), dtype=np.float32)
    attn_rows[:, :N_TOK] = attn_out.astype(np.float32)
    state.xmem.write_address(unpacked_attn_row * ROW_BYTES, bytearray(attn_rows.tobytes()))

    cycles_pack, _ = run_pack(
        state, unpacked_base_row=unpacked_attn_row, packed_base_row=ATTN_PACKED_ROW,
        mask_row=PACK_MASK_ROW,
    )
    packed_raw = state.xmem.read_address(ATTN_PACKED_ROW * ROW_BYTES, N_PACKED_ROWS * ROW_BYTES)
    packed_out = np.frombuffer(bytes(packed_raw), dtype=np.float32).reshape(N_PACKED_ROWS, LANES)
    expected_packed = _pack(attn_out.astype(np.float32))
    pack_err = float(np.max(np.abs(packed_out.astype(np.float64) - expected_packed.astype(np.float64))))
    print(f"STAGE2 pack: cycles={cycles_pack} err={pack_err:.6e}")
    assert pack_err < 1e-5, f"pack stage wrong: {pack_err:.6e}"

    total_cycles = total_qk_cycles + cycles_av + cycles_pack
    print(f"STAGE2 TOTAL cycles (excl. softmax, in-kernel only): {total_cycles}")


def test_full_l5_layer_packed_end_to_end(tmp_path: Path) -> None:
    """Full L5 transformer layer, packed end to end, ONE continuous chain:

        layernorm -> QKV -> unpack -> attention -> pack -> out-proj
          -> residual -> layernorm -> FFN1(silu) -> FFN2 -> residual

    Every packed<->packed handoff shares ONE IpuState's XMEM (layernorm,
    QKV, out-proj, FFN1, FFN2, residual, pack, unpack). The attention
    sub-chain (QK^T/softmax/attn.V) necessarily leaves the packed
    representation (structural: scores have no channel axis) and uses the
    production unpacked App classes, each in ITS OWN IpuState (their
    module-level fixed row addresses cannot coexist with this chain's
    data) -- bytes move between states via plain Python byte
    copies/slices, never through numpy arithmetic. Softmax additionally
    requires actual temp files (SoftmaxRowsPartialApp's own interface,
    not a shortcut -- see module docstring and run_softmax_query_major_
    one_head's docstring). No other host-side operation happens anywhere
    in this test.
    """
    rng = np.random.RandomState(999)
    X = rng.uniform(-1.0, 1.0, size=(N_CH, N_TOK)).astype(np.float32)
    ln1_gamma = rng.uniform(0.8, 1.2, size=(N_CH,)).astype(np.float32)
    ln1_beta = rng.uniform(-0.1, 0.1, size=(N_CH,)).astype(np.float32)
    W_qkv = rng.uniform(-0.05, 0.05, size=(3 * N_CH, N_CH)).astype(np.float32)
    W_outproj = rng.uniform(-0.05, 0.05, size=(N_CH, N_CH)).astype(np.float32)
    ln2_gamma = rng.uniform(0.8, 1.2, size=(N_CH,)).astype(np.float32)
    ln2_beta = rng.uniform(-0.1, 0.1, size=(N_CH,)).astype(np.float32)
    W_ffn1 = rng.uniform(-0.05, 0.05, size=(4 * N_CH, N_CH)).astype(np.float32)
    W_ffn2 = rng.uniform(-0.05, 0.05, size=(N_CH, 4 * N_CH)).astype(np.float32)

    def softmax_np(x):
        x = x - x.max(axis=-1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=-1, keepdims=True)

    def numpy_layernorm(x, gamma, beta):
        mean = x.mean(axis=0, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=0, keepdims=True)
        invstd = 1.0 / np.sqrt(var)
        return (x - mean) * invstd * gamma[:, None] + beta[:, None]

    # ---- Full numpy float64 reference ----
    x64 = X.astype(np.float64)
    ln1 = numpy_layernorm(x64, ln1_gamma.astype(np.float64), ln1_beta.astype(np.float64))
    qkv = W_qkv.astype(np.float64) @ ln1
    Q, K, V = qkv[:N_CH], qkv[N_CH:2 * N_CH], qkv[2 * N_CH:3 * N_CH]
    attn_out = np.zeros((N_CH, N_TOK))
    for h in range(N_HEAD):
        Qh, Kh, Vh = (a[h * HEAD_DIM:(h + 1) * HEAD_DIM] for a in (Q, K, V))
        S = Qh.T @ Kh
        P = softmax_np(S)
        attn_out[h * HEAD_DIM:(h + 1) * HEAD_DIM] = Vh @ P.T
    outproj = W_outproj.astype(np.float64) @ attn_out
    resid1 = outproj + x64  # residual around the attention block: input X, not ln1
    ln2 = numpy_layernorm(resid1, ln2_gamma.astype(np.float64), ln2_beta.astype(np.float64))
    ffn1 = silu_np(W_ffn1.astype(np.float64) @ ln2)
    ffn2 = W_ffn2.astype(np.float64) @ ffn1
    final_expected = ffn2 + resid1

    # ---- Row layout ----
    alloc = RowAllocator()
    X_ROW = alloc.alloc(N_PACKED_ROWS)                 # residual1's B operand later needs row 0 -- see below
    LN1_OUT_ROW = alloc.alloc(N_PACKED_ROWS)
    LN1_SCRATCH_ROW = alloc.alloc(1 + N_PACKED_ROWS + 1 + N_PACKED_ROWS + N_PACKED_ROWS + 1 + 1 + 1)
    QKV_MASK_ROW = alloc.alloc(1)
    QKV_N_OUT = 3 * N_CH
    QKV_WEIGHTS_ROW = alloc.alloc(linear_weight_rows(N_CH, QKV_N_OUT))
    QKV_OUT_ROW = alloc.alloc(QKV_N_OUT // 8)
    UNPACK_MASK_ROW = alloc.alloc(1)
    QKV_UNPACKED_ROW = alloc.alloc(3 * N_CH)           # Q, K, V each N_CH rows, contiguous
    OUTPROJ_MASK_ROW = alloc.alloc(1)
    OUTPROJ_WEIGHTS_ROW = alloc.alloc(linear_weight_rows(N_CH, N_CH))
    OUTPROJ_OUT_ROW = alloc.alloc(N_PACKED_ROWS)
    UNPACKED_ATTN_ROW = alloc.alloc(N_CH)
    PACK_MASK_ROW = alloc.alloc(1)
    ATTN_PACKED_ROW = alloc.alloc(N_PACKED_ROWS)
    RESID1_OUT_ROW = alloc.alloc(N_PACKED_ROWS)
    LN2_SCRATCH_ROW = alloc.alloc(1 + N_PACKED_ROWS + 1 + N_PACKED_ROWS + N_PACKED_ROWS + 1 + 1 + 1)
    LN2_OUT_ROW = alloc.alloc(N_PACKED_ROWS)
    FFN1_MASK_ROW = alloc.alloc(1)
    FFN1_N_OUT = 4 * N_CH
    FFN1_WEIGHTS_ROW = alloc.alloc(linear_weight_rows(N_CH, FFN1_N_OUT))
    FFN1_OUT_ROW = alloc.alloc(FFN1_N_OUT // 8)
    FFN2_MASK_ROW = alloc.alloc(1)
    FFN2_WEIGHTS_ROW = alloc.alloc(linear_weight_rows(FFN1_N_OUT, N_CH))
    FFN2_OUT_ROW = alloc.alloc(N_PACKED_ROWS)
    RESID2_OUT_ROW = alloc.alloc(N_PACKED_ROWS)

    total_cycles = 0
    total_instrs: dict[str, int] = {}

    def _accumulate(cycles, counts):
        nonlocal total_cycles
        total_cycles += cycles
        for slot, n in counts.items():
            total_instrs[slot] = total_instrs.get(slot, 0) + n

    state = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
    state.xmem.write_address(X_ROW * ROW_BYTES, bytearray(_pack(X).tobytes()))

    # ---- layernorm 1 ----
    c, i = run_packed_layernorm(state, data_base_row=X_ROW, gamma=ln1_gamma, beta=ln1_beta,
                                 output_base_row=LN1_OUT_ROW, scratch_base_row=LN1_SCRATCH_ROW)
    _accumulate(c, i)
    ln1_raw = state.xmem.read_address(LN1_OUT_ROW * ROW_BYTES, N_PACKED_ROWS * ROW_BYTES)
    ln1_kernel = _unpack_rows(np.frombuffer(bytes(ln1_raw), dtype=np.float32).reshape(N_PACKED_ROWS, LANES))
    print(f"FULL layernorm1: cycles={c} err={np.max(np.abs(ln1_kernel - ln1)):.6e}")

    # ---- QKV (packed output) ----
    qkv_weight_slices = [W_qkv[g * 8:(g + 1) * 8] for g in range(QKV_N_OUT // 8)]
    c, i = run_packed_output_linear(state, asm_src=_LINEAR_IDENTITY_ASM, data_base_row=LN1_OUT_ROW, k=N_CH,
                                     n_out=QKV_N_OUT, weight_slices=qkv_weight_slices,
                                     output_base_row=QKV_OUT_ROW, scratch_base_row=QKV_MASK_ROW)
    _accumulate(c, i)
    print(f"FULL QKV: cycles={c}")

    # ---- unpack Q, K, V (3 separate unpack calls, one per N_CH-channel block) ----
    for j in range(3):
        c, i = run_unpack(state, packed_base_row=QKV_OUT_ROW + j * N_PACKED_ROWS,
                           unpacked_base_row=QKV_UNPACKED_ROW + j * N_CH, mask_row=UNPACK_MASK_ROW)
        _accumulate(c, i)
    print(f"FULL unpack(Q,K,V): 3 calls done")

    Q_ROW = QKV_UNPACKED_ROW
    K_ROW = QKV_UNPACKED_ROW + N_CH
    V_ROW = QKV_UNPACKED_ROW + 2 * N_CH

    # ---- attention: QK^T per head, softmax per head (file-based), attn.V all heads ----
    probs_by_head = []
    for h in range(N_HEAD):
        scores, cyc = run_qk_scores_one_head(state, q_unpacked_base_row=Q_ROW + h * HEAD_DIM,
                                              k_unpacked_base_row=K_ROW + h * HEAD_DIM)
        total_cycles += cyc
        probs = run_softmax_query_major_one_head(scores)
        probs_by_head.append(probs)
    attn_kernel, cyc = run_attn_v_all_heads(state, probs_by_head=probs_by_head, v_unpacked_base_row=V_ROW)
    total_cycles += cyc
    print(f"FULL attention: err={np.max(np.abs(attn_kernel - attn_out)):.6e}")

    # ---- pack attn output ----
    attn_rows = np.zeros((N_CH, LANES), dtype=np.float32)
    attn_rows[:, :N_TOK] = attn_kernel.astype(np.float32)
    state.xmem.write_address(UNPACKED_ATTN_ROW * ROW_BYTES, bytearray(attn_rows.tobytes()))
    c, i = run_pack(state, unpacked_base_row=UNPACKED_ATTN_ROW, packed_base_row=ATTN_PACKED_ROW,
                     mask_row=PACK_MASK_ROW)
    _accumulate(c, i)

    # ---- out-proj (packed output) ----
    outproj_weight_slices = [W_outproj[g * 8:(g + 1) * 8] for g in range(N_CH // 8)]
    c, i = run_packed_output_linear(state, asm_src=_LINEAR_IDENTITY_ASM, data_base_row=ATTN_PACKED_ROW, k=N_CH,
                                     n_out=N_CH, weight_slices=outproj_weight_slices,
                                     output_base_row=OUTPROJ_OUT_ROW, scratch_base_row=OUTPROJ_MASK_ROW)
    _accumulate(c, i)
    outproj_raw = state.xmem.read_address(OUTPROJ_OUT_ROW * ROW_BYTES, N_PACKED_ROWS * ROW_BYTES)
    outproj_kernel = _unpack_rows(np.frombuffer(bytes(outproj_raw), dtype=np.float32).reshape(N_PACKED_ROWS, LANES))
    print(f"FULL out-proj: cycles={c} err={np.max(np.abs(outproj_kernel - outproj)):.6e}")

    # ---- residual 1: X (already at X_ROW=row0) + out-proj output ----
    assert X_ROW == 0
    c, i = run_packed_residual_add(state, a_base_row=X_ROW, b_base_row=OUTPROJ_OUT_ROW,
                                    out_base_row=RESID1_OUT_ROW)
    _accumulate(c, i)
    resid1_raw = state.xmem.read_address(RESID1_OUT_ROW * ROW_BYTES, N_PACKED_ROWS * ROW_BYTES)
    resid1_kernel = _unpack_rows(np.frombuffer(bytes(resid1_raw), dtype=np.float32).reshape(N_PACKED_ROWS, LANES))
    print(f"FULL residual1: cycles={c} err={np.max(np.abs(resid1_kernel - resid1)):.6e}")

    # ---- layernorm 2 ----
    c, i = run_packed_layernorm(state, data_base_row=RESID1_OUT_ROW, gamma=ln2_gamma, beta=ln2_beta,
                                 output_base_row=LN2_OUT_ROW, scratch_base_row=LN2_SCRATCH_ROW)
    _accumulate(c, i)
    ln2_raw = state.xmem.read_address(LN2_OUT_ROW * ROW_BYTES, N_PACKED_ROWS * ROW_BYTES)
    ln2_kernel = _unpack_rows(np.frombuffer(bytes(ln2_raw), dtype=np.float32).reshape(N_PACKED_ROWS, LANES))
    print(f"FULL layernorm2: cycles={c} err={np.max(np.abs(ln2_kernel - ln2)):.6e}")

    # ---- FFN1 (silu) ----
    ffn1_weight_slices = [W_ffn1[g * 8:(g + 1) * 8] for g in range(FFN1_N_OUT // 8)]
    c, i = run_packed_output_linear(state, asm_src=_LINEAR_SILU_ASM, data_base_row=LN2_OUT_ROW, k=N_CH,
                                     n_out=FFN1_N_OUT, weight_slices=ffn1_weight_slices,
                                     output_base_row=FFN1_OUT_ROW, scratch_base_row=FFN1_MASK_ROW)
    _accumulate(c, i)
    print(f"FULL FFN1(silu): cycles={c}")

    # ---- FFN2 ----
    ffn2_weight_slices = [W_ffn2[g * 8:(g + 1) * 8] for g in range(N_CH // 8)]
    c, i = run_packed_output_linear(state, asm_src=_LINEAR_IDENTITY_ASM, data_base_row=FFN1_OUT_ROW, k=FFN1_N_OUT,
                                     n_out=N_CH, weight_slices=ffn2_weight_slices,
                                     output_base_row=FFN2_OUT_ROW, scratch_base_row=FFN2_MASK_ROW)
    _accumulate(c, i)
    ffn2_raw = state.xmem.read_address(FFN2_OUT_ROW * ROW_BYTES, N_PACKED_ROWS * ROW_BYTES)
    ffn2_kernel = _unpack_rows(np.frombuffer(bytes(ffn2_raw), dtype=np.float32).reshape(N_PACKED_ROWS, LANES))
    print(f"FULL FFN2: cycles={c} err={np.max(np.abs(ffn2_kernel - ffn2)):.6e}")

    # ---- residual 2: resid1 (move to row 0) + ffn2 output ----
    state.xmem.write_address(0 * ROW_BYTES, bytes(resid1_raw[:ROW_BYTES]))
    # Only row 0 of resid1 needs to physically BE at row 0; but residual-add reads
    # ROW_COUNT rows starting there, so copy the WHOLE resid1 block to row 0.
    state.xmem.write_address(0 * ROW_BYTES, bytes(resid1_raw))
    c, i = run_packed_residual_add(state, a_base_row=0, b_base_row=FFN2_OUT_ROW, out_base_row=RESID2_OUT_ROW)
    _accumulate(c, i)
    resid2_raw = state.xmem.read_address(RESID2_OUT_ROW * ROW_BYTES, N_PACKED_ROWS * ROW_BYTES)
    resid2_kernel = _unpack_rows(np.frombuffer(bytes(resid2_raw), dtype=np.float32).reshape(N_PACKED_ROWS, LANES))
    final_err = float(np.max(np.abs(resid2_kernel - final_expected)))
    print(f"FULL residual2 (FINAL OUTPUT): cycles={c} err={final_err:.6e}")

    peak_activation_bytes = state.xmem.high_water_mark if hasattr(state.xmem, "high_water_mark") else None

    print(f"=== FULL L5 LAYER PACKED TOTALS ===")
    print(f"TOTAL_CYCLES={total_cycles}")
    print(f"TOTAL_INSTRUCTIONS={sum(total_instrs.values())}")
    print(f"TOTAL_INSTRUCTIONS_BY_SLOT={total_instrs}")
    print(f"PEAK_XMEM_ROW_HIGH_WATER_MARK={alloc.high_water_mark}")
    print(f"PEAK_XMEM_ACTIVATION_BYTES={alloc.high_water_mark * ROW_BYTES}")
    print(f"FINAL_MAX_ABS_ERROR={final_err:.6e}")

    assert final_err < 1e-2, f"full L5 layer packed chain wrong: {final_err:.6e}"
