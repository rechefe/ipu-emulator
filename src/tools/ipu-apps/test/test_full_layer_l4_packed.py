"""Full L4 transformer layer, PACKED end to end:

    layernorm -> QKV -> unpack -> attention -> pack -> out-proj
      -> residual -> layernorm -> FFN1(silu) -> FFN2 -> residual

L4 shape: 192 channels, N_TOK=64 tokens, N_HEAD=4, HEAD_DIM=48, packing
factor 2 (partition_size(64)=64 -> 128/64=2, see docs/isa_friction_log.md
and asm_packed_output_linear_generic_p4.asm's header -- vs L5's factor 8).

Every handoff between STANDALONE PACKED kernels (layernorm, linear/output,
residual, pack/unpack) goes through a single shared IpuState's XMEM --
verified by writing intermediate activations ONLY via state.xmem, never via
numpy in between two packed-kernel runs. The one stage that leaves the
packed representation is QK^T/softmax/attn.V (structural: scores have no
channel axis; softmax is David Sheinenzon's kernel, not modified). This
chain uses the QUERY-MAJOR mapping (qk_scores_64x48 + attn_v_64x48, AGG-
based) -- each called ONCE TOTAL, internally covering all
P_STREAM*N_HEAD=16 (stream, head) blocks, exactly matching how the
production unpacked L4 suite (test_full_layer_l4.py) exercises this
mapping. The key-major mapping (attn_scores_km_64x48 + attn_v_bcast_48,
ACC.ADD, no AGG, called once per head) is NOT exercised here -- it is a
distinct, non-mixable chain (see kernel_docs/kernel_layer_map.md's "Two
attention mappings -- never mix them"); this choice is a deliberate default
(single-call-per-kernel, same shape as L5's own chain), not a completeness
gap, and is called out explicitly per the task brief.

THE ONE REMAINING HOST-SIDE OPERATION: softmax. SoftmaxRowsPartialApp's
_pack_input()/teardown() do a REAL host-side (Python struct-level) repack
between row-major-file and its own partitioned-chunk layout -- this is not
a pass-through XMEM write like every other App class in this chain, it is
computation on the host between two kernel boundaries. Reported here as the
same unavoidable host-side/file boundary the L5 session found and reported,
not silently worked around.

STANDALONE: does not modify layernorm_64x192, qk_scores_64x48,
attn_v_64x48, attn_scores_km_64x48, attn_v_bcast_48, residual_add_64x192,
proj_*_p4, or any softmax kernel. Uses them read-only. No BUILD.bazel
target.
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
from ipu_emu.emulator import load_program_from_binary, run_until_complete
from ipu_emu.ipu_state import IpuState, WideVectorArithmetic
from ipu_emu.ipu_config import encode_dstructure, Partition

from fixture_packed_l4_measure import count_instructions

from ipu_apps.qk_scores_64x48 import QkScores64x48App
from ipu_apps.attn_v_64x48 import AttnV64x48App
from ipu_apps.softmax.softmax_rows_partial import SoftmaxRowsPartialApp

N_CH = 192
N_TOK = 64
P_STREAM = 4
N_HEAD = 4
HEAD_DIM = 48
N_BLOCK = P_STREAM * N_HEAD  # 16
LANES = 128
ROW_BYTES = 512
PACK = 2
N_PACKED_ROWS = N_CH // PACK
assert N_CH % PACK == 0

_PACKED_LAYERNORM_ASM = Path(__file__).resolve().parent / "asm_packed_layernorm_192x64.asm"
_UNPACK_ASM = Path(__file__).resolve().parent / "asm_packed_unpack_192x64.asm"
_PACK_ASM = Path(__file__).resolve().parent / "asm_packed_pack_192x64.asm"
_LINEAR_IDENTITY_ASM = Path(__file__).resolve().parent / "asm_packed_output_linear_generic_p4.asm"
_LINEAR_SILU_ASM = Path(__file__).resolve().parent / "asm_packed_output_linear_silu_p4.asm"
_RESIDUAL_ASM = Path(__file__).resolve().parent / "asm_packed_residual_add_192x64.asm"


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


def _mask_row_2() -> bytes:
    mrow = bytearray(128)
    for p_out in range(PACK):
        bits = 0
        for b in range(N_TOK * p_out, N_TOK * p_out + N_TOK):
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
        assert w % PACK == 0
        widths.append(w)
        remaining -= w
    return widths


def linear_weight_rows(k: int, n_out: int) -> int:
    """Row count asm_packed_output_linear_*_p4.asm's weight region needs
    for a given (k, n_out) shape: PACK rows per weight-chunk (one per
    p_out), times ceil(k/128) weight-chunks, times n_out/PACK groups. Same
    derivation as the L5 session's linear_weight_rows, with PACK=2 in
    place of L5's PACK=8 (see that session's report for the k/PACK
    overshoot bug this form avoids)."""
    w_chunks = len(_chunk_widths(k))
    return PACK * w_chunks * (n_out // PACK)


class RowAllocator:
    """Sequential XMEM row-region allocator -- avoids hand-computed row
    arithmetic, same pattern as the L5 session's RowAllocator."""

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
# ---------------------------------------------------------------------------

def run_packed_layernorm(state: IpuState, *, data_base_row: int, gamma: np.ndarray,
                          beta: np.ndarray, output_base_row: int,
                          scratch_base_row: int, max_cycles: int = 200_000) -> tuple[int, dict]:
    neg_mean_tile_row = scratch_base_row
    centered_base_row = neg_mean_tile_row + 1
    invstd_tile_row = centered_base_row + N_PACKED_ROWS
    gamma_tile_base_row = invstd_tile_row + 1
    beta_tile_base_row = gamma_tile_base_row + N_PACKED_ROWS
    mask_row = beta_tile_base_row + N_PACKED_ROWS
    allones_mask_row = mask_row + 1
    scratch64_row = allones_mask_row + 1

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
        state.xmem.write_address(mask_row * ROW_BYTES, _mask_row_2())
        state.xmem.write_address(allones_mask_row * ROW_BYTES, bytes([0xFF] * 128))

        state.regfile.set_cr(2, data_base_row)
        state.regfile.set_cr(3, N_PACKED_ROWS)
        state.regfile.set_cr(4, 1)
        state.regfile.set_cr(5, scratch64_row)
        state.regfile.set_cr(6, neg_mean_tile_row)
        state.regfile.set_cr(7, centered_base_row)
        state.regfile.set_cr(8, invstd_tile_row)
        state.regfile.set_cr(9, gamma_tile_base_row)
        state.regfile.set_cr(10, beta_tile_base_row)
        state.regfile.set_cr(11, output_base_row)
        state.regfile.set_cr(12, mask_row)
        state.regfile.set_cr(13, encode_dstructure(valid_elements=N_TOK))
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
        state.xmem.write_address(mask_row * ROW_BYTES, _mask_row_2())
        state.regfile.set_cr(2, packed_base_row)
        state.regfile.set_cr(3, N_PACKED_ROWS)
        state.regfile.set_cr(4, 1)
        state.regfile.set_cr(5, unpacked_base_row)
        state.regfile.set_cr(6, mask_row)
        state.regfile.set_cr(7, encode_dstructure(valid_elements=N_TOK))
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
        state.xmem.write_address(mask_row * ROW_BYTES, _mask_row_2())
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
    """weight_slices: list of n_out//PACK arrays, each [PACK, k]. Writes
    n_out//PACK packed output rows starting at output_base_row."""
    widths = _chunk_widths(k)
    w_chunks = len(widths)
    mask_row = scratch_base_row
    weights_base_row = mask_row + 1

    rendered = jinja2.Template(asm_src.read_text()).render(chunk_widths=widths)
    state.xmem.write_address(mask_row * ROW_BYTES, _mask_row_2())

    total_cycles = 0
    total_counts: dict[str, int] = {}

    for group in range(n_out // PACK):
        W2 = weight_slices[group]
        this_weights_base = weights_base_row + group * PACK * w_chunks
        this_output_row = output_base_row + group

        reset_labels()
        with tempfile.TemporaryDirectory() as tmpdir:
            bin_path = Path(tmpdir) / "lin.bin"
            assemble_to_bin_file(rendered, str(bin_path))
            load_program_from_binary(state, bin_path)
            state.program_counter = 0

            w_rows = np.zeros((PACK * w_chunks, LANES), dtype=np.float32)
            for c, width in enumerate(widths):
                off = sum(widths[:c])
                for p_out in range(PACK):
                    w_rows[c * PACK + p_out, :width] = W2[p_out, off:off + width]
            state.xmem.write_address(this_weights_base * ROW_BYTES, bytearray(w_rows.tobytes()))

            state.regfile.set_cr(2, data_base_row)
            state.regfile.set_cr(3, this_weights_base)
            state.regfile.set_cr(4, this_output_row)
            state.regfile.set_cr(5, mask_row)
            state.regfile.set_cr(6, encode_dstructure(valid_elements=128, partition=Partition.P2))
            state.regfile.set_cr(7, encode_dstructure(valid_elements=128))
            for p_out in range(PACK):
                seed = (512 - 64 * p_out - 64) % 512
                state.regfile.set_cr(8 + p_out, seed)

            with count_instructions() as counts:
                cycles = run_until_complete(state, max_cycles=max_cycles)
            total_cycles += cycles
            for slot, n in counts.by_slot.items():
                total_counts[slot] = total_counts.get(slot, 0) + n

    return total_cycles, total_counts


def run_packed_residual_add(state: IpuState, *, a_base_row: int, b_base_row: int,
                             out_base_row: int, max_cycles: int = 100_000) -> tuple[int, dict]:
    """asm_packed_residual_add_192x64.asm hardcodes A_BASE=cr0 (hardwired
    read-only ZERO), so A's base row MUST be 0 -- same constraint as the L5
    residual-add kernel."""
    assert a_base_row == 0, (
        "asm_packed_residual_add_192x64.asm hardcodes A_BASE=cr0 (hardwired "
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
        # Cross-kernel R_MASK bleed hazard (documented in docs/isa_friction_log.md,
        # first found in the L5 session): explicitly restore the all-ones mask
        # before every call, since this kernel relies on R_MASK's regfile-init
        # default and the packed-output-linear kernel run just before it leaves
        # R_MASK restricted to a one-hot 2-slot mask.
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
# Attention sub-chain: query-major QK^T (one call, all 16 blocks) -> softmax
# (per block, file-based) -> attn.V (one call, all 16 blocks).
# ---------------------------------------------------------------------------

def run_attention_query_major(main_state: IpuState, *, q_unpacked_base_row: int,
                               k_unpacked_base_row: int, v_unpacked_base_row: int,
                               tmp_path: Path) -> tuple[np.ndarray, int]:
    """Q/K/V are staged unpacked, channel-major, N_CH=D_MODEL=192 rows each
    (one real "stream" worth of data, exactly what the packed chain
    computes) in main_state's XMEM. qk_scores_64x48/attn_v_64x48 always
    process their full fixed N_BLOCK=16 (stream, head) loop bound
    internally regardless of how much real data exists -- this test has
    only ONE stream's worth (N_HEAD=4 blocks) of genuine Q/K/V, so blocks
    4..15 (the other 3 "streams" the kernel's loop bound expects) are
    given deterministic zero input here: the kernel computes attention
    over them like any other block, but this test only verifies blocks
    0..3 (the real data) against the numpy reference -- it does not
    fabricate an independent reference for the zero-filled blocks. Runs
    QkScores64x48App ONCE (all 16 blocks), softmax once PER BLOCK (16
    file-based calls -- softmax has no multi-block batching interface),
    then AttnV64x48App ONCE (all 16 blocks). Dedicated IpuStates for the
    QK/AttnV kernels (their module-level fixed row addresses cannot
    coexist with main_state's own row-0 packed data) -- bytes move via
    plain byte slicing, never numpy arithmetic, matching the L5 session's
    established pattern. Returns (attn_out [N_BLOCK, HEAD_DIM, N_TOK]
    float64, total cycles excluding softmax)."""
    import ipu_apps.qk_scores_64x48 as qk
    import ipu_apps.attn_v_64x48 as av

    # ---- Stage Q/K into qk_scores_64x48's expected [block, channel, token] files ----
    def _read_block_channel_major(base_row: int) -> np.ndarray:
        """main_state holds Q/K/V unpacked, channel-major: N_CH=192 rows
        (one real stream's worth, block=head, block*D+channel), N_TOK
        valid lanes each. Returns [N_BLOCK, D, N_TOK] float32, zero-padded
        for blocks N_HEAD..N_BLOCK-1 (no real data exists there -- see
        docstring above)."""
        out = np.zeros((N_BLOCK, HEAD_DIM, N_TOK), dtype=np.float32)
        for b in range(N_HEAD):
            for c in range(HEAD_DIM):
                row = main_state.xmem.read_address((base_row + b * HEAD_DIM + c) * ROW_BYTES, N_TOK * 4)
                out[b, c] = np.frombuffer(bytes(row), dtype=np.float32)
        return out

    q_arr = _read_block_channel_major(q_unpacked_base_row)
    k_arr = _read_block_channel_major(k_unpacked_base_row)
    v_arr = _read_block_channel_major(v_unpacked_base_row)

    q_path = tmp_path / "q.bin"
    k_path = tmp_path / "k.bin"
    q_path.write_bytes(q_arr.tobytes())
    k_path.write_bytes(k_arr.tobytes())

    reset_labels()
    qk_bin = tmp_path / "qk.bin"
    assemble_to_bin_file((Path(qk.__file__).parent / "qk_scores_64x48.asm").read_text(), str(qk_bin))
    qk_state = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
    scores_path = tmp_path / "scores.bin"
    qk_app = QkScores64x48App(inst_path=qk_bin, query_path=q_path, key_path=k_path, output_path=scores_path)
    _, qk_cycles = qk_app.run(max_cycles=2_000_000, state=qk_state)

    scores = np.frombuffer(scores_path.read_bytes(), dtype=np.float32).reshape(N_BLOCK, N_TOK, N_TOK)

    # ---- softmax, per block (no multi-block batching interface) ----
    probs = np.zeros((N_BLOCK, N_TOK, N_TOK), dtype=np.float64)
    for b in range(N_BLOCK):
        with tempfile.TemporaryDirectory() as smdir:
            inst_dir = Path(SoftmaxRowsPartialApp.__module__.replace(".", "/")).parent
            asm_path = (Path(__file__).resolve().parents[1] / "src" / inst_dir /
                        "softmax_rows_partial" / "softmax_rows_partial.asm")
            inst_path = Path(smdir) / "softmax.bin"
            reset_labels()
            assemble_to_bin_file(asm_path.read_text(), str(inst_path))

            inp = Path(smdir) / "in.bin"
            outp = Path(smdir) / "out.bin"
            inp.write_bytes(scores[b].astype(np.float32).tobytes())
            sm_app = SoftmaxRowsPartialApp(inst_path=inst_path, input_path=inp, output_path=outp,
                                            n=N_TOK, rows=N_TOK)
            sm_app.run(max_cycles=20_000_000)
            probs[b] = np.frombuffer(outp.read_bytes(), dtype=np.float32).reshape(N_TOK, N_TOK)

    # ---- attn.V, all 16 blocks in one call ----
    # P: query-major, one row per (block, query), N_TOK valid lanes.
    p_bytes = bytearray(av.P_ROWS * av.ROW_BYTES)
    for b in range(N_BLOCK):
        for i in range(N_TOK):
            row = bytearray(av.ROW_BYTES)
            vals = probs[b, i].astype(np.float32).tobytes()
            row[:len(vals)] = vals
            off = (b * av.P_BLOCK_ROWS + i * av.PV_STRIDE_ROWS) * av.ROW_BYTES
            p_bytes[off:off + av.ROW_BYTES] = row
    p_path = tmp_path / "p.bin"
    p_path.write_bytes(bytes(p_bytes))

    # V: channel-major, one row per (block, channel), N_TOK valid lanes.
    v_bytes = bytearray(av.V_ROWS * av.ROW_BYTES)
    for b in range(N_BLOCK):
        for c in range(HEAD_DIM):
            row = bytearray(av.ROW_BYTES)
            vals = v_arr[b, c].tobytes()
            row[:len(vals)] = vals
            off = (b * HEAD_DIM + c) * av.ROW_BYTES
            v_bytes[off:off + av.ROW_BYTES] = row
    v_path = tmp_path / "v.bin"
    v_path.write_bytes(bytes(v_bytes))

    reset_labels()
    av_bin = tmp_path / "av.bin"
    assemble_to_bin_file((Path(av.__file__).parent / "attn_v_64x48.asm").read_text(), str(av_bin))
    av_state = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
    out_path = tmp_path / "attn_out.bin"
    av_app = AttnV64x48App(inst_path=av_bin, p_path=p_path, v_path=v_path, output_path=out_path)
    _, av_cycles = av_app.run(max_cycles=2_000_000, state=av_state)

    # AttnV64x48App's teardown crops N_CHAN=N_BLOCK*HEAD_DIM=768 rows (one per
    # (block,channel), block-major: row b*HEAD_DIM+c), each to its N_TOK
    # valid lanes -- already exactly [N_BLOCK, HEAD_DIM, N_TOK] with no
    # further reshape needed (an earlier draft incorrectly reshaped this to
    # [N_CH=192, N_TOK], silently dropping 3 of the 4 "stream" blocks'
    # worth of data -- caught by the array-size mismatch, not a numeric
    # near-miss, since 192*64 != 768*64).
    attn_out = np.frombuffer(out_path.read_bytes(), dtype=np.float32).reshape(N_BLOCK, HEAD_DIM, N_TOK).astype(np.float64)

    return attn_out, qk_cycles + av_cycles


# ---------------------------------------------------------------------------
# Full end-to-end packed L4 layer test
# ---------------------------------------------------------------------------

def test_full_l4_layer_packed_end_to_end(tmp_path: Path) -> None:
    """Full L4 transformer layer, packed end to end, ONE continuous chain
    for every packed<->packed handoff, matching the shape of
    test_full_l5_layer_packed_end_to_end. Attention is query-major
    (qk_scores_64x48 + attn_v_64x48), each called once total."""
    rng = np.random.RandomState(4001)
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

    # ---- Full numpy float64 reference, independently built for this shape ----
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
    resid1 = outproj + x64
    ln2 = numpy_layernorm(resid1, ln2_gamma.astype(np.float64), ln2_beta.astype(np.float64))
    ffn1 = silu_np(W_ffn1.astype(np.float64) @ ln2)
    ffn2 = W_ffn2.astype(np.float64) @ ffn1
    final_expected = ffn2 + resid1

    # NOTE: this test exercises ONE stream's worth of the attention
    # computation (P_STREAM=1 effectively, block = head only, N_HEAD=4
    # real blocks) -- qk_scores_64x48/attn_v_64x48 always process their
    # fixed N_BLOCK=16 (stream, head) loop bound internally regardless, so
    # blocks 4..15 (the other 3 "streams" the kernel's loop bound expects)
    # are zero-filled by run_attention_query_major (see that function's
    # docstring) rather than given a second, independently-fabricated
    # random reference -- only the N_HEAD=4 real blocks are checked below.
    attn_out_full = np.zeros((N_BLOCK, HEAD_DIM, N_TOK))
    for h in range(N_HEAD):
        attn_out_full[h] = attn_out[h * HEAD_DIM:(h + 1) * HEAD_DIM]

    # ---- Row layout ----
    alloc = RowAllocator()
    X_ROW = alloc.alloc(N_PACKED_ROWS)
    LN1_OUT_ROW = alloc.alloc(N_PACKED_ROWS)
    LN1_SCRATCH_ROW = alloc.alloc(1 + N_PACKED_ROWS + 1 + N_PACKED_ROWS + N_PACKED_ROWS + 1 + 1 + 1)
    QKV_MASK_ROW = alloc.alloc(1)
    QKV_N_OUT = 3 * N_CH
    QKV_WEIGHTS_ROW = alloc.alloc(linear_weight_rows(N_CH, QKV_N_OUT))
    QKV_OUT_ROW = alloc.alloc(QKV_N_OUT // PACK)
    UNPACK_MASK_ROW = alloc.alloc(1)
    QKV_UNPACKED_ROW = alloc.alloc(3 * N_CH)
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
    FFN1_OUT_ROW = alloc.alloc(FFN1_N_OUT // PACK)
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
    qkv_weight_slices = [W_qkv[g * PACK:(g + 1) * PACK] for g in range(QKV_N_OUT // PACK)]
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

    # ---- attention (query-major, one call each for QK^T and attn.V) ----
    attn_kernel, cyc = run_attention_query_major(
        state, q_unpacked_base_row=Q_ROW, k_unpacked_base_row=K_ROW, v_unpacked_base_row=V_ROW,
        tmp_path=tmp_path,
    )
    total_cycles += cyc
    # attn_kernel is [N_BLOCK, HEAD_DIM, N_TOK]; only blocks 0..N_HEAD-1
    # carry real data (see run_attention_query_major's docstring) -- blocks
    # N_HEAD..N_BLOCK-1 were zero-filled input, so their kernel output is
    # not compared against a fabricated reference.
    attn_err = float(np.max(np.abs(attn_kernel[:N_HEAD] - attn_out_full[:N_HEAD])))
    print(f"FULL attention ({N_HEAD} real blocks of {N_BLOCK}): cycles={cyc} err={attn_err:.6e}")
    assert attn_err < 1e-2, f"attention sub-chain wrong: {attn_err:.6e}"
    # Only the first N_HEAD=4 blocks correspond to this test's single real
    # attention sequence (the numpy `attn_out` computed from the full-layer
    # weight chain above); use those for the rest of the chain.
    attn_kernel_seq = attn_kernel[:N_HEAD].reshape(N_CH, N_TOK)

    # ---- pack attn output ----
    attn_rows = np.zeros((N_CH, LANES), dtype=np.float32)
    attn_rows[:, :N_TOK] = attn_kernel_seq.astype(np.float32)
    state.xmem.write_address(UNPACKED_ATTN_ROW * ROW_BYTES, bytearray(attn_rows.tobytes()))
    c, i = run_pack(state, unpacked_base_row=UNPACKED_ATTN_ROW, packed_base_row=ATTN_PACKED_ROW,
                     mask_row=PACK_MASK_ROW)
    _accumulate(c, i)

    # ---- out-proj (packed output) ----
    outproj_weight_slices = [W_outproj[g * PACK:(g + 1) * PACK] for g in range(N_CH // PACK)]
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
    ffn1_weight_slices = [W_ffn1[g * PACK:(g + 1) * PACK] for g in range(FFN1_N_OUT // PACK)]
    c, i = run_packed_output_linear(state, asm_src=_LINEAR_SILU_ASM, data_base_row=LN2_OUT_ROW, k=N_CH,
                                     n_out=FFN1_N_OUT, weight_slices=ffn1_weight_slices,
                                     output_base_row=FFN1_OUT_ROW, scratch_base_row=FFN1_MASK_ROW)
    _accumulate(c, i)
    print(f"FULL FFN1(silu): cycles={c}")

    # ---- FFN2 ----
    ffn2_weight_slices = [W_ffn2[g * PACK:(g + 1) * PACK] for g in range(N_CH // PACK)]
    c, i = run_packed_output_linear(state, asm_src=_LINEAR_IDENTITY_ASM, data_base_row=FFN1_OUT_ROW, k=FFN1_N_OUT,
                                     n_out=N_CH, weight_slices=ffn2_weight_slices,
                                     output_base_row=FFN2_OUT_ROW, scratch_base_row=FFN2_MASK_ROW)
    _accumulate(c, i)
    ffn2_raw = state.xmem.read_address(FFN2_OUT_ROW * ROW_BYTES, N_PACKED_ROWS * ROW_BYTES)
    ffn2_kernel = _unpack_rows(np.frombuffer(bytes(ffn2_raw), dtype=np.float32).reshape(N_PACKED_ROWS, LANES))
    print(f"FULL FFN2: cycles={c} err={np.max(np.abs(ffn2_kernel - ffn2)):.6e}")

    # ---- residual 2: resid1 (move to row 0) + ffn2 output ----
    state.xmem.write_address(0 * ROW_BYTES, bytes(resid1_raw))
    c, i = run_packed_residual_add(state, a_base_row=0, b_base_row=FFN2_OUT_ROW, out_base_row=RESID2_OUT_ROW)
    _accumulate(c, i)
    resid2_raw = state.xmem.read_address(RESID2_OUT_ROW * ROW_BYTES, N_PACKED_ROWS * ROW_BYTES)
    resid2_kernel = _unpack_rows(np.frombuffer(bytes(resid2_raw), dtype=np.float32).reshape(N_PACKED_ROWS, LANES))
    final_err = float(np.max(np.abs(resid2_kernel - final_expected)))
    print(f"FULL residual2 (FINAL OUTPUT): cycles={c} err={final_err:.6e}")

    peak_activation_bytes = alloc.high_water_mark * ROW_BYTES

    print(f"=== FULL L4 LAYER PACKED TOTALS ===")
    print(f"TOTAL_CYCLES={total_cycles}")
    print(f"TOTAL_INSTRUCTIONS={sum(total_instrs.values())}")
    print(f"TOTAL_INSTRUCTIONS_BY_SLOT={total_instrs}")
    print(f"PEAK_XMEM_ROW_HIGH_WATER_MARK={alloc.high_water_mark}")
    print(f"PEAK_XMEM_ACTIVATION_BYTES={peak_activation_bytes}")
    print(f"FINAL_MAX_ABS_ERROR={final_err:.6e}")

    assert final_err < 1e-2, f"full L4 layer packed chain wrong: {final_err:.6e}"
