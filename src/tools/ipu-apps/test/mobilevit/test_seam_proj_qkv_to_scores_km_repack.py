"""Seam bridge: proj_qkv_192_p4 / proj_qkv_240_p4 (QKV, all P=4 streams in one
invocation) -> attn_scores_km_64x48 / attn_scores_km_16x60.

Investigation 2 (prior seam audit) established: the QKV projection's output
cannot feed the KM score kernel via PURE base+stride addressing alone -- the
store pitch disagrees (512 B/channel, padded, vs N*4 B/channel, tightly
packed). It CAN be bridged with a minimal, mechanical repack: strip each row's
padding lanes, with no reordering, no transpose, no value transform. This
module provides that repack as a real, reusable function (`repack_qkv_to_km`)
rather than leaving it as inline test-only code, and proves two separate
claims:

1. The repack is padding-strip-only -- verified by constructing an input
   where every (stream, head, channel, token) index carries a UNIQUE,
   identifiable value, so any accidental permutation (swapped head/channel
   axes, wrong stream order, transposed token axis) would produce a
   detectably wrong element at some index, not just a numerically-close
   wrong answer.
2. The repack is a strict superset of what the prior single-stream test
   proved: it now feeds `attn_scores_km_64x48` directly from `proj_qkv_192_p4`
   (ONE host invocation covering all 4 streams) rather than from four
   separate `matmul_576x192_x128` runs -- proving the multi-stream kernel and
   the repack compose correctly together, at both L4 and L5.

Method: poison the score kernel's destination XMEM region before running it,
assert on raw stored bytes, and use a mutation-first control (corrupt one
repacked element, confirm the run diverges) before trusting the clean pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from ipu_apps.kernels.attention.attn_scores_km_16x60 import app as km5
from ipu_apps.kernels.attention.attn_scores_km_64x48 import app as km4
from ipu_apps.kernels.projections.proj_qkv_192_p4 import app as qkv4
from ipu_apps.kernels.projections.proj_qkv_240_p4 import app as qkv5

from fixture_kernel_support import kernel_inst, poison

# attn_scores_km_16x60 (unlike attn_scores_km_64x48) processes ONE stream's
# Q/K per invocation -- its input file is canonical [N_HEADS*D, N_TOK]
# per-head-channel-major with NO stream axis at all (see its
# _load_q_channel_major docstring). A full L5 chain therefore still needs P=4
# score-kernel calls, one per stream, each fed that stream's repacked Q/K block.

# Q/K/V block boundaries within a stream's N_OUT QKV-matmul output rows,
# shared shape for both layers: N_OUT = 3 * N_HEAD * D.
_Q_ROW0 = 0


def repack_qkv_to_km(
    stream_outputs: list[np.ndarray], *, n_head: int, d: int, n_tok: int, row0: int,
) -> np.ndarray:
    """Strip-padding repack: proj_qkv_*_p4's per-stream [N_OUT, LANES] raw
    output rows -> attn_scores_km_*'s canonical [P, N_HEAD, D, N] flat block.

    stream_outputs[p] is stream p's full [N_OUT, LANES] output (padded rows,
    channel c of head h at row row0 + h*d + c). This slices each row's first
    n_tok lanes (the ONLY transform -- padding strip) and stacks by stream.
    No reordering: row order, head order, and channel order are preserved
    exactly as the projection kernel produced them.
    """
    P = len(stream_outputs)
    blocks = np.stack([
        stream_outputs[p][row0:row0 + n_head * d, :n_tok]
        for p in range(P)
    ])  # [P, N_HEAD*D, N]
    return blocks.reshape(P, n_head, d, n_tok)


def _run_proj_qkv_p4(module, app_class, D_acts: list[np.ndarray], W: np.ndarray,
                     tmp_path: Path, tag: str) -> list[np.ndarray]:
    """One proj_qkv_*_p4 invocation over all 4 streams.

    The harness takes all streams in ONE input file, ``(N_STREAM, N_TG, K,
    N_TOK)`` FP32, and writes ONE output file of raw XMEM rows,
    ``(N_STREAM, N_TG, N_OUT, LANES)``. Returns stream p's [N_OUT, LANES] rows.
    """
    input_path = tmp_path / f"d_{tag}.bin"
    input_path.write_bytes(np.stack(D_acts).astype(np.float32).tobytes())   # N_TG = 1
    weights_path = tmp_path / f"w_{tag}.bin"
    weights_path.write_bytes(W.astype(np.float32).tobytes())
    output_path = tmp_path / f"out_{tag}.bin"

    app = app_class(inst_path=kernel_inst(module.__package__.rpartition(".")[2]),
                    input_path=input_path, weights_path=weights_path, output_path=output_path)
    _, cycles = app.run(max_cycles=20_000_000)
    assert cycles > 0

    raw = output_path.read_bytes()
    assert len(raw) == module.N_STREAM * module.N_OUT * module.LANES * 4
    rows = np.frombuffer(raw, dtype=np.float32).reshape(module.N_STREAM, module.N_OUT, module.LANES)
    return [rows[p] for p in range(module.N_STREAM)]


def _run_scores_km(module, app_class, q_blocks, k_blocks, head, out_rows, tmp_path: Path, tag: str):
    q_path = tmp_path / f"q_{tag}.bin"
    k_path = tmp_path / f"k_{tag}.bin"
    q_path.write_bytes(q_blocks.astype(np.float32).tobytes())
    k_path.write_bytes(k_blocks.astype(np.float32).tobytes())
    output_path = tmp_path / f"scores_{tag}.bin"

    app = app_class(inst_path=kernel_inst(module.__package__.rpartition(".")[2]),
                    input_path=q_path, weights_path=k_path, output_path=output_path, head=head)
    state = app.make_state()
    poison(state, module.SBASE, module.S_ROWS)
    state, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0
    return np.frombuffer(output_path.read_bytes(), dtype=np.float32).reshape(out_rows, module.LANES)


def _l4_inputs(seed):
    rng = np.random.RandomState(seed)
    W = rng.uniform(-1.0, 1.0, size=(qkv4.N_OUT, qkv4.K)).astype(np.float32)
    D_acts = [rng.uniform(-1.0, 1.0, size=(qkv4.K, qkv4.N_TOK)).astype(np.float32)
              for _ in range(qkv4.N_STREAM)]
    return W, D_acts


def _l4_expected(W, D_acts, head, k_row0):
    expected = np.zeros((km4.P, km4.N, km4.N), dtype=np.float64)
    for p in range(km4.P):
        C = W.astype(np.float64) @ D_acts[p].astype(np.float64)
        Qh = C[head * km4.D:(head + 1) * km4.D, :]
        Kh = C[k_row0 + head * km4.D: k_row0 + (head + 1) * km4.D, :]
        expected[p] = np.einsum("ci,cs->is", Qh, Kh).T
    return expected.reshape(km4.P * km4.N, km4.N)


def test_repack_is_padding_strip_only_no_permutation() -> None:
    """Construct a stream_outputs input where element [p, h, c, t] is a
    UNIQUE value (not just per-stream-random), repack it, and verify every
    output element lands at the index its unique value encodes. Any
    accidental axis swap (h<->c, wrong p order, transposed t) would place a
    value at the wrong index and fail this exact-match check -- a numeric
    tolerance test could not catch that class of bug.
    """
    n_head, d, n_tok, P, n_out, lanes = 4, 48, 64, 4, 576, 128

    p_idx, h_idx, c_idx, t_idx = np.meshgrid(
        np.arange(P), np.arange(n_head), np.arange(d), np.arange(n_tok), indexing="ij")
    # Injective encoding into a plain float -- decodable by inspection.
    encoded = (p_idx * 10_000 + h_idx * 1_000 + c_idx * 10 + t_idx).astype(np.float32)

    stream_outputs = []
    for p in range(P):
        block = np.zeros((n_out, lanes), dtype=np.float32)
        # padding lanes stay 0 -- must never appear in valid output
        block[:n_head * d, :n_tok] = encoded[p].reshape(n_head * d, n_tok)
        stream_outputs.append(block)

    got = repack_qkv_to_km(stream_outputs, n_head=n_head, d=d, n_tok=n_tok, row0=0)
    assert got.shape == (P, n_head, d, n_tok)
    mismatch = np.argwhere(got != encoded)
    assert mismatch.size == 0, (
        f"repack misplaced element(s) at [p,h,c,t] = {mismatch[:5].tolist()} -- "
        "this indicates a permutation, not just a padding-strip"
    )


def test_repack_control_mismatch_is_detected(tmp_path: Path) -> None:
    """Mutation-first control: corrupt one repacked element and confirm the
    downstream score kernel's result diverges from the reference -- proves
    the full-pipeline numeric test below is actually sensitive to the repack
    being wrong, not passing vacuously.
    """
    W, D_acts = _l4_inputs(0xBAD5EED)
    stream_outputs = _run_proj_qkv_p4(qkv4, qkv4.ProjQkv192P4App, D_acts, W, tmp_path, tag="ctrl")

    head = 2
    k_row0 = km4.N_HEAD * km4.D
    q_blocks = repack_qkv_to_km(stream_outputs, n_head=km4.N_HEAD, d=km4.D, n_tok=km4.N, row0=_Q_ROW0)
    k_blocks = repack_qkv_to_km(stream_outputs, n_head=km4.N_HEAD, d=km4.D, n_tok=km4.N, row0=k_row0)

    # Corrupt one element of Q for the selected head.
    q_blocks[1, head, 3, :] = 999.0

    rows = _run_scores_km(km4, km4.AttnScoresKM64x48App, q_blocks, k_blocks, head,
                          km4.P * km4.N, tmp_path, tag="ctrl")
    got = rows[:, :km4.N]

    max_err = float(np.max(np.abs(got.astype(np.float64) - _l4_expected(W, D_acts, head, k_row0))))
    assert max_err > 1.0, (
        f"corrupted-repack control did not diverge (max_err={max_err:.3e}) -- "
        "the full-pipeline test below is not actually sensitive to a bad repack"
    )


def test_proj_qkv_192_p4_to_scores_km_64x48_agrees_l4(tmp_path: Path) -> None:
    """Real pipeline, L4: ONE proj_qkv_192_p4 invocation (all 4 streams) ->
    repack -> attn_scores_km_64x48, poisoned destination, checked against an
    independent reference computed straight from W/D_act.
    """
    W, D_acts = _l4_inputs(0xF00D4)
    stream_outputs = _run_proj_qkv_p4(qkv4, qkv4.ProjQkv192P4App, D_acts, W, tmp_path, tag="clean")

    head = 1
    k_row0 = km4.N_HEAD * km4.D
    q_blocks = repack_qkv_to_km(stream_outputs, n_head=km4.N_HEAD, d=km4.D, n_tok=km4.N, row0=_Q_ROW0)
    k_blocks = repack_qkv_to_km(stream_outputs, n_head=km4.N_HEAD, d=km4.D, n_tok=km4.N, row0=k_row0)

    rows = _run_scores_km(km4, km4.AttnScoresKM64x48App, q_blocks, k_blocks, head,
                          km4.P * km4.N, tmp_path, tag="clean")
    got = rows[:, :km4.N]
    expected = _l4_expected(W, D_acts, head, k_row0)

    max_err = float(np.max(np.abs(got.astype(np.float64) - expected)))
    print(f"seam proj_qkv_192_p4(repacked)->attn_scores_km_64x48 max abs error = {max_err:.3e}")

    np.testing.assert_allclose(
        got, expected, rtol=2e-3, atol=2e-2,
        err_msg="proj_qkv_192_p4 -> repack -> attn_scores_km_64x48 (L4) mismatch",
    )


def _repack_one_stream_to_km5_file(stream_output: np.ndarray, *, row0: int) -> np.ndarray:
    """attn_scores_km_16x60's canonical file layout is [N_HEADS*D, N_TOK]
    per-head-channel-major (channel (h*D+c) at element (h*D+c)*N_TOK+t) --
    NOT the [P, N_HEAD, D, N] shape attn_scores_km_64x48 uses. Padding-strip
    only: take one stream's [N_HEADS*D, LANES] block starting at row0 and
    crop each row to its first N_TOK lanes -- row order (== head*D+c order)
    is preserved exactly.
    """
    return stream_output[row0:row0 + km5.N_HEADS * km5.D, :km5.N_TOK]  # [N_HEADS*D, N_TOK]


def test_proj_qkv_240_p4_to_scores_km_16x60_agrees_l5(tmp_path: Path) -> None:
    """Same idea as the L4 test above, at L5 (proj_qkv_240_p4 ->
    attn_scores_km_16x60). Unlike L4's score kernel, attn_scores_km_16x60
    takes ONE stream's Q/K per call (no P axis in its own file contract), so
    this loops P=4 real score-kernel invocations, one per stream, each fed
    that stream's repacked (padding-stripped only) Q/K block.
    """
    rng = np.random.RandomState(0xF00D5)
    W = rng.uniform(-1.0, 1.0, size=(qkv5.N_OUT, qkv5.K)).astype(np.float32)
    D_acts = [rng.uniform(-1.0, 1.0, size=(qkv5.K, qkv5.N_TOK)).astype(np.float32)
              for _ in range(qkv5.N_STREAM)]

    stream_outputs = _run_proj_qkv_p4(qkv5, qkv5.ProjQKV240P4App, D_acts, W, tmp_path, tag="clean5")

    head = 0
    k_row0 = km5.N_HEADS * km5.D

    for p in range(qkv5.N_STREAM):
        q_block = _repack_one_stream_to_km5_file(stream_outputs[p], row0=_Q_ROW0)
        k_block = _repack_one_stream_to_km5_file(stream_outputs[p], row0=k_row0)
        rows = _run_scores_km(km5, km5.AttnScoresKM16x60App, q_block, k_block, head,
                              km5.N_TOK, tmp_path, tag=f"clean5_p{p}")
        got = rows[:, :km5.N_TOK]

        C = W.astype(np.float64) @ D_acts[p].astype(np.float64)
        Qh = C[head * km5.D:(head + 1) * km5.D, :]
        Kh = C[k_row0 + head * km5.D: k_row0 + (head + 1) * km5.D, :]
        expected = np.einsum("ci,cs->is", Qh, Kh).T   # key-major

        max_err = float(np.max(np.abs(got.astype(np.float64) - expected)))
        print(f"seam proj_qkv_240_p4(repacked, stream {p})->attn_scores_km_16x60 "
              f"max abs error = {max_err:.3e}")

        np.testing.assert_allclose(
            got, expected, rtol=2e-3, atol=2e-2,
            err_msg=f"proj_qkv_240_p4 -> repack -> attn_scores_km_16x60 (L5, stream {p}) mismatch",
        )
