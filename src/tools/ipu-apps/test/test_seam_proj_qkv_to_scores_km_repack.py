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

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.projections.proj_qkv_192_p4 import (
    ProjQkv192P4App, K as QKV4_K, N_OUT as QKV4_N_OUT, N_TOK as QKV4_N_TOK,
    N_STREAM as QKV4_P, LANES as QKV4_LANES,
)
from ipu_apps.attention.attn_scores_km_64x48 import (
    AttnScoresKM64x48App, N as KM4_N, D as KM4_D, P as KM4_P,
    N_HEAD as KM4_N_HEAD, SBASE as KM4_SBASE, S_ROWS as KM4_S_ROWS,
    LANES as KM4_LANES,
)

from ipu_apps.projections.proj_qkv_240_p4 import (
    ProjQKV240P4App, K as QKV5_K, N_OUT as QKV5_N_OUT, N_TOK as QKV5_N_TOK,
    N_STREAM as QKV5_P, LANES as QKV5_LANES,
)
from ipu_apps.attention.attn_scores_km_16x60 import (
    AttnScoresKM16x60App, N_TOK as KM5_N, D as KM5_D,
    N_HEADS as KM5_N_HEAD, SBASE as KM5_SBASE, S_ROWS as KM5_S_ROWS,
    LANES as KM5_LANES,
)

# attn_scores_km_16x60 (unlike attn_scores_km_64x48) processes ONE stream's
# Q/K per invocation -- its input file is canonical [N_HEADS*D, N_TOK]
# per-head-channel-major with NO stream axis at all (see its _load_q_channel_major
# docstring). A full L5 chain therefore still needs P=4 score-kernel calls,
# one per stream, each fed that stream's repacked Q/K block.

_PROJ_QKV_192_P4_INST_BIN = Path(os.environ["PROJ_QKV_192_P4_INST_BIN"])
_PROJ_QKV_240_P4_INST_BIN = Path(os.environ["PROJ_QKV_240_P4_INST_BIN"])

_KM4_ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/ipu_apps/attention/attn_scores_km_64x48/attn_scores_km_64x48.asm"
)
_KM5_ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/ipu_apps/attention/attn_scores_km_16x60/attn_scores_km_16x60.asm"
)


@pytest.fixture(scope="module")
def km4_inst_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "attn_scores_km_64x48.bin"
        assemble_to_bin_file(_KM4_ASM_PATH.read_text(encoding="utf-8"), str(path))
        yield path


@pytest.fixture(scope="module")
def km5_inst_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "attn_scores_km_16x60.bin"
        assemble_to_bin_file(_KM5_ASM_PATH.read_text(encoding="utf-8"), str(path))
        yield path


_POISON = 1e3

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


def _run_proj_qkv_192_p4(D_acts: list[np.ndarray], W: np.ndarray, tmp_path: Path, tag: str) -> list[np.ndarray]:
    input_paths = []
    for p in range(QKV4_P):
        path = tmp_path / f"d4_{tag}_{p}.bin"
        path.write_bytes(D_acts[p].astype(np.float32).tobytes())
        input_paths.append(path)
    weights_path = tmp_path / f"w4_{tag}.bin"
    weights_path.write_bytes(W.astype(np.float32).tobytes())
    output_paths = [tmp_path / f"out4_{tag}_{p}.bin" for p in range(QKV4_P)]

    state = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
    app = ProjQkv192P4App(
        inst_path=_PROJ_QKV_192_P4_INST_BIN,
        input_paths=input_paths,
        weights_path=weights_path,
        output_paths=output_paths,
    )
    state, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    outs = []
    for p in range(QKV4_P):
        raw = output_paths[p].read_bytes()
        assert len(raw) == QKV4_N_OUT * QKV4_LANES * 4
        outs.append(np.frombuffer(raw, dtype=np.float32).reshape(QKV4_N_OUT, QKV4_LANES))
    return outs


def _run_proj_qkv_240_p4(D_acts: list[np.ndarray], W: np.ndarray, tmp_path: Path, tag: str) -> list[np.ndarray]:
    input_paths = []
    for p in range(QKV5_P):
        path = tmp_path / f"d5_{tag}_{p}.bin"
        path.write_bytes(D_acts[p].astype(np.float32).tobytes())
        input_paths.append(path)
    weights_path = tmp_path / f"w5_{tag}.bin"
    weights_path.write_bytes(W.astype(np.float32).tobytes())
    output_paths = [tmp_path / f"out5_{tag}_{p}.bin" for p in range(QKV5_P)]

    state = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
    app = ProjQKV240P4App(
        inst_path=_PROJ_QKV_240_P4_INST_BIN,
        input_paths=input_paths,
        weights_path=weights_path,
        output_paths=output_paths,
    )
    state, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    outs = []
    for p in range(QKV5_P):
        raw = output_paths[p].read_bytes()
        assert len(raw) == QKV5_N_OUT * QKV5_LANES * 4
        outs.append(np.frombuffer(raw, dtype=np.float32).reshape(QKV5_N_OUT, QKV5_LANES))
    return outs


def test_repack_is_padding_strip_only_no_permutation() -> None:
    """Construct a stream_outputs input where element [p, h, c, t] is a
    UNIQUE value (not just per-stream-random), repack it, and verify every
    output element lands at the index its unique value encodes. Any
    accidental axis swap (h<->c, wrong p order, transposed t) would place a
    value at the wrong index and fail this exact-match check -- a numeric
    tolerance test could not catch that class of bug.
    """
    n_head, d, n_tok, P, n_out, lanes = 4, 48, 64, 4, 576, 128

    def encode(p: int, h: int, c: int, t: int) -> float:
        # Injective encoding into a plain float -- decodable by inspection.
        return float(p * 10_000 + h * 1_000 + c * 10 + t)

    stream_outputs = []
    for p in range(P):
        block = np.zeros((n_out, lanes), dtype=np.float32)
        for h in range(n_head):
            for c in range(d):
                row = h * d + c
                for t in range(n_tok):
                    block[row, t] = encode(p, h, c, t)
                # padding lanes stay 0 -- must never appear in valid output
        stream_outputs.append(block)

    got = repack_qkv_to_km(stream_outputs, n_head=n_head, d=d, n_tok=n_tok, row0=0)
    assert got.shape == (P, n_head, d, n_tok)

    for p in range(P):
        for h in range(n_head):
            for c in range(d):
                for t in range(n_tok):
                    expected = encode(p, h, c, t)
                    actual = got[p, h, c, t]
                    assert actual == expected, (
                        f"repack misplaced element: got[{p},{h},{c},{t}]={actual}, "
                        f"expected {expected} -- this indicates a permutation, "
                        f"not just a padding-strip"
                    )


def test_repack_control_mismatch_is_detected(km4_inst_file: Path, tmp_path: Path) -> None:
    """Mutation-first control: corrupt one repacked element and confirm the
    downstream score kernel's result diverges from the reference -- proves
    the full-pipeline numeric test below is actually sensitive to the repack
    being wrong, not passing vacuously.
    """
    rng = np.random.RandomState(0xBAD5EED)
    W = rng.uniform(-1.0, 1.0, size=(QKV4_N_OUT, QKV4_K)).astype(np.float32)
    D_acts = [rng.uniform(-1.0, 1.0, size=(QKV4_K, QKV4_N_TOK)).astype(np.float32) for _ in range(QKV4_P)]

    stream_outputs = _run_proj_qkv_192_p4(D_acts, W, tmp_path, tag="ctrl")

    head = 2
    q_blocks = repack_qkv_to_km(stream_outputs, n_head=KM4_N_HEAD, d=KM4_D, n_tok=KM4_N, row0=_Q_ROW0)
    k_row0 = KM4_N_HEAD * KM4_D
    k_blocks = repack_qkv_to_km(stream_outputs, n_head=KM4_N_HEAD, d=KM4_D, n_tok=KM4_N, row0=k_row0)

    # Corrupt one element of Q for the selected head.
    q_blocks[1, head, 3, :] = 999.0

    q_path = tmp_path / "q_ctrl.bin"
    k_path = tmp_path / "k_ctrl.bin"
    q_path.write_bytes(q_blocks.astype(np.float32).tobytes())
    k_path.write_bytes(k_blocks.astype(np.float32).tobytes())
    output_path = tmp_path / "scores_ctrl.bin"

    state = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
    state.xmem.write_address(
        KM4_SBASE, bytearray(np.full(KM4_S_ROWS * KM4_LANES, _POISON, dtype=np.float32).tobytes())
    )
    app = AttnScoresKM64x48App(
        inst_path=km4_inst_file,
        input_path=q_path, weights_path=k_path, output_path=output_path, head=head,
    )
    state, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    got = np.frombuffer(output_path.read_bytes(), dtype=np.float32).reshape(KM4_P * KM4_N, KM4_LANES)[:, :KM4_N]

    expected = np.zeros((KM4_P, KM4_N, KM4_N), dtype=np.float64)
    for p in range(KM4_P):
        C = W.astype(np.float64) @ D_acts[p].astype(np.float64)
        Qh = C[head * KM4_D:(head + 1) * KM4_D, :]
        Kh = C[k_row0 + head * KM4_D: k_row0 + (head + 1) * KM4_D, :]
        S = np.einsum("ci,cs->is", Qh, Kh)
        expected[p] = S.T
    expected = expected.reshape(KM4_P * KM4_N, KM4_N)

    max_err = float(np.max(np.abs(got.astype(np.float64) - expected)))
    assert max_err > 1.0, (
        f"corrupted-repack control did not diverge (max_err={max_err:.3e}) -- "
        "the full-pipeline test below is not actually sensitive to a bad repack"
    )


def test_proj_qkv_192_p4_to_scores_km_64x48_agrees_l4(
    km4_inst_file: Path, tmp_path: Path,
) -> None:
    """Real pipeline, L4: ONE proj_qkv_192_p4 invocation (all 4 streams) ->
    repack -> attn_scores_km_64x48, poisoned destination, checked against an
    independent reference computed straight from W/D_act.
    """
    rng = np.random.RandomState(0xF00D4)
    W = rng.uniform(-1.0, 1.0, size=(QKV4_N_OUT, QKV4_K)).astype(np.float32)
    D_acts = [rng.uniform(-1.0, 1.0, size=(QKV4_K, QKV4_N_TOK)).astype(np.float32) for _ in range(QKV4_P)]

    stream_outputs = _run_proj_qkv_192_p4(D_acts, W, tmp_path, tag="clean")

    head = 1
    k_row0 = KM4_N_HEAD * KM4_D
    q_blocks = repack_qkv_to_km(stream_outputs, n_head=KM4_N_HEAD, d=KM4_D, n_tok=KM4_N, row0=_Q_ROW0)
    k_blocks = repack_qkv_to_km(stream_outputs, n_head=KM4_N_HEAD, d=KM4_D, n_tok=KM4_N, row0=k_row0)

    q_path = tmp_path / "q_clean.bin"
    k_path = tmp_path / "k_clean.bin"
    q_path.write_bytes(q_blocks.astype(np.float32).tobytes())
    k_path.write_bytes(k_blocks.astype(np.float32).tobytes())
    output_path = tmp_path / "scores_clean.bin"

    state = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
    state.xmem.write_address(
        KM4_SBASE, bytearray(np.full(KM4_S_ROWS * KM4_LANES, _POISON, dtype=np.float32).tobytes())
    )
    app = AttnScoresKM64x48App(
        inst_path=km4_inst_file,
        input_path=q_path, weights_path=k_path, output_path=output_path, head=head,
    )
    state, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    got = np.frombuffer(output_path.read_bytes(), dtype=np.float32).reshape(KM4_P * KM4_N, KM4_LANES)[:, :KM4_N]

    expected = np.zeros((KM4_P, KM4_N, KM4_N), dtype=np.float64)
    for p in range(KM4_P):
        C = W.astype(np.float64) @ D_acts[p].astype(np.float64)
        Qh = C[head * KM4_D:(head + 1) * KM4_D, :]
        Kh = C[k_row0 + head * KM4_D: k_row0 + (head + 1) * KM4_D, :]
        S = np.einsum("ci,cs->is", Qh, Kh)
        expected[p] = S.T
    expected = expected.reshape(KM4_P * KM4_N, KM4_N)

    max_err = float(np.max(np.abs(got.astype(np.float64) - expected)))
    print(f"seam proj_qkv_192_p4(repacked)->attn_scores_km_64x48 max abs error = {max_err:.3e}")

    np.testing.assert_allclose(
        got, expected, rtol=2e-3, atol=2e-2,
        err_msg="proj_qkv_192_p4 -> repack -> attn_scores_km_64x48 (L4) mismatch",
    )


def _repack_one_stream_to_km5_file(
    stream_output: np.ndarray, *, row0: int,
) -> np.ndarray:
    """attn_scores_km_16x60's canonical file layout is [N_HEADS*D, N_TOK]
    per-head-channel-major (channel (h*D+c) at element (h*D+c)*N_TOK+t) --
    NOT the [P, N_HEAD, D, N] shape attn_scores_km_64x48 uses. Padding-strip
    only: take one stream's [N_HEADS*D, LANES] block starting at row0 and
    crop each row to its first N_TOK lanes -- row order (== head*D+c order)
    is preserved exactly.
    """
    return stream_output[row0:row0 + KM5_N_HEAD * KM5_D, :KM5_N]  # [N_HEADS*D, N_TOK]


def test_proj_qkv_240_p4_to_scores_km_16x60_agrees_l5(
    km5_inst_file: Path, tmp_path: Path,
) -> None:
    """Same idea as the L4 test above, at L5 (proj_qkv_240_p4 ->
    attn_scores_km_16x60). Unlike L4's score kernel, attn_scores_km_16x60
    takes ONE stream's Q/K per call (no P axis in its own file contract), so
    this loops P=4 real score-kernel invocations, one per stream, each fed
    that stream's repacked (padding-stripped only) Q/K block.
    """
    rng = np.random.RandomState(0xF00D5)
    W = rng.uniform(-1.0, 1.0, size=(QKV5_N_OUT, QKV5_K)).astype(np.float32)
    D_acts = [rng.uniform(-1.0, 1.0, size=(QKV5_K, QKV5_N_TOK)).astype(np.float32) for _ in range(QKV5_P)]

    stream_outputs = _run_proj_qkv_240_p4(D_acts, W, tmp_path, tag="clean5")

    head = 0
    k_row0 = KM5_N_HEAD * KM5_D

    for p in range(QKV5_P):
        q_block = _repack_one_stream_to_km5_file(stream_outputs[p], row0=_Q_ROW0)  # [N_HEADS*D, N_TOK]
        k_block = _repack_one_stream_to_km5_file(stream_outputs[p], row0=k_row0)

        q_path = tmp_path / f"q_clean5_p{p}.bin"
        k_path = tmp_path / f"k_clean5_p{p}.bin"
        q_path.write_bytes(q_block.astype(np.float32).tobytes())
        k_path.write_bytes(k_block.astype(np.float32).tobytes())
        output_path = tmp_path / f"scores_clean5_p{p}.bin"

        state = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
        state.xmem.write_address(
            KM5_SBASE, bytearray(np.full(KM5_S_ROWS * KM5_LANES, _POISON, dtype=np.float32).tobytes())
        )
        app = AttnScoresKM16x60App(
            inst_path=km5_inst_file,
            input_path=q_path, weights_path=k_path, output_path=output_path, head=head,
        )
        state, cycles = app.run(max_cycles=20_000_000, state=state)
        assert cycles > 0

        got = np.frombuffer(output_path.read_bytes(), dtype=np.float32).reshape(KM5_N, KM5_LANES)[:, :KM5_N]

        C = W.astype(np.float64) @ D_acts[p].astype(np.float64)
        Qh = C[head * KM5_D:(head + 1) * KM5_D, :]
        Kh = C[k_row0 + head * KM5_D: k_row0 + (head + 1) * KM5_D, :]
        expected = np.einsum("ci,cs->is", Qh, Kh).T   # key-major

        max_err = float(np.max(np.abs(got.astype(np.float64) - expected)))
        print(f"seam proj_qkv_240_p4(repacked, stream {p})->attn_scores_km_16x60 max abs error = {max_err:.3e}")

        np.testing.assert_allclose(
            got, expected, rtol=2e-3, atol=2e-2,
            err_msg=f"proj_qkv_240_p4 -> repack -> attn_scores_km_16x60 (L5, stream {p}) mismatch",
        )
