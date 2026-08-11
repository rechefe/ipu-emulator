"""Seam test: attn_scores_km_64x48 -> attn_v_bcast_48 (Layer 4, key-major chain).

Nothing today runs these two kernels back to back: each is tested in isolation
against its own hand-staged input. This test wires the REAL producer's raw
XMEM store output into the REAL consumer's input, byte for byte, to catch
pitch/base/extent/ordering mismatches that per-kernel tests cannot see.

attn_scores_km_64x48 scores ONE head across all P=4 streams per run (256 key
rows). attn_v_bcast_48 wants all N_BLOCK=16 (stream, head) blocks in one P
file addressed as row (b * N_TOK) with b = p * N_HEAD + h. So the chain
requires N_HEAD=4 producer runs (h=0..3), each contributing 4 blocks
(p=0..3, at fixed h) that are NOT contiguous in the consumer's b ordering --
this is exactly the shape of index arithmetic bug this audit is checking for.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.attn_scores_km_64x48 import (
    AttnScoresKM64x48App, N as KM_N, D as KM_D, P as KM_P, N_HEAD, LANES,
    SBASE, S_ROWS, OUTPUT_ROW_BYTES as KM_OUTPUT_ROW_BYTES,
)
from ipu_apps.attn_v_bcast_48 import (
    AttnVBcast48App, N_TOK, D, N_BLOCK, N_CHAN,
)

_KM_INST_BIN = Path(os.environ["ATTN_SCORES_KM_64X48_INST_BIN"])
_BCAST_INST_BIN = Path(os.environ["ATTN_V_BCAST_48_INST_BIN"])

_POISON = 1e3


def _run_km_producer(q: np.ndarray, k: np.ndarray, head: int, tmp_path: Path) -> bytes:
    """Run the real attn_scores_km_64x48 kernel for one head, with its whole
    output region poisoned first. Returns the raw output file bytes
    (P * N rows of ROW_BYTES, uncropped -- exactly what teardown wrote)."""
    q_path = tmp_path / f"q_{head}.bin"
    k_path = tmp_path / f"k_{head}.bin"
    q_path.write_bytes(q.tobytes())
    k_path.write_bytes(k.tobytes())
    output_path = tmp_path / f"km_out_{head}.bin"

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    # Poison the producer's entire output region BEFORE it runs, so any row
    # the kernel fails to write shows up as 1e3 in the output file.
    state.xmem.write_address(
        SBASE, bytearray(np.full(S_ROWS * LANES, _POISON, dtype=np.float32).tobytes())
    )

    app = AttnScoresKM64x48App(
        inst_path=_KM_INST_BIN,
        input_path=q_path,
        weights_path=k_path,
        output_path=output_path,
        head=head,
    )
    state, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    raw = output_path.read_bytes()
    assert len(raw) == KM_P * KM_N * KM_OUTPUT_ROW_BYTES, (
        f"producer output size {len(raw)} != expected "
        f"{KM_P * KM_N * KM_OUTPUT_ROW_BYTES}"
    )
    # No row should still read as pure poison -- every row must have been
    # written by the kernel's own store path.
    rows = np.frombuffer(raw, dtype=np.float32).reshape(KM_P * KM_N, LANES)
    all_poison_rows = np.all(rows == _POISON, axis=1)
    assert not all_poison_rows.any(), (
        f"producer left {all_poison_rows.sum()} whole rows as untouched poison "
        f"(row indices {np.nonzero(all_poison_rows)[0].tolist()})"
    )
    return raw


def _build_p_file(head_outputs: dict[int, bytes]) -> bytes:
    """Assemble attn_v_bcast_48's p_path from 4 single-head producer runs.

    Consumer wants block b = p*N_HEAD + h at row range [b*N_TOK, (b+1)*N_TOK).
    Producer run h emits row (p*KM_N + s) = K[p,s,:] for p in [0, KM_P), i.e.
    blocks (p, h) for p=0..3 -- NOT block-contiguous, so blocks must be
    interleaved by p on the way in.
    """
    assert KM_N == N_TOK and KM_D == D and KM_P == 4 and N_HEAD == 4
    out = bytearray(N_BLOCK * N_TOK * LANES * 4)
    for h, raw in head_outputs.items():
        rows = np.frombuffer(raw, dtype=np.float32).reshape(KM_P, KM_N, LANES)
        for p in range(KM_P):
            b = p * N_HEAD + h
            lo = b * N_TOK * LANES * 4
            hi = lo + N_TOK * LANES * 4
            out[lo:hi] = rows[p].tobytes()
    return bytes(out)


def test_seam_km_bcast_48_agrees(tmp_path: Path) -> None:
    """Full chain: 4 attn_scores_km_64x48 runs (one per head) feed one
    attn_v_bcast_48 run. Destination poisoned before the producer runs.
    Compares against a from-scratch two-stage numpy reference."""
    rng = np.random.RandomState(0xC48)

    Q = rng.uniform(-1.0, 1.0, size=(KM_P, N_HEAD, KM_D, KM_N)).astype(np.float32)
    K = rng.uniform(-1.0, 1.0, size=(KM_P, N_HEAD, KM_D, KM_N)).astype(np.float32)
    V = rng.uniform(-1.0, 1.0, size=(N_BLOCK, D, N_TOK)).astype(np.float32)

    head_outputs = {
        h: _run_km_producer(Q, K, h, tmp_path) for h in range(N_HEAD)
    }
    p_bytes = _build_p_file(head_outputs)

    # V is staged as attn_v_bcast_48 expects it directly (V has no producer
    # kernel in this repo -- it is host-staged input in every existing test).
    v_buf = np.zeros((N_CHAN, LANES), dtype=np.float32)
    for b in range(N_BLOCK):
        v_buf[b * D:(b + 1) * D, :N_TOK] = V[b]

    p_path = tmp_path / "p_chained.bin"
    v_path = tmp_path / "v.bin"
    p_path.write_bytes(p_bytes)
    v_path.write_bytes(v_buf.tobytes())
    output_path = tmp_path / "bcast_out.bin"

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    app = AttnVBcast48App(
        inst_path=_BCAST_INST_BIN,
        p_path=p_path,
        v_path=v_path,
        output_path=output_path,
    )
    state, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    got = np.frombuffer(output_path.read_bytes(), dtype=np.float32).reshape(
        N_BLOCK, D, N_TOK
    )

    # Two-stage reference computed independently from Q, K, V (not from either
    # kernel's own isolated golden): S[p,h] = Q[p,h].T @ K[p,h] (key-major),
    # then O[b,t,i] = sum_s S[b][s,i] * V[b,t,s].
    expected = np.zeros((N_BLOCK, D, N_TOK), dtype=np.float64)
    for p in range(KM_P):
        for h in range(N_HEAD):
            b = p * N_HEAD + h
            # S[key, query] = sum_c Q[p,h,c,query] * K[p,h,c,key]
            S = np.einsum("cq,ck->kq", Q[p, h].astype(np.float64), K[p, h].astype(np.float64))
            # O[t, query] = sum_key S[key, query] * V[b, t, key]
            expected[b] = np.einsum("kq,tk->tq", S, V[b].astype(np.float64))

    max_err = float(np.max(np.abs(got.astype(np.float64) - expected)))
    print(f"seam km_64x48->bcast_48 max abs error = {max_err:.3e}")

    np.testing.assert_allclose(
        got, expected, rtol=2e-3, atol=2e-2,
        err_msg=(
            "chained attn_scores_km_64x48 -> attn_v_bcast_48 mismatch vs "
            "independent two-stage numpy reference"
        ),
    )


def test_seam_km_bcast_48_row_pitch_mismatch_is_caught(tmp_path: Path) -> None:
    """Mutation check: if the producer's row pitch were narrower than a whole
    512 B row (the bug class this audit exists to catch), the seam test must
    fail. Simulates that bug directly on already-captured producer output by
    re-packing it at the old, wrong 256 B (N*ELEM_BYTES) pitch, then staging
    that into attn_v_bcast_48 exactly as it stages p_path (verbatim, one key
    per whole 512 B row) and confirming the result disagrees with the
    reference by more than a rounding-level tolerance.
    """
    rng = np.random.RandomState(0xC49)
    ELEM_BYTES = 4

    Q = rng.uniform(-1.0, 1.0, size=(KM_P, N_HEAD, KM_D, KM_N)).astype(np.float32)
    K = rng.uniform(-1.0, 1.0, size=(KM_P, N_HEAD, KM_D, KM_N)).astype(np.float32)
    V = rng.uniform(-1.0, 1.0, size=(N_BLOCK, D, N_TOK)).astype(np.float32)

    head_outputs = {h: _run_km_producer(Q, K, h, tmp_path) for h in range(N_HEAD)}

    # Correct chain (full 512 B rows) -- build once as a baseline.
    p_bytes_correct = _build_p_file(head_outputs)

    # Deliberately mis-pitch: re-encode each producer row at the OLD buggy
    # 256 B (N * ELEM_BYTES) stride instead of the real 512 B row stride, then
    # stage the result exactly as attn_v_bcast_48 expects (one key per whole
    # 512 B row) -- i.e. bytes drift out of alignment after the first row,
    # exactly like the original km_64x48/bcast_48 bug.
    mispitched = bytearray(N_BLOCK * N_TOK * LANES * ELEM_BYTES)
    cursor = 0
    for h, raw in head_outputs.items():
        rows = np.frombuffer(raw, dtype=np.float32).reshape(KM_P, KM_N, LANES)
        for p in range(KM_P):
            for s in range(KM_N):
                row_bytes = rows[p, s, :KM_N].tobytes()  # only N*ELEM_BYTES = 256 B
                mispitched[cursor: cursor + len(row_bytes)] = row_bytes
                cursor += len(row_bytes)
    # Pad remainder (mispitched is shorter than the correct buffer).
    mispitched = bytes(mispitched[:cursor]) + bytes(
        len(p_bytes_correct) - cursor
    )

    v_buf = np.zeros((N_CHAN, LANES), dtype=np.float32)
    for b in range(N_BLOCK):
        v_buf[b * D:(b + 1) * D, :N_TOK] = V[b]
    v_path = tmp_path / "v_mut.bin"
    v_path.write_bytes(v_buf.tobytes())

    def run_bcast(p_bytes: bytes, tag: str) -> np.ndarray:
        p_path = tmp_path / f"p_{tag}.bin"
        p_path.write_bytes(p_bytes)
        output_path = tmp_path / f"out_{tag}.bin"
        state = IpuState(
            wide_vector_debug=True,
            wide_vector_arithmetic=WideVectorArithmetic.FP32,
        )
        app = AttnVBcast48App(
            inst_path=_BCAST_INST_BIN,
            p_path=p_path,
            v_path=v_path,
            output_path=output_path,
        )
        app.run(max_cycles=20_000_000, state=state)
        return np.frombuffer(output_path.read_bytes(), dtype=np.float32).reshape(
            N_BLOCK, D, N_TOK
        )

    got_correct = run_bcast(p_bytes_correct, "correct")
    got_mispitched = run_bcast(mispitched, "mispitched")

    # The mutation must actually change the result -- if it doesn't, this
    # "seam test" has no teeth.
    max_diff = float(np.max(np.abs(got_correct - got_mispitched)))
    assert max_diff > 1e-2, (
        "mutating the producer's row pitch to the old buggy 256 B stride did "
        f"not change attn_v_bcast_48's output (max diff {max_diff:.3e}) -- "
        "this seam test would not have caught the original bug"
    )
