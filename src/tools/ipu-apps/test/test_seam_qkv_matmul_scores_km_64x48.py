"""Seam investigation: matmul_576x192_x128 (QKV projection) -> attn_scores_km_64x48.

Question: at L4, matmul_576x192_x128 produces 576 output channels (Q's 192,
then K's 192, then V's 192, each 192 = 4 heads x 48) for ONE stream per run.
attn_scores_km_64x48 wants Q/K staged as flat files shaped
[P=4, N_HEAD=4, D=48, N=64] (channel-major per (stream, head) block, no XMEM
row padding -- see attn_scores_km_64x48._read_head).

This test runs the REAL matmul kernel once per stream (P=4 streams, sharing
one weight matrix -- "the projection" -- but with independent per-stream
activations), poisons the destination region each time, and checks whether
the resulting XMEM/output-file bytes can feed attn_scores_km_64x48 through
PURE ADDRESSING: base+stride slicing of the matmul's own raw output, with NO
value transform (no transpose, no numeric rewrite) -- only byte-range
selection and concatenation by (stream, head) index.

Finding under test: matmul_576x192_x128 stores one output channel per WHOLE
512 B XMEM row (64 valid tokens + 64 zero-padding lanes, per the
one-channel-per-row convention -- see matmul_576x192_x128/__init__.py
OUTPUT_ROW_BYTES=512). attn_scores_km_64x48's own input-file contract
(_read_head) is a TIGHTLY PACKED flat array with no row padding at all: N=64
FP32 elements per (block, channel), not 128. So even though the row ORDERING
already matches exactly (channel c of head h sits at output row h*48+c,
exactly the [N_HEAD, D] block order attn_scores_km_64x48 expects), the STORE
PITCH differs (512 B/channel vs 256 B/channel) and bytes must be re-packed
(padding stripped) before the score kernel's own setup() will parse the file
correctly. This is a fixed, mechanical strip-only operation (slice each row's
first N_TOK*4 bytes, drop the rest, concatenate) -- not a data-value
transform and not a reshape/transpose -- but it is not zero data movement
either, so it does not clear the "pure addressing, no copying code" bar the
seam audit sets.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.matmuls.matmul_576x192_x128 import (
    MatMul576x192x128App, K as MM_K, N_OUT, N_TOK as MM_N_TOK, LANES as MM_LANES,
    OUTPUT_BASE, OUTPUT_ROW_BYTES,
)
from ipu_apps.attention.attn_scores_km_64x48 import (
    AttnScoresKM64x48App, N, D, P, N_HEAD, N_BLOCK, LANES, SBASE, S_ROWS,
)

_MATMUL_INST_BIN = Path(os.environ["MATMUL_576X192_X128_INST_BIN"])

_KM_ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/ipu_apps/attention/attn_scores_km_64x48/attn_scores_km_64x48.asm"
)


@pytest.fixture(scope="module")
def km_inst_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "attn_scores_km_64x48.bin"
        assemble_to_bin_file(_KM_ASM_PATH.read_text(encoding="utf-8"), str(path))
        yield path


_POISON = 1e3

# Q/K/V block boundaries within the 576 QKV-matmul output rows.
_Q_ROW0, _K_ROW0, _V_ROW0 = 0, N_HEAD * D, 2 * N_HEAD * D  # 0, 192, 384
assert _K_ROW0 == 192 and _V_ROW0 == 384 and 3 * N_HEAD * D == N_OUT


def _run_qkv_matmul_one_stream(
    D_act: np.ndarray, W: np.ndarray, tmp_path: Path, tag: str
) -> bytes:
    """Run the real matmul_576x192_x128 kernel for one stream's activations.

    D_act: [K=192, N_TOK=64] channel-major per-stream activation input.
    W: [N_OUT=576, K=192] shared QKV projection weight (same across streams).
    Returns the raw output file bytes: N_OUT=576 rows of 512 B, uncropped --
    exactly what teardown wrote (one output channel per whole row).
    """
    data_path = tmp_path / f"d_{tag}.bin"
    weights_path = tmp_path / f"w_{tag}.bin"
    data_path.write_bytes(D_act.astype(np.float32).tobytes())
    weights_path.write_bytes(W.astype(np.float32).tobytes())
    output_path = tmp_path / f"mm_out_{tag}.bin"

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    # Poison the producer's entire output region before it runs, so any row
    # the kernel fails to write shows up as 1e3 in the output file.
    state.xmem.write_address(
        OUTPUT_BASE,
        bytearray(np.full(N_OUT * MM_LANES, _POISON, dtype=np.float32).tobytes()),
    )

    app = MatMul576x192x128App(
        inst_path=_MATMUL_INST_BIN,
        input_path=data_path,
        weights_path=weights_path,
        output_path=output_path,
    )
    state, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    raw = output_path.read_bytes()
    assert len(raw) == N_OUT * OUTPUT_ROW_BYTES, (
        f"producer output size {len(raw)} != expected {N_OUT * OUTPUT_ROW_BYTES}"
    )
    rows = np.frombuffer(raw, dtype=np.float32).reshape(N_OUT, MM_LANES)
    all_poison_rows = np.all(rows == _POISON, axis=1)
    assert not all_poison_rows.any(), (
        f"producer left {all_poison_rows.sum()} whole rows as untouched poison "
        f"(row indices {np.nonzero(all_poison_rows)[0].tolist()})"
    )
    return raw


def _addressing_only_slice(raw_per_stream: dict[int, bytes], row0: int) -> bytes | None:
    """Attempt to build attn_scores_km_64x48's flat [P,N_HEAD,D,N] input file
    using ONLY base+stride byte-range selection over the matmul's raw output
    -- no numeric transform. Returns None if the row PITCH itself disagrees
    (512 B/channel produced vs 256 B/channel -- i.e. N*4 -- expected), which
    means a strip/re-pack step is unavoidable and pure addressing fails.
    """
    mm_row_bytes = OUTPUT_ROW_BYTES              # 512: matmul's store pitch
    km_row_bytes = N * 4                          # 256: score kernel's expected pitch
    if mm_row_bytes != km_row_bytes:
        return None
    # (unreachable while pitches disagree, kept for completeness / future-proofing)
    out = bytearray(P * N_HEAD * D * N * 4)
    for p, raw in raw_per_stream.items():
        block = raw[row0 * mm_row_bytes:(row0 + N_HEAD * D) * mm_row_bytes]
        lo = p * N_HEAD * D * N * 4
        out[lo:lo + len(block)] = block
    return bytes(out)


def test_qkv_matmul_output_pitch_blocks_pure_addressing_into_scores_km(
    tmp_path: Path,
) -> None:
    """Confirms the finding: matmul_576x192_x128's row pitch (512 B/channel,
    padded) disagrees with attn_scores_km_64x48's expected input file pitch
    (256 B/channel, tightly packed), so the seam cannot be bridged by pure
    base+stride addressing alone -- a strip-padding repack is required.
    """
    assert OUTPUT_ROW_BYTES == 512
    assert N * 4 == 256
    assert OUTPUT_ROW_BYTES != N * 4, (
        "if this ever becomes equal, pure addressing may become possible -- "
        "re-derive the seam instead of trusting this xfail-style assertion"
    )

    rng = np.random.RandomState(0x9EC0)
    W = rng.uniform(-1.0, 1.0, size=(N_OUT, MM_K)).astype(np.float32)
    D_acts = {
        p: rng.uniform(-1.0, 1.0, size=(MM_K, MM_N_TOK)).astype(np.float32)
        for p in range(P)
    }

    raw_per_stream = {
        p: _run_qkv_matmul_one_stream(D_acts[p], W, tmp_path, tag=str(p))
        for p in range(P)
    }

    q_bytes = _addressing_only_slice(raw_per_stream, _Q_ROW0)
    k_bytes = _addressing_only_slice(raw_per_stream, _K_ROW0)
    assert q_bytes is None and k_bytes is None, (
        "pure addressing unexpectedly succeeded -- update the report"
    )


def test_qkv_matmul_to_scores_km_agrees_after_repack(
    km_inst_file: Path, tmp_path: Path,
) -> None:
    """Companion correctness check: WITH a minimal strip-padding repack (drop
    each row's 64 padding lanes; the row ORDER and (stream,head,channel)
    mapping need no other change), the chain is numerically correct end to
    end against an independent reference computed straight from W and D_act.
    This isolates the one disagreement (store pitch) from everything else
    (block order, channel order, per-stream invocation) -- all of which
    already line up.
    """
    rng = np.random.RandomState(0x9EC1)
    W = rng.uniform(-1.0, 1.0, size=(N_OUT, MM_K)).astype(np.float32)
    D_acts = {
        p: rng.uniform(-1.0, 1.0, size=(MM_K, MM_N_TOK)).astype(np.float32)
        for p in range(P)
    }

    raw_per_stream = {
        p: _run_qkv_matmul_one_stream(D_acts[p], W, tmp_path, tag=f"r{p}")
        for p in range(P)
    }

    # Minimal repack: strip the 64 padding lanes from each 512 B row (the ONE
    # byte-layout disagreement), keep every index (row order, block order)
    # exactly as the matmul produced it.
    def block_tightly_packed(raw: bytes, row0: int) -> np.ndarray:
        rows = np.frombuffer(raw, dtype=np.float32).reshape(N_OUT, MM_LANES)
        return rows[row0:row0 + N_HEAD * D, :N]     # [N_HEAD*D, N], padding stripped

    q_blocks = np.stack(
        [block_tightly_packed(raw_per_stream[p], _Q_ROW0) for p in range(P)]
    ).reshape(P, N_HEAD, D, N)
    k_blocks = np.stack(
        [block_tightly_packed(raw_per_stream[p], _K_ROW0) for p in range(P)]
    ).reshape(P, N_HEAD, D, N)

    q_path = tmp_path / "q_repacked.bin"
    k_path = tmp_path / "k_repacked.bin"
    q_path.write_bytes(q_blocks.astype(np.float32).tobytes())
    k_path.write_bytes(k_blocks.astype(np.float32).tobytes())
    output_path = tmp_path / "scores_out.bin"

    head = 1
    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    state.xmem.write_address(
        SBASE, bytearray(np.full(S_ROWS * LANES, _POISON, dtype=np.float32).tobytes())
    )
    app = AttnScoresKM64x48App(
        inst_path=km_inst_file,
        input_path=q_path,
        weights_path=k_path,
        output_path=output_path,
        head=head,
    )
    state, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    got = np.frombuffer(output_path.read_bytes(), dtype=np.float32).reshape(
        P * N, LANES
    )[:, :N]

    # Independent reference: C = W @ D_act per stream, then slice Q/K blocks
    # for `head` directly from W/D_act -- not from either kernel's own golden.
    expected = np.zeros((P, N, N), dtype=np.float64)
    for p in range(P):
        C = (W.astype(np.float64) @ D_acts[p].astype(np.float64))  # [576, 64]
        Qh = C[_Q_ROW0 + head * D: _Q_ROW0 + (head + 1) * D, :]     # [D, N]
        Kh = C[_K_ROW0 + head * D: _K_ROW0 + (head + 1) * D, :]     # [D, N]
        # S[p, i, s] = sum_c Q[p,i,c] * K[p,s,c]; kernel stores key-major:
        # row (p*N + s) holds S[p, 0:N, s].
        S = np.einsum("ci,cs->is", Qh, Kh)                          # [query i, key s]
        expected[p] = S.T                                           # key-major

    expected = expected.reshape(P * N, N)

    max_err = float(np.max(np.abs(got.astype(np.float64) - expected)))
    print(f"seam qkv_matmul->scores_km max abs error = {max_err:.3e}")

    np.testing.assert_allclose(
        got, expected, rtol=2e-3, atol=2e-2,
        err_msg=(
            "chained matmul_576x192_x128 -> attn_scores_km_64x48 (after "
            "padding-strip repack) mismatch vs independent reference"
        ),
    )
