"""DECISIVE TEST (throwaway, not part of standing suite): feed
qk_scores_16x60's raw, uncropped output directly into softmax_rows_partial
with NO numpy repacking in between, and report whether the result is
correct or garbage. See conversation record for the question this answers.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ipu_as.lark_tree import assemble_to_bin_file
from ipu_as.label import reset_labels
from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.attention.qk_scores_16x60 import QkScores16x60App, N, D, ROW_BYTES
from ipu_apps.softmax.softmax_rows_partial import SoftmaxRowsPartialApp
from ipu_apps.attention.attn_scores_km_16x60 import (
    AttnScoresKM16x60App, N_TOK, N_TG, N_TPG, N_HEADS, LANES as KM_LANES,
)
from ipu_apps.softmax.softmax_columns_packed import SoftmaxColumnsPackedApp

_QK_SRC = Path(__file__).resolve().parents[1] / "src/ipu_apps/attention/qk_scores_16x60/qk_scores_16x60.asm"
_SM_SRC = Path(__file__).resolve().parents[1] / "src/ipu_apps/softmax/softmax_rows_partial/softmax_rows_partial.asm"
_KM_SRC = Path(__file__).resolve().parents[1] / "src/ipu_apps/attention/attn_scores_km_16x60/attn_scores_km_16x60.asm"
_SMC_SRC = Path(__file__).resolve().parents[1] / "src/ipu_apps/softmax/softmax_columns_packed/softmax_columns_packed.asm"


def _ref_rows_softmax(x: np.ndarray) -> np.ndarray:
    m = np.max(x, axis=1, keepdims=True)
    z = np.exp(x - m)
    return z / z.sum(axis=1, keepdims=True)


def test_decisive_uncropped_feed(tmp_path: Path) -> None:
    reset_labels()
    qk_bin = tmp_path / "qk.bin"
    assemble_to_bin_file(_QK_SRC.read_text(encoding="utf-8"), str(qk_bin))

    reset_labels()
    sm_bin = tmp_path / "sm.bin"
    assemble_to_bin_file(_SM_SRC.read_text(encoding="utf-8"), str(sm_bin))

    rng = np.random.RandomState(0)
    Q = (rng.randn(D, N) * 2).astype(np.float32)
    K = (rng.randn(D, N) * 2).astype(np.float32)

    q_path = tmp_path / "q.bin"
    k_path = tmp_path / "k.bin"
    q_path.write_bytes(Q.tobytes())
    k_path.write_bytes(K.tobytes())

    qk_state = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
    qk_out = tmp_path / "qk_out.bin"
    qk_app = QkScores16x60App(inst_path=qk_bin, query_path=q_path, key_path=k_path, output_path=qk_out)
    _, qk_cycles = qk_app.run(max_cycles=2_000_000, state=qk_state)
    print(f"qk_scores_16x60 ran: {qk_cycles} cycles")

    raw = qk_out.read_bytes()
    print(f"qk_scores_16x60 raw output size: {len(raw)} bytes (N*ROW_BYTES = {N * ROW_BYTES})")
    assert len(raw) == N * ROW_BYTES

    rows = np.frombuffer(raw, dtype=np.float32).reshape(N, ROW_BYTES // 4)
    print("row 0 lanes[0:20]:", rows[0][:20])
    print("row 0 lanes[16:32] (should be padding):", rows[0][16:32])
    print("row 1 lanes[0:20]:", rows[1][:20])

    scores_true = Q.T @ K
    expected = _ref_rows_softmax(scores_true)

    sm_in = tmp_path / "sm_in.bin"
    sm_in.write_bytes(raw)  # ALL 8192 bytes, untouched -- no crop, no repack
    sm_out = tmp_path / "sm_out.bin"

    sm_app = SoftmaxRowsPartialApp(inst_path=sm_bin, input_path=sm_in, output_path=sm_out, n=N, rows=N)
    try:
        _, sm_cycles = sm_app.run(max_cycles=2_000_000)
        print(f"softmax_rows_partial ran: {sm_cycles} cycles")
        got_raw = sm_out.read_bytes()
        print(f"softmax_rows_partial output size: {len(got_raw)} bytes (expected {N * N * 4})")
        got = np.frombuffer(got_raw, dtype=np.float32).reshape(N, N)
        for r in range(N):
            row_err = float(np.max(np.abs(got[r] - expected[r])))
            print(f"row {r:2d}: max abs error = {row_err:.6e}")
        err = float(np.max(np.abs(got - expected)))
        print(f"max abs error (all rows): {err:.6e}")
        print("VERDICT:", "CORRECT" if err < 1e-4 else "GARBAGE")
    except Exception as e:
        print(f"softmax_rows_partial CRASHED on uncropped input: {type(e).__name__}: {e}")
        raise


def test_decisive_uncropped_feed_key_major(tmp_path: Path) -> None:
    reset_labels()
    km_bin = tmp_path / "km.bin"
    assemble_to_bin_file(_KM_SRC.read_text(encoding="utf-8"), str(km_bin))

    reset_labels()
    smc_bin = tmp_path / "smc.bin"
    assemble_to_bin_file(_SMC_SRC.read_text(encoding="utf-8"), str(smc_bin))

    rng = np.random.RandomState(1)
    head = 1
    n_chan = N_HEADS * D
    Q = rng.uniform(-1.0, 1.0, size=(n_chan, N_TOK)).astype(np.float32)
    K = rng.uniform(-1.0, 1.0, size=(n_chan, N_TOK)).astype(np.float32)

    q_path = tmp_path / "q.bin"
    k_path = tmp_path / "k.bin"
    q_path.write_bytes(Q.tobytes())
    k_path.write_bytes(K.tobytes())

    km_state = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
    km_out = tmp_path / "km_out.bin"
    km_app = AttnScoresKM16x60App(inst_path=km_bin, input_path=q_path, weights_path=k_path,
                                   output_path=km_out, head=head)
    _, km_cycles = km_app.run(max_cycles=2_000_000, state=km_state)
    print(f"attn_scores_km_16x60 ran: {km_cycles} cycles")

    raw = km_out.read_bytes()
    print(f"attn_scores_km_16x60 raw output size: {len(raw)} bytes "
          f"(N_TOK*N_TG*KM_LANES*4 = {N_TOK * N_TG * KM_LANES * 4})")
    assert len(raw) == N_TOK * N_TG * KM_LANES * 4

    rows = np.frombuffer(raw, dtype=np.float32).reshape(N_TOK, KM_LANES)
    print("row 0 lanes[0:20]:", rows[0][:20])
    print("row 1 lanes[0:20]:", rows[1][:20])

    lo = head * D
    q_head = Q[lo:lo + D]
    k_head = K[lo:lo + D]
    scores_true = q_head.T @ k_head  # [query, key]

    def ref_cols_softmax(x):
        m = np.max(x, axis=0, keepdims=True)
        z = np.exp(x - m)
        return z / z.sum(axis=0, keepdims=True)

    # key-major storage: row s = key s = scores_true[:, s]; softmax reduces
    # over the key axis, i.e. down each COLUMN of key-major storage (axis=-2
    # in [key, query] shape) -- see softmax_key_major's contract.
    scores_km_true = scores_true.T  # [key, query]
    expected_km = ref_cols_softmax(scores_km_true.T).T  # reduce over key axis, stay [key, query]

    smc_in = tmp_path / "smc_in.bin"
    smc_in.write_bytes(raw)  # ALL bytes, untouched -- no crop, no repack
    smc_out = tmp_path / "smc_out.bin"

    smc_app = SoftmaxColumnsPackedApp(inst_path=smc_bin, input_path=smc_in, output_path=smc_out,
                                       rows=N_TOK, width=N_TOK)
    try:
        _, smc_cycles = smc_app.run(max_cycles=2_000_000)
        print(f"softmax_columns_packed ran: {smc_cycles} cycles")
        got_raw = smc_out.read_bytes()
        print(f"softmax_columns_packed output size: {len(got_raw)} bytes (expected {N_TOK * N_TOK * 4})")
        got = np.frombuffer(got_raw, dtype=np.float32).reshape(N_TOK, N_TOK)
        for r in range(N_TOK):
            row_err = float(np.max(np.abs(got[r] - expected_km[r])))
            print(f"row {r:2d}: max abs error = {row_err:.6e}")
        err = float(np.max(np.abs(got - expected_km)))
        print(f"max abs error (all rows): {err:.6e}")
        print("VERDICT:", "CORRECT" if err < 1e-4 else "GARBAGE")
    except Exception as e:
        print(f"softmax_columns_packed CRASHED on uncropped input: {type(e).__name__}: {e}")
        raise
