"""The full Layer 5 layer in the packed layout on one ``Chain`` (E07-24): 240
channels, 16 tokens, 8 channels a packed row, 4 heads of 60.

    packed LayerNorm -> QKV -> unpack -> attention -> pack -> OutProj
      -> residual add -> packed LayerNorm -> FFN1 (SiLU) -> FFN2 -> residual add

The three parts of the layer each on their own data, then the whole layer.
Every packed hand-off is a resident tensor (``mobilevit_packed``); attention
is the query-major mapping (``qk_scores_16x60`` a head a step,
``softmax_rows_partial``, ``attn_v_16x60`` the four heads in one step). Each
stage is held to the shared references chained in float64, at the old tests'
bounds: LayerNorm 1e-3, every other stage 1e-2.

Nightly (``:tests_nightly``).
"""

from __future__ import annotations

import numpy as np
import torch
from mobilevit_layer import FP32, STAGES, LayerRun, attention, reference, reference_tail, weights
from mobilevit_packed import PACKED_L5, back, front, hold_at_row_0, pack, packed, run_layer
from mobilevit_packed import attention as packed_attention

from ipu_harness.chain import Chain

CFG = PACKED_L5
C, N, HEADS = CFG.channels, CFG.tokens, CFG.layer.heads
BOUNDS = {name: 1e-3 if name.startswith("ln") else 1e-2 for name in STAGES}


def _weights(rs: np.random.RandomState) -> dict[str, np.ndarray]:
    return weights(rs, C, 4 * C, gamma=(0.8, 1.2), beta=(-0.1, 0.1), w=(-0.05, 0.05))


def _held(got: dict[str, np.ndarray], want: dict[str, np.ndarray], run: LayerRun) -> None:
    """Each stage of ``got`` within its bound of ``want``, and no step past its golden."""
    errors = {name: float(np.max(np.abs(got[name] - want[name]))) for name in got if name in want}
    print(f"L5 packed: max abs error by stage {errors}")
    print(f"L5 packed: cycles by kernel {dict(sorted(run.cycles.items()))}")
    assert not run.misses, "steps missed their own goldens:\n" + "\n".join(run.misses)
    over = {name: err for name, err in errors.items() if err >= BOUNDS[name]}
    assert not over, f"stages over the old bounds: {over}"


def test_stage1_layernorm_qkv_unpack():
    """LayerNorm -> QKV -> unpack: each unpacked third is its stage's channels."""
    rs = np.random.RandomState(301)
    x = rs.uniform(-1.0, 1.0, size=(C, N)).astype(np.float32)
    w = _weights(rs)
    run, chain = LayerRun(), Chain(FP32)
    got, qkv = front(run, chain, CFG, hold_at_row_0(chain, x), w)
    _held(got, reference(x, w, heads=HEADS), run)
    for j, third in enumerate(qkv):
        np.testing.assert_array_equal(third, got["qkv"][j * C : (j + 1) * C])


def test_stage2_attention():
    """Q, K, V -> scores (a head a step) -> softmax -> attn@V -> pack."""
    rs = np.random.RandomState(303)
    q, k, v = (rs.uniform(-1.0, 1.0, size=(C, N)).astype(np.float32) for _ in range(3))
    run, chain = LayerRun(), Chain(FP32)
    got = packed_attention(run, chain, CFG, q, k, v)
    got["packed"] = chain.read(pack(run, chain, CFG, got["attn_v"])).astype(np.float64)
    heads = [torch.from_numpy(a.astype(np.float64)).reshape(HEADS, -1, N) for a in (q, k, v)]
    scores, probs, out = (t.numpy() for t in attention(*heads))
    _held(got, {"scores": scores, "softmax": probs, "attn_v": out.reshape(C, N)}, run)
    np.testing.assert_array_equal(got["packed"], packed(got["attn_v"]))


def test_stage3_outproj_residual_ffn_residual():
    """OutProj -> residual add (with the residual input at row 0) ->
    LayerNorm -> FFN1 (SiLU) -> FFN2 -> residual add."""
    rs = np.random.RandomState(302)

    def u(lo: float, hi: float, *shape: int) -> np.ndarray:
        return rs.uniform(lo, hi, size=shape).astype(np.float32)

    attn, w_outproj, residual_in = u(-1, 1, C, N), u(-0.05, 0.05, C, C), u(-1, 1, C, N)
    w = {
        "w_outproj": w_outproj,
        "ln2_gamma": u(0.8, 1.2, C),
        "ln2_beta": u(-0.1, 0.1, C),
        "w_ffn1": u(-0.05, 0.05, 4 * C, C),
        "w_ffn2": u(-0.05, 0.05, C, 4 * C),
    }  # the old draw order
    run, chain = LayerRun(), Chain(FP32)
    x0 = hold_at_row_0(chain, residual_in)
    got = back(run, chain, CFG, chain.write(packed(attn)), x0, w)
    _held(got, reference_tail(attn, residual_in, w), run)


def test_full_l5_layer_packed_end_to_end():
    rs = np.random.RandomState(999)
    x = rs.uniform(-1.0, 1.0, size=(C, N)).astype(np.float32)
    w = _weights(rs)
    run = LayerRun()
    got = run_layer(run, Chain(FP32), CFG, x, w)
    _held(got, reference(x, w, heads=HEADS), run)
    np.testing.assert_array_equal(got["packed"], packed(got["attn_v"]))
