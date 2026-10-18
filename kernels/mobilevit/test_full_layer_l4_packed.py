"""The full Layer 4 layer in the packed layout on one ``Chain`` (E07-24, from
the old harness's ``test/mobilevit/test_full_layer_l4_packed.py``): 192
channels, 64 tokens, 2 channels a packed row, 4 heads of 48.

    packed LayerNorm -> QKV -> unpack -> attention -> pack -> OutProj
      -> residual add -> packed LayerNorm -> FFN1 (SiLU) -> FFN2 -> residual add

Every packed hand-off is a resident tensor (``mobilevit_packed``). Attention
is the query-major mapping (``qk_scores_64x48``, ``softmax_rows_partial``,
``attn_v_64x48``), each one step over its 16 (stream, head) blocks: the
layer is one stream, so blocks 4 to 15 are zero and only the real four are
compared. Each stage is held to the shared references chained in float64,
at the old test's bounds.

Nightly (``:tests_nightly``).
"""

from __future__ import annotations

import numpy as np
from mobilevit_layer import FP32, STAGES, LayerRun, reference, weights
from mobilevit_packed import PACKED_L4, packed, run_layer

from ipu_harness.chain import Chain

#: The old test's bounds: LayerNorm 1e-3, every other stage 1e-2.
BOUNDS = {name: 1e-3 if name.startswith("ln") else 1e-2 for name in STAGES}


def test_full_l4_layer_packed_end_to_end():
    cfg = PACKED_L4
    rs = np.random.RandomState(4001)
    x = rs.uniform(-1.0, 1.0, size=(cfg.channels, cfg.tokens)).astype(np.float32)
    w = weights(
        rs, cfg.channels, 4 * cfg.channels, gamma=(0.8, 1.2), beta=(-0.1, 0.1), w=(-0.05, 0.05)
    )
    run = LayerRun()
    got = run_layer(run, Chain(FP32), cfg, x, w)
    want = reference(x, w, heads=cfg.layer.heads)
    errors = {name: float(np.max(np.abs(got[name] - want[name]))) for name in STAGES}
    print(f"L4 packed: max abs error by stage {errors}")
    print(f"L4 packed: cycles by kernel {dict(sorted(run.cycles.items()))}")
    assert not run.misses, "steps missed their own goldens:\n" + "\n".join(run.misses)
    np.testing.assert_array_equal(got["packed"], packed(got["attn_v"]))
    over = {name: err for name, err in errors.items() if err >= BOUNDS[name]}
    assert not over, f"stages over the old bounds: {over}"
