"""The full Layer 4 transformer layer on one ``Chain`` (E07-24, from
``src/tools/ipu-apps/test/mobilevit/test_full_layer_l4.py``): d = 192, 64
tokens a stream, 4 streams, 4 heads of 48, 4 layers.

A seam test sees two kernels; this runs the whole layer, four times with
distinct weights, so it sees what no seam can: error accumulating through a
layer and across layers, and state a step leaves for the next. Both attention
mappings run, never mixed: query-major (``qk_scores_64x48``,
``attn_v_64x48``, AGG) and key-major (``attn_scores_km_64x48``,
``attn_v_bcast_48``, ACC.ADD); softmax is the real kernels'
(``softmax_rows_partial``, ``softmax_columns_packed``). Each layer's output
is held to the shared references chained in float64 (``mobilevit_layer``).

Nightly (``:tests_nightly``): a layer takes minutes in the emulator.
"""

from __future__ import annotations

import numpy as np
import pytest
from mobilevit_layer import L4, STAGES, run_stack, stage_errors

from ipu_harness.tolerance import Tier

LAYERS = 4
SCALE = 1.0 / np.sqrt(L4.head_dim)
#: The old bound on each layer's output, unscaled and scaled.
LAYER_BOUND = 5e-1


@pytest.mark.parametrize("scaled", [False, True], ids=["unscaled", "scaled"])
@pytest.mark.parametrize("mapping", ["query_major", "key_major"])
def test_full_l4_layer_stack_agrees(mapping: str, scaled: bool):
    """Four layers back to back; ``scaled`` applies 1/sqrt(head_dim) to Q on
    both sides (the scale experiment: does the missing scale explain the
    error's growth from layer to layer?)."""
    seed = (0x1_4002 if scaled else 0x1_4000) + {"query_major": 0, "key_major": 1}[mapping]
    errors, run = run_stack(L4, mapping, seed=seed, layers=LAYERS, scale=SCALE if scaled else 1.0)
    assert not run.misses, "steps missed their own goldens:\n" + "\n".join(run.misses)
    for i, err in enumerate(errors):
        assert err < LAYER_BOUND, f"layer {i}: max abs error {err:.3e} (trace {errors})"


def _budget(tier: str, atol: float, measured: float) -> Tier:
    return Tier(
        tier, 0.0, atol, None, f"one chained L4 layer, measured {measured:.2e} (E07-24); 2x"
    )


#: Each stage's max abs error against the float64 reference, measured on the
#: probe layer below, with 2x headroom: a stage that gets worse fails here.
BUDGETS = {
    "unscaled": {
        "ln1": _budget("FP32_LONG", 1.5e-6, 7.17e-07),
        "qkv": _budget("FP32_LONG", 1.1e-5, 5.42e-06),
        "scores": _budget("FP32_ORDER", 1.8e-4, 8.73e-05),
        "softmax": _budget("TRANSCENDENT", 3.3e-5, 1.60e-05),
        "attn_v": _budget("FP32_ORDER", 2.5e-4, 1.22e-04),
        "out_proj": _budget("FP32_LONG", 4.6e-4, 2.26e-04),
        "resid1": _budget("FP32_LONG", 4.6e-4, 2.26e-04),
        "ln2": _budget("FP32_LONG", 8.5e-5, 4.23e-05),
        "ffn1": _budget("FP32_LONG", 1.7e-4, 8.07e-05),
        "ffn2": _budget("FP32_LONG", 4.4e-4, 2.17e-04),
        "resid2": _budget("FP32_LONG", 6.8e-4, 3.36e-04),
    },
    "scaled": {
        "ln1": _budget("FP32_LONG", 1.8e-6, 8.53e-07),
        "qkv": _budget("FP32_LONG", 1.4e-5, 6.94e-06),
        "scores": _budget("FP32_ORDER", 2.8e-5, 1.40e-05),
        "softmax": _budget("TRANSCENDENT", 6.7e-6, 3.32e-06),
        "attn_v": _budget("FP32_ORDER", 5.6e-5, 2.76e-05),
        "out_proj": _budget("FP32_LONG", 9.5e-5, 4.71e-05),
        "resid1": _budget("FP32_LONG", 9.5e-5, 4.71e-05),
        "ln2": _budget("FP32_LONG", 2.1e-5, 1.04e-05),
        "ffn1": _budget("FP32_LONG", 4.9e-5, 2.42e-05),
        "ffn2": _budget("FP32_LONG", 1.1e-4, 5.01e-05),
        "resid2": _budget("FP32_LONG", 1.5e-4, 7.22e-05),
    },
}


@pytest.mark.parametrize("label", ["unscaled", "scaled"])
def test_full_l4_per_stage_error(label: str):
    """One query-major layer, each stage against its own float64 reference:
    where the error enters, and that no stage's grows past its budget."""
    scaled = label == "scaled"
    errors, run = stage_errors(
        L4,
        seed=0x1_4010 if scaled else 0x1_4011,
        tag="l4_stage_probe",
        scale=SCALE if scaled else 1.0,
    )
    assert not run.misses, "steps missed their own goldens:\n" + "\n".join(run.misses)
    assert tuple(errors) == STAGES
    over = {
        stage: f"{err:.3e} > {BUDGETS[label][stage].atol:.3e}"
        for stage, err in errors.items()
        if err > BUDGETS[label][stage].atol
    }
    assert not over, f"stages over budget: {over}"
