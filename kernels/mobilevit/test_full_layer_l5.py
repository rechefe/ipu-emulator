"""The full Layer 5 transformer layer on one ``Chain`` (E07-24): d = 240, 16
tokens a stream, 4 streams, 4 heads of 60, 3 layers.

A seam test sees two kernels; this runs the whole layer, three times with
distinct weights, so it sees what no seam can: error accumulating through a
layer and across layers, and state a step leaves for the next. Both attention
mappings run, never mixed: query-major (``qk_scores_16x60``,
``attn_v_16x60``, AGG) and key-major (``attn_scores_km_16x60``,
``attn_v_bcast_60``, ACC.ADD); softmax is the real kernels'
(``softmax_rows_partial``, ``softmax_columns_packed``). Each layer's output
is held to the shared references chained in float64 (``mobilevit_layer``).

Nightly (``:tests_nightly``): a layer takes minutes in the emulator.
"""

from __future__ import annotations

import numpy as np
import pytest
from mobilevit_layer import L5, STAGES, run_stack, stage_errors

from ipu_harness.tolerance import Tier

LAYERS = 3
SCALE = 1.0 / np.sqrt(L5.head_dim)
#: The old bound on each layer's output, unscaled and scaled.
LAYER_BOUND = 5e-1


@pytest.mark.parametrize("scaled", [False, True], ids=["unscaled", "scaled"])
@pytest.mark.parametrize("mapping", ["query_major", "key_major"])
def test_full_l5_layer_stack_agrees(mapping: str, scaled: bool):
    """Three layers back to back; ``scaled`` applies 1/sqrt(head_dim) to Q on
    both sides (the scale experiment: does the missing scale explain the
    error's growth from layer to layer?)."""
    seed = (0x1_5002 if scaled else 0x1_5000) + {"query_major": 0, "key_major": 1}[mapping]
    errors, run = run_stack(L5, mapping, seed=seed, layers=LAYERS, scale=SCALE if scaled else 1.0)
    assert not run.misses, "steps missed their own goldens:\n" + "\n".join(run.misses)
    for i, err in enumerate(errors):
        assert err < LAYER_BOUND, f"layer {i}: max abs error {err:.3e} (trace {errors})"


def _budget(tier: str, atol: float, measured: float) -> Tier:
    return Tier(
        tier, 0.0, atol, None, f"one chained L5 layer, measured {measured:.2e} (E07-24); 2x"
    )


#: Each stage's max abs error against the float64 reference, measured on the
#: probe layer below, with 2x headroom: a stage that gets worse fails here.
BUDGETS = {
    "unscaled": {
        "ln1": _budget("FP32_LONG", 1.7e-6, 8.00e-07),
        "qkv": _budget("FP32_LONG", 1.2e-5, 5.74e-06),
        "scores": _budget("FP32_ORDER", 2.1e-4, 1.01e-04),
        "softmax": _budget("TRANSCENDENT", 1.1e-5, 5.25e-06),
        "attn_v": _budget("FP32_ORDER", 9.8e-5, 4.86e-05),
        "out_proj": _budget("FP32_LONG", 1.5e-4, 7.07e-05),
        "resid1": _budget("FP32_LONG", 1.5e-4, 7.05e-05),
        "ln2": _budget("FP32_LONG", 2.0e-5, 9.77e-06),
        "ffn1": _budget("FP32_LONG", 4.9e-5, 2.42e-05),
        "ffn2": _budget("FP32_LONG", 1.3e-4, 6.30e-05),
        "resid2": _budget("FP32_LONG", 1.9e-4, 9.23e-05),
    },
    "scaled": {
        "ln1": _budget("FP32_LONG", 2.3e-6, 1.13e-06),
        "qkv": _budget("FP32_LONG", 1.7e-5, 8.47e-06),
        "scores": _budget("FP32_ORDER", 3.6e-5, 1.79e-05),
        "softmax": _budget("TRANSCENDENT", 7.1e-6, 3.55e-06),
        "attn_v": _budget("FP32_ORDER", 6.9e-5, 3.44e-05),
        "out_proj": _budget("FP32_LONG", 1.2e-4, 5.77e-05),
        "resid1": _budget("FP32_LONG", 1.2e-4, 5.81e-05),
        "ln2": _budget("FP32_LONG", 2.6e-5, 1.28e-05),
        "ffn1": _budget("FP32_LONG", 4.8e-5, 2.39e-05),
        "ffn2": _budget("FP32_LONG", 1.3e-4, 6.15e-05),
        "resid2": _budget("FP32_LONG", 1.8e-4, 8.81e-05),
    },
}


@pytest.mark.parametrize("label", ["unscaled", "scaled"])
def test_full_l5_per_stage_error(label: str):
    """One query-major layer, each stage against its own float64 reference:
    where the error enters, and that no stage's grows past its budget."""
    scaled = label == "scaled"
    errors, run = stage_errors(
        L5,
        seed=0x1_5010 if scaled else 0x1_5011,
        tag="l5_stage_probe",
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
