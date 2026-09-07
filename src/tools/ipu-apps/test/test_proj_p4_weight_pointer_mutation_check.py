"""Mutation-based re-check of the weight-pointer-persistence bug class across
all 8 L4/L5 ``proj_*_p4`` kernels (QKV/OutProj/FFN1/FFN2, d in {192, 240}).

kernel_docs/kernel_layer_map.md records that the first draft of
``proj_outproj_144_p4`` (L3) reset its per-output-channel weight-row pointer
(``weight_row_off``) in the wrong place, so every output channel ``j`` read
output channel 0's weight row instead of its own -- 99.3% of output elements
wrong, caught by that kernel's own numeric wide test comparing against
``W @ D[p][tg]`` computed with the REAL, per-row-varying ``W``. That test
already has the right shape to catch this; this module exists because the L3
fix was later PORTED to L4/L5 by copying the same ``__init__.py`` pattern
(confirmed byte-identical in this task's kernel research) across 12
``proj_*_p4`` kernels, and the task asks to re-check the 8 L4/L5 members of
that family BY MUTATION rather than by re-reading the (correct-looking)
``.asm`` -- inspection is exactly what let the original bug ship once.

"Mutation" here does not mean editing the kernel or ipu.py (both are
untouched, per the standing method for this round). It means constructing an
input the buggy behaviour and the correct behaviour would visibly disagree
on, and confirming the REAL kernel's output matches the CORRECT behaviour,
not the buggy one. Concretely: give every output channel `j` a weight row
that is injectively identifiable (row j is filled with a distinct constant),
so if the kernel silently reused channel 0's row for every `j` (the exact
historical bug), every channel's output would read back as channel 0's
value -- a gross, unmissable mismatch against the corresponding per-channel
expected value, not a subtle numeric one. This is stronger than the existing
per-kernel wide tests' random `W`, which WOULD also catch the bug (any two
distinct random rows differ) but does not make the bug's signature -- "every
row equals row 0" -- the explicit thing being checked for.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.projections.proj_qkv_192_p4 import ProjQkv192P4App, K as QKV192_K, N_OUT as QKV192_N_OUT, N_TOK as QKV192_N_TOK, N_STREAM as QKV192_P, LANES as QKV192_LANES
from ipu_apps.projections.proj_qkv_240_p4 import ProjQKV240P4App, K as QKV240_K, N_OUT as QKV240_N_OUT, N_TOK as QKV240_N_TOK, N_STREAM as QKV240_P, LANES as QKV240_LANES
from ipu_apps.projections.proj_outproj_192_p4 import ProjOutProj192P4App, K as OP192_K, N_OUT as OP192_N_OUT, N_TOK as OP192_N_TOK, N_STREAM as OP192_P, LANES as OP192_LANES
from ipu_apps.projections.proj_outproj_240_p4 import ProjOutProj240P4App, K as OP240_K, N_OUT as OP240_N_OUT, N_TOK as OP240_N_TOK, N_STREAM as OP240_P, LANES as OP240_LANES
from ipu_apps.projections.proj_ffn1_192_p4 import ProjFFN1192P4App, K as F1_192_K, N_OUT as F1_192_N_OUT, N_TOK as F1_192_N_TOK, N_STREAM as F1_192_P, LANES as F1_192_LANES
from ipu_apps.projections.proj_ffn1_240_p4 import ProjFFN1240P4App, K as F1_240_K, N_OUT as F1_240_N_OUT, N_TOK as F1_240_N_TOK, N_STREAM as F1_240_P, LANES as F1_240_LANES
from ipu_apps.projections.proj_ffn2_192_p4 import ProjFFN2192P4App, K as F2_192_K, N_OUT as F2_192_N_OUT, N_TOK as F2_192_N_TOK, N_STREAM as F2_192_P, LANES as F2_192_LANES
from ipu_apps.projections.proj_ffn2_240_p4 import ProjFFN2240P4App, K as F2_240_K, N_OUT as F2_240_N_OUT, N_TOK as F2_240_N_TOK, N_STREAM as F2_240_P, LANES as F2_240_LANES

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

_APP_SRC = Path(__file__).resolve().parents[1] / "src/ipu_apps/projections"
_INST_CACHE: dict[str, Path] = {}


def _inst(name: str) -> Path:
    """Assemble (once) and cache ``ipu_apps/projections/<name>/<name>.asm``."""
    if name not in _INST_CACHE:
        asm_path = _APP_SRC / name / f"{name}.asm"
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"proj_mutation_{name}_"))
        inst_path = tmp_dir / f"{name}.bin"
        assemble_to_bin_file(asm_path.read_text(encoding="utf-8"), str(inst_path))
        _INST_CACHE[name] = inst_path
    return _INST_CACHE[name]


_KERNELS = [
    pytest.param(ProjQkv192P4App, "proj_qkv_192_p4", QKV192_K, QKV192_N_OUT, QKV192_N_TOK, QKV192_P, QKV192_LANES, False, id="proj_qkv_192_p4"),
    pytest.param(ProjQKV240P4App, "proj_qkv_240_p4", QKV240_K, QKV240_N_OUT, QKV240_N_TOK, QKV240_P, QKV240_LANES, False, id="proj_qkv_240_p4"),
    pytest.param(ProjOutProj192P4App, "proj_outproj_192_p4", OP192_K, OP192_N_OUT, OP192_N_TOK, OP192_P, OP192_LANES, False, id="proj_outproj_192_p4"),
    pytest.param(ProjOutProj240P4App, "proj_outproj_240_p4", OP240_K, OP240_N_OUT, OP240_N_TOK, OP240_P, OP240_LANES, False, id="proj_outproj_240_p4"),
    # FFN1 applies ACTIVATE.QUANTIZE silu at its store (see
    # kernel_layer_map.md's "FFN activation is silu, not identity") -- the
    # expected per-channel value must go through silu too, or a perfectly
    # correct kernel misreports as "not equal", not signal a real bug.
    pytest.param(ProjFFN1192P4App, "proj_ffn1_192_p4", F1_192_K, F1_192_N_OUT, F1_192_N_TOK, F1_192_P, F1_192_LANES, True, id="proj_ffn1_192_p4"),
    pytest.param(ProjFFN1240P4App, "proj_ffn1_240_p4", F1_240_K, F1_240_N_OUT, F1_240_N_TOK, F1_240_P, F1_240_LANES, True, id="proj_ffn1_240_p4"),
    pytest.param(ProjFFN2192P4App, "proj_ffn2_192_p4", F2_192_K, F2_192_N_OUT, F2_192_N_TOK, F2_192_P, F2_192_LANES, False, id="proj_ffn2_192_p4"),
    pytest.param(ProjFFN2240P4App, "proj_ffn2_240_p4", F2_240_K, F2_240_N_OUT, F2_240_N_TOK, F2_240_P, F2_240_LANES, False, id="proj_ffn2_240_p4"),
]


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


@pytest.mark.parametrize("App,name,K,N_OUT,N_TOK,N_STREAM,LANES,applies_silu", _KERNELS)
def test_proj_p4_weight_row_is_not_stuck_at_channel_zero(
    App, name, K, N_OUT, N_TOK, N_STREAM, LANES, applies_silu, tmp_path: Path,
) -> None:
    """Mutation-sensitive re-check: give every output channel j a UNIQUE,
    identifiable weight row (row j filled with the constant j+1, scaled small
    to stay in a well-conditioned FP32 range) and an identity-like activation
    (D[p] = ones), so C[p, j, :] should read back as EXACTLY (j+1)*K for
    every j -- the value channel j's row alone determines. If the kernel
    silently reused weight_row_off from channel 0 for every j (the exact bug
    kernel_docs/kernel_layer_map.md records for proj_outproj_144_p4), every
    channel would instead read back as channel 0's value (1*K), which this
    assertion would catch immediately and specifically -- not just "some
    numeric mismatch somewhere."
    """
    inst_bin = _inst(name)

    # W[j, :] = (j+1) * scale, so C[p, j, t] = sum_k W[j,k] * D[p,k,t] with
    # D[p] = 1 everywhere gives C[p, j, t] = (j+1) * scale * K exactly.
    scale = 0.01
    W = np.zeros((N_OUT, K), dtype=np.float32)
    for j in range(N_OUT):
        W[j, :] = (j + 1) * scale
    D = [np.ones((K, N_TOK), dtype=np.float32) for _ in range(N_STREAM)]

    input_paths = []
    for p in range(N_STREAM):
        p_path = tmp_path / f"d_{p}.bin"
        p_path.write_bytes(D[p].tobytes())
        input_paths.append(p_path)
    weights_path = tmp_path / "w.bin"
    weights_path.write_bytes(W.tobytes())
    output_paths = [tmp_path / f"o_{p}.bin" for p in range(N_STREAM)]

    state = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
    app = App(inst_path=inst_bin, input_paths=input_paths, weights_path=weights_path,
              output_paths=output_paths)
    _, cycles = app.run(max_cycles=30_000_000, state=state)
    assert cycles > 0

    expected_per_channel = (np.arange(N_OUT, dtype=np.float64) + 1) * scale * K
    if applies_silu:
        expected_per_channel = _silu(expected_per_channel)

    for p in range(N_STREAM):
        raw = np.frombuffer(output_paths[p].read_bytes(), dtype=np.float32)
        got = raw.reshape(N_OUT, LANES)[:, :N_TOK].astype(np.float64)

        # Every token in channel j must equal (j+1)*scale*K -- check the whole
        # per-channel row, not just token 0, since a partial pointer-freeze
        # bug could in principle vary by token.
        for j in range(N_OUT):
            np.testing.assert_allclose(
                got[j], expected_per_channel[j], rtol=1e-4, atol=1e-3,
                err_msg=(
                    f"{App.__name__} stream {p} channel {j}: got {got[j, 0]:.6f}, "
                    f"expected {expected_per_channel[j]:.6f} (channel 0's expected "
                    f"value would be {expected_per_channel[0]:.6f} -- if got[j] == "
                    f"that instead, weight_row_off is stuck at channel 0, the exact "
                    f"pointer-persistence bug this test re-checks for)"
                ),
            )

        # Explicit, named assertion for the failure mode itself: no channel
        # past 0 may equal channel 0's expected value (which would happen if
        # the whole channel loop silently reused row 0).
        stuck_at_zero = np.isclose(got[1:], expected_per_channel[0], rtol=1e-4, atol=1e-3)
        assert not stuck_at_zero.any(), (
            f"{App.__name__} stream {p}: {stuck_at_zero.sum()} channel-token "
            f"entries past channel 0 read back as channel 0's value -- this IS "
            f"the weight_row_off pointer-persistence bug pattern"
        )
