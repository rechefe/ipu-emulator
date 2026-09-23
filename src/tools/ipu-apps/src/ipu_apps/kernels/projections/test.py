"""Projection family tests, beyond each kernel's own cases and test.py.

* **Router** -- every ``proj_*_p4`` kernel is found by exactly its
  ``(k, n_out, activation)``, and no other. In particular a plain projection
  query never routes to an FFN1 kernel, whose store fuses ``silu``.
* **Activation contract** -- the activation each SPEC claims is the one its
  ``.asm`` actually applies at the store (``ACTIVATE.QUANTIZE``).
* **Weight-pointer mutation check** (slow, opt-in) -- see
  :func:`test_proj_p4_weight_row_is_not_stuck_at_channel_zero`. It assembles
  and runs 8 kernels (15-70+ min depending on load), so it is skipped unless
  ``IPU_RUN_SLOW_TESTS`` is set. Run it explicitly when touching proj_*_p4
  addressing::

      bazel test //src/tools/ipu-apps:projections_test \\
          --test_env=IPU_RUN_SLOW_TESTS=1 --test_timeout=7200
"""

from __future__ import annotations

import os
import re
from importlib import import_module
from importlib.resources import files

import numpy as np
import pytest

from ipu_apps.kernel_registry import kernel_spec, kernels, resolve
from ipu_apps.kernel_registry.cases import KernelCase, run_case
from ipu_apps.kernels.projections.app import N_STREAM, OP, ProjectionP4App
from ipu_apps.kernels.projections.cases import projection_case

KERNELS = sorted(spec.name for spec in kernels(OP))


def _kernel(name):
    """(spec, layout, activation) of a projection kernel."""
    spec = kernel_spec(name)
    return spec, spec.app_class.layout, import_module(spec.resource_package + ".app").ACTIVATION


def test_the_family_has_all_twelve_kernels():
    assert KERNELS == sorted(
        f"proj_{role}_{d}_p4"
        for role in ("qkv", "outproj", "ffn1", "ffn2") for d in (144, 192, 240)
    )


# -- router -----------------------------------------------------------------


@pytest.mark.parametrize("name", KERNELS)
def test_each_kernel_is_routed_by_its_exact_query(name):
    _, layout, activation = _kernel(name)
    verdict = resolve(OP, k=layout.k, n_out=layout.n_out, activation=activation)
    assert verdict.app_name == name, verdict.describe()
    assert verdict.alternatives == ()


@pytest.mark.parametrize("name", KERNELS)
def test_the_wrong_activation_is_refused(name):
    """An FFN1 kernel fuses silu: a plain (or default) query for its shape must
    be refused rather than silently return silu(C). Conversely, no identity
    kernel may claim a silu query."""
    _, layout, activation = _kernel(name)
    other = "none" if activation == "silu" else "silu"
    verdict = resolve(OP, k=layout.k, n_out=layout.n_out, activation=other)
    assert not verdict, verdict.describe()
    assert f"fuses activation={activation!r}" in verdict.reason
    if activation == "silu":
        assert not resolve(OP, k=layout.k, n_out=layout.n_out)   # activation defaults to "none"


def test_ffn1_kernels_are_exactly_the_silu_kernels():
    silu = {name for name in KERNELS if _kernel(name)[2] == "silu"}
    assert silu == {"proj_ffn1_144_p4", "proj_ffn1_192_p4", "proj_ffn1_240_p4"}


def test_other_stream_counts_and_unknown_activations_are_refused():
    verdict = resolve(OP, k=240, n_out=720, n_streams=1)
    assert not verdict
    assert f"exactly {N_STREAM} pixel-streams" in verdict.reason
    verdict = resolve(OP, k=240, n_out=720, activation="gelu")
    assert not verdict
    assert "activation must be one of" in verdict.reason


def test_the_family_base_harness_cannot_be_constructed():
    with pytest.raises(TypeError, match="no layout"):
        ProjectionP4App(inst_path="x", input_path="y", weights_path="z")


# -- activation contract ----------------------------------------------------


@pytest.mark.parametrize("name", KERNELS)
def test_spec_activation_matches_the_asm_store(name):
    """The SPEC's activation is declared by hand beside the .asm; check it
    against the ``ACTIVATE.QUANTIZE`` the .asm really executes, so the router
    cannot claim a computation the kernel does not do."""
    spec, _, activation = _kernel(name)
    source = files(spec.resource_package).joinpath(spec.asm).read_text()
    code = "\n".join(line.split("#", 1)[0] for line in source.splitlines())
    applied = set(re.findall(r"ACTIVATE\.QUANTIZE\s+(\w+)", code))
    assert applied == {"silu" if activation == "silu" else "identity"}, applied


# -- weight-pointer mutation check (slow, opt-in) ---------------------------

SLOW = pytest.mark.skipif(
    not os.environ.get("IPU_RUN_SLOW_TESTS"),
    reason="15-70+ min; set IPU_RUN_SLOW_TESTS=1 (bazel: --test_env=IPU_RUN_SLOW_TESTS=1)",
)

# All 8 L4/L5 proj_*_p4 kernels (QKV/OutProj/FFN1/FFN2, d in {192, 240}).
MUTATION_KERNELS = [name for name in KERNELS if not name.endswith("_144_p4")]


@SLOW
@pytest.mark.parametrize("name", MUTATION_KERNELS)
def test_proj_p4_weight_row_is_not_stuck_at_channel_zero(name, tmp_path) -> None:
    """Mutation-based check of the weight-pointer-persistence bug class.

    If a kernel resets its per-output-channel weight-row pointer
    (``weight_row_off``) in the wrong place, every output channel ``j`` reads
    output channel 0's weight row instead of its own. The 8 L4/L5 members are
    checked here BY MUTATION rather than by reading the ``.asm``.

    "Mutation" does not mean editing the kernel or ipu.py. It means an input
    on which the buggy and the correct behaviour visibly disagree: every
    output channel j gets a UNIQUE, identifiable weight row (row j filled
    with the constant (j+1)*scale, scaled small to stay in a
    well-conditioned FP32 range) and an identity-like activation (D[p] =
    ones), so C[p, j, :] should read back as EXACTLY (j+1)*scale*K for every
    j -- the value channel j's row alone determines. If the kernel silently
    reused channel 0's row for every j, every channel would instead read back
    as channel 0's value, which this test catches immediately and
    specifically -- not just "some numeric mismatch somewhere". The kernels'
    own random-W cases WOULD also catch the bug (any two distinct random rows
    differ), but do not make its signature -- "every row equals row 0" -- the
    explicit thing being checked for.
    """
    _, layout, activation = _kernel(name)
    k, n_out = layout.k, layout.n_out

    # W[j, :] = (j+1) * scale, so with D[p] = 1 everywhere
    # C[p, j, t] = sum_k W[j,k] * D[p,k,t] = (j+1) * scale * K exactly.
    scale = 0.01
    w = np.repeat(((np.arange(n_out) + 1) * scale)[:, None], k, axis=1).astype(np.float32)
    d = np.ones(layout.input_shape, dtype=np.float32)

    expected_per_channel = (np.arange(n_out, dtype=np.float64) + 1) * scale * k
    if activation == "silu":
        # FFN1 applies ACTIVATE.QUANTIZE silu at its store: the expected
        # per-channel value must go through silu too, or a perfectly correct
        # kernel misreports as "not equal" rather than signalling a real bug.
        expected_per_channel = expected_per_channel / (1.0 + np.exp(-expected_per_channel))

    def check(got):
        for p in range(N_STREAM):
            for tg in range(layout.n_tg):
                g = got[p, tg].astype(np.float64)
                # Every token in channel j must equal (j+1)*scale*K -- check
                # the whole per-channel row, not just token 0, since a partial
                # pointer-freeze bug could in principle vary by token.
                for j in range(n_out):
                    np.testing.assert_allclose(
                        g[j], expected_per_channel[j], rtol=1e-4, atol=1e-3,
                        err_msg=(
                            f"{name} stream {p} channel {j}: got {g[j, 0]:.6f}, "
                            f"expected {expected_per_channel[j]:.6f} (channel 0's expected "
                            f"value would be {expected_per_channel[0]:.6f} -- if got[j] == "
                            f"that instead, weight_row_off is stuck at channel 0, the exact "
                            f"pointer-persistence bug this test re-checks for)"
                        ),
                    )
                # Explicit, named assertion for the failure mode itself: no
                # channel past 0 may equal channel 0's expected value.
                stuck_at_zero = np.isclose(g[1:], expected_per_channel[0], rtol=1e-4, atol=1e-3)
                assert not stuck_at_zero.any(), (
                    f"{name} stream {p}: {stuck_at_zero.sum()} channel-token entries "
                    f"past channel 0 read back as channel 0's value -- this IS the "
                    f"weight_row_off pointer-persistence bug pattern"
                )

    case = KernelCase(
        lambda workspace: projection_case(workspace, layout, d, w, activation, check=check),
        max_cycles=30_000_000,
    )
    _, cycles = run_case(name, case, workspace=tmp_path)
    assert cycles > 0
