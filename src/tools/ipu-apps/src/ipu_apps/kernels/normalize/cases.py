"""Shared normalize cases: random FP32 layernorm inputs checked against a NumPy reference.

A layernorm kernel's ``cases.py`` is one call, :func:`layernorm_cases`,
passing its app module and its input recipe. The
module's constants give the geometry: ``N_CH`` channels, ``N_TG`` token groups
(default 1) and ``N_TPG`` or ``N_TOK`` tokens per group.

File layout (what every layernorm harness reads): the input is
``N_CH * N_TG`` FP32 rows of ``LANES`` lanes in ``(ch, tg)`` order, tokens in
the first lanes and zero padding after; gamma/beta are ``N_CH`` FP32 values
(``pad_params`` writes them zero-padded to whole 128-lane rows instead, as
``layernorm_128x16`` stores them verbatim as a row).

Reference: ``output[ch, i] = gamma[ch] * (x[ch,i] - mu[i]) / sigma[i] + beta[ch]``
where mu/sigma reduce over the CHANNEL axis, independently per token. No
epsilon -- this matches the rsqrt activation, which maps a non-positive
variance to 0 rather than to an infinity.
"""
import numpy as np

from ipu_emu.ipu import LANES

from ipu_apps.kernel_registry.cases import KernelCase, PreparedCase


def reference_layernorm(x, gamma, beta):
    """LayerNorm over axis 0 (channels) of ``x``; gamma/beta are ``[N_CH]``."""
    mean = x.mean(axis=0)
    centered = x - mean
    var = (centered ** 2).mean(axis=0)
    inv_std = np.where(var > 0.0, 1.0 / np.sqrt(var), 0.0).astype(np.float32)
    normalized = centered * inv_std
    bcast = (slice(None),) + (None,) * (x.ndim - 1)
    return (gamma[bcast] * normalized + beta[bcast]).astype(np.float32)


def randn_inputs(rng, channels, x_shape):
    """Normal x, gamma near 1, small beta (layernorm_128x16, layernorm_256x144)."""
    x = rng.randn(*x_shape).astype(np.float32)
    gamma = rng.randn(channels).astype(np.float32) * 0.5 + 1.0  # near 1
    beta = rng.randn(channels).astype(np.float32) * 0.1
    return x, gamma, beta


def uniform_inputs(rng, channels, x_shape):
    """Uniform x in [-4, 4), gamma in [0.5, 1.5), beta in [-0.5, 0.5) (layernorm_16x240, layernorm_64x192)."""
    x = rng.uniform(-4.0, 4.0, size=x_shape).astype(np.float32)
    gamma = rng.uniform(0.5, 1.5, size=channels).astype(np.float32)
    beta = rng.uniform(-0.5, 0.5, size=channels).astype(np.float32)
    return x, gamma, beta


def _fp32_rows(values):
    """Zero-pad a 1-D FP32 array to whole ``LANES``-lane rows."""
    rows = -(-len(values) // LANES)
    padded = np.zeros(rows * LANES, dtype=np.float32)
    padded[: len(values)] = values
    return padded


def check_not_degenerate(actual, expected, x, gamma, beta):
    """Guards against a degenerate pass.

    A kernel that emitted all zeros, or passed its input straight through,
    could not satisfy these. ``actual``/``x`` are ``[N_CH, tokens]``.
    """
    # 1. The output actually varies, and varies the way the reference does.
    if not actual.std() > 0.1:
        raise AssertionError("output is (near) constant -- degenerate result")
    np.testing.assert_allclose(actual.std(), expected.std(), rtol=1e-3)

    # 2. Real LayerNorm centering: undo gamma/beta and the per-channel scale,
    #    then the mean over CHANNELS must be ~0 for every token, and the
    #    standard deviation over channels must be ~1.
    de_scaled = (actual - beta[:, None]) / gamma[:, None]
    tokens = actual.shape[1]
    np.testing.assert_allclose(
        de_scaled.mean(axis=0), np.zeros(tokens), atol=1e-4,
        err_msg="de-scaled output is not centred over the channel axis",
    )
    np.testing.assert_allclose(
        de_scaled.std(axis=0), np.ones(tokens), rtol=1e-3,
        err_msg="de-scaled output does not have unit variance over channels",
    )

    # 3. The output is genuinely different from the input, so a pass-through
    #    of x could not have produced it.
    if np.allclose(actual, x, rtol=1e-2, atol=1e-2):
        raise AssertionError("output equals the input -- the kernel is a pass-through")


def layernorm_cases(app, *, inputs, seed, rtol, atol, pad_params=False,
                    cropped_output=False, guards=False, max_cycles=5_000_000):
    """``CASES`` for the layernorm kernel whose harness module is ``app``.

    Args:
        inputs:         :func:`randn_inputs` or :func:`uniform_inputs`.
        seed:           Default ``RandomState`` seed.
        rtol, atol:     Tolerance of the reference comparison.
        pad_params:     Write gamma/beta zero-padded to whole rows.
        cropped_output: The harness crops each output row to its valid tokens.
        guards:         Also run :func:`check_not_degenerate`.
    """
    channels = app.N_CH
    groups = getattr(app, "N_TG", 1)
    per_group = getattr(app, "N_TPG", None) or app.N_TOK
    tokens = groups * per_group

    def prepare(workspace, *, seed):
        rng = np.random.RandomState(seed)
        x, gamma, beta = inputs(rng, channels, (channels, groups, per_group))
        rows = np.zeros((channels, groups, LANES), dtype=np.float32)
        rows[..., :per_group] = x

        inp, gamma_path, beta_path, out = (
            workspace / name for name in
            ("input_x_fp32.bin", "gamma_fp32.bin", "beta_fp32.bin", "output.bin"))
        inp.write_bytes(rows.tobytes())
        gamma_path.write_bytes((_fp32_rows(gamma) if pad_params else gamma).tobytes())
        beta_path.write_bytes((_fp32_rows(beta) if pad_params else beta).tobytes())
        expected = reference_layernorm(x, gamma, beta)          # [N_CH, N_TG, per_group]

        def check():
            raw = np.fromfile(out, dtype="<f4")
            lanes = per_group if cropped_output else LANES
            want = channels * groups * lanes
            if raw.size != want:
                raise ValueError(f"output has {raw.size} FP32 values, expected {want}")
            got = raw.reshape(channels, groups, lanes)[..., :per_group]
            np.testing.assert_allclose(
                got, expected, rtol=rtol, atol=atol,
                err_msg=f"{app.__package__.rpartition('.')[2]} output does not match reference")
            if guards:
                check_not_degenerate(got.reshape(channels, tokens),
                                     expected.reshape(channels, tokens),
                                     x.reshape(channels, tokens), gamma, beta)

        params = {"shape": (channels, tokens)}
        return PreparedCase(params, {"input_path": inp, "gamma_path": gamma_path,
                                     "beta_path": beta_path, "output_path": out}, check)

    return {"default": KernelCase(prepare, {"seed": seed}, max_cycles)}
