"""Shared unfold/fold cases: seeded FP32 tensors checked against NumPy layouts.

Unfold and fold are pure data movement (the multiply is by 1.0), so every
reference here is an indexing expression, not arithmetic -- any mismatch is a
real layout bug. The two layouts, both one XMEM row of ``LANES`` FP32 values
per row:

* **striped spatial** (:func:`stripe_input`): unfold's input, and fold's
  output. Row ``(stripe, ch)`` holds channel ``ch``'s ``H / N_STRIPES``
  spatial rows x ``W`` columns, flattened row-major. (unfold_8x8x240's input
  additionally permutes spatial rows -- see its ``pack_input_rows`` -- while
  fold_8x8x240's output is this plain layout with one stripe.)
* **stream rows** (:func:`stream_rows`): unfold's output, and fold's input.
  The four streams are a stride-2 space-to-depth decimation, NOT four
  contiguous quadrants: stream ``s`` takes every other row and column at
  phase ``(s // 2, s % 2)``, i.e. the standard stride-2 convolution
  decomposition (TL/TR/BL/BR name the phase within each 2x2 block, not a
  corner of the image). Each channel's decimated tokens, row-major, fill
  ``ceil(tokens / LANES)`` rows; a row that is not full carries stale lanes
  after its valid prefix.
"""
import numpy as np

from ipu_emu.ipu import LANES

from ipu_apps.kernel_registry.cases import KernelCase, PreparedCase

N_STREAMS = 4
MAX_CYCLES = 20_000_000


def spatial_tensor(seed, c, h, w):
    """A seeded ``[C, H, W]`` FP32 tensor, uniform in [-1, 1)."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-1.0, 1.0, size=(c, h, w)).astype(np.float32)


def stripe_input(x, n_stripes, fill=0.0):
    """[C, H, W] -> NHCW-striped rows ``[N_STRIPES * C, LANES]``."""
    c, h, w = x.shape
    stripe_h = h // n_stripes
    src = np.full((n_stripes * c, LANES), fill, dtype=np.float32)
    for stripe in range(n_stripes):
        block = x[:, stripe * stripe_h:(stripe + 1) * stripe_h, :]
        src[stripe * c:(stripe + 1) * c, :stripe_h * w] = block.reshape(c, -1)
    return src


def stream_tokens(x):
    """Tokens per stream row: a channel's decimated grid, capped at one row."""
    _, h, w = x.shape
    return min((h // 2) * (w // 2), LANES)


def stream_rows(x, fill=0.0):
    """[C, H, W] -> unfold's raw output rows ``[N_STREAMS, C * rows_per_ch, LANES]``.

    Lanes after each row's valid prefix hold ``fill``.
    """
    c = x.shape[0]
    per_row = stream_tokens(x)
    streams = [x[:, s // 2::2, s % 2::2].reshape(c, -1) for s in range(N_STREAMS)]
    rows = streams[0].shape[1] // per_row
    out = np.full((N_STREAMS, c * rows, LANES), fill, dtype=np.float32)
    for s, tokens in enumerate(streams):
        out[s, :, :per_row] = tokens.reshape(c * rows, per_row)
    return out


def _read(path, size):
    raw = np.fromfile(path, dtype="<f4")
    if raw.size != size:
        raise ValueError(f"output has {raw.size} FP32 values, expected {size}")
    return raw


def check_streams(out_path, x, *, row_width=LANES, rtol=1e-6, atol=1e-6):
    """Compare unfold's output (rows ``row_width`` wide) with :func:`stream_rows`."""
    want = stream_rows(x)
    per_row = stream_tokens(x)
    got = _read(out_path, want.shape[0] * want.shape[1] * row_width)
    got = got.reshape(want.shape[0], want.shape[1], row_width)
    for s in range(N_STREAMS):
        np.testing.assert_allclose(
            got[s, :, :per_row], want[s, :, :per_row], rtol=rtol, atol=atol,
            err_msg=f"unfold stream {s} (phase {s // 2},{s % 2}) mismatch",
        )


def check_striped(out_path, x, n_stripes, *, rtol=1e-6, atol=1e-6):
    """Compare fold's output with :func:`stripe_input`, over each row's valid lanes."""
    c, h, w = x.shape
    valid = (h // n_stripes) * w
    got = _read(out_path, n_stripes * c * LANES).reshape(n_stripes * c, LANES)
    np.testing.assert_allclose(
        got[:, :valid], stripe_input(x, n_stripes)[:, :valid], rtol=rtol, atol=atol,
        err_msg="fold output != original spatial tensor",
    )


def unfold_cases(app, *, seed):
    """``CASES`` for unfold_32x32x144 / unfold_16x16x192 (plain striped input)."""

    def prepare(workspace, *, seed):
        x = spatial_tensor(seed, app.C, app.H, app.W)
        inp, out = workspace / "input.bin", workspace / "output.bin"
        inp.write_bytes(stripe_input(x, app.N_STRIPES).tobytes())
        return PreparedCase({"shape": (app.H, app.W, app.C)},
                            {"input_path": inp, "output_path": out},
                            lambda: check_streams(out, x))

    return {"default": KernelCase(prepare, {"seed": seed}, MAX_CYCLES)}


def fold_cases(app, *, seed):
    """``CASES`` for a fold kernel: hand-built streams, no unfold dependency.

    The streams are built directly with :func:`stream_rows`; every lane after
    a row's valid prefix is NaN, so a fold that ever read unfold's stale
    padding lanes would fail here.
    """
    n_stripes = getattr(app, "N_STRIPES", 1)

    def prepare(workspace, *, seed):
        x = spatial_tensor(seed, app.C, app.H, app.W)
        inp, out = workspace / "input.bin", workspace / "output.bin"
        inp.write_bytes(stream_rows(x, fill=np.nan).tobytes())
        return PreparedCase({"shape": (app.H, app.W, app.C)},
                            {"input_path": inp, "output_path": out},
                            lambda: check_striped(out, x, n_stripes))

    return {"default": KernelCase(prepare, {"seed": seed}, MAX_CYCLES)}
