"""Descriptor fixtures, input packing, and row-wise checks for the registry runner.

``reference_rows`` is an independent four-corner reference used only for
validation; the kernel never uses it. Builders take the kernel's harness class
(a :class:`~ipu_apps.kernels.sampling.app.DescriptorApp`).
"""
from functools import partial

import numpy as np

from ipu_apps.kernel_registry.cases import KernelCase
from ipu_apps.kernel_registry.case_support import prepared_image
from ipu_apps.kernel_registry.memory import positive_shape


def precompute_axis_weights(shape=(60, 80, 256)):
    """Stock-coordinate (Ky[H,3,8], Kx[W,3,8]), before any XY products."""
    h, w, c = positive_shape(shape, 3)
    if c != 256:
        raise ValueError("expected HWC with 256 channels")
    offsets = np.arange(-1, 2, dtype=np.float32)

    def axis(n):
        p = np.arange(8 * n, dtype=np.float32) - np.float32(3.5)
        p /= np.float32(8 * n - 4.5)
        p = p * np.float32(2) - np.float32(1)
        p = (p + np.float32(1)) / np.float32(2) * np.float32(n - 1)
        delta = p.reshape(n, 8)[:, None, :] - np.arange(n, dtype=np.float32)[:, None, None]
        return np.maximum(np.float32(0), np.float32(1) - np.abs(delta - offsets[None, :, None]))

    return axis(h), axis(w)


def precompute_weights(shape=(60, 80, 256), mode="shared"):
    """Return K[dy,dx,a,b] or K[i,j,dy,dx,a,b], with offsets dy/dx - 1.

    Stock mode follows the FP32 grid construction and align_corners=True
    unnormalization in third_party/superglue/models/superpoint.py. Only the
    geometry is evaluated here; descriptors never participate in preparation.
    """
    h, w, c = positive_shape(shape, 3)
    if c != 256 or mode not in ("shared", "stock"):
        raise ValueError("expected HWC with 256 channels and shared/stock mode")
    offsets = np.arange(-1, 2, dtype=np.float32)
    if mode == "shared":
        t = (np.arange(8, dtype=np.float32) - np.float32(3.5)) / np.float32(8)
        k = np.maximum(np.float32(0), np.float32(1) - np.abs(t[None, :] - offsets[:, None]))
        return k[:, None, :, None] * k[None, :, None, :]

    ky, kx = precompute_axis_weights(shape)
    return ky[:, None, :, None, :, None] * kx[None, :, None, :, None, :]


def pack_input(descriptors, app, *, mode="shared", weights=None,
               cell_row_start=0, cell_row_count=None):
    """Return (preformatted FP32 XMEM input rows, MemoryLayout) for ``app``.

    Caller-supplied coefficients are supported in shared mode. Halo packing
    retains the whole coarse map even when a sub-band is requested.
    A separable ``app`` (the compact stock kernel) requires mode="stock".
    """
    descriptors = np.asarray(descriptors, dtype='<f4')
    shape = descriptors.shape
    separable = app.separable
    layout = app.memory_layout(shape=shape, mode=mode,
                              cell_row_start=cell_row_start, cell_row_count=cell_row_count)
    if not np.isfinite(descriptors).all():
        raise ValueError("descriptors must be finite")
    if weights is not None and mode != "shared":
        raise ValueError("caller-provided weights require shared mode")
    h, w, _ = shape
    descriptor_rows = (h + 2) * (w + 2) * 2
    image = np.zeros((layout.input_rows, 128), dtype='<f4')
    image[:descriptor_rows].reshape(h + 2, w + 2, 256)[1:-1, 1:-1] = descriptors
    if separable:
        ky, kx = precompute_axis_weights(shape)
        image[descriptor_rows:descriptor_rows+h, :24] = ky.reshape(h, 24)
        image[descriptor_rows+h:descriptor_rows+h+w, :24] = kx.reshape(w, 24)
    else:
        if weights is None:
            weights = precompute_weights(shape, mode)
        weights = np.asarray(weights, dtype='<f4')
        expected = (3, 3, 8, 8) if mode == "shared" else (*shape[:2], 3, 3, 8, 8)
        if weights.shape != expected or not np.isfinite(weights).all():
            raise ValueError(f"weights must be finite with shape {expected}")
        image[descriptor_rows:, :64] = weights.reshape(-1, 64)
    return image, layout


def reference_rows(descriptors, mode, y_start=0, y_count=None):
    h, w, _ = descriptors.shape
    if y_count is None:
        y_count = h * 8 - y_start
    x = np.arange(w * 8, dtype=np.float32)
    y = np.arange(y_start, y_start + y_count, dtype=np.float32)
    if mode == "shared":
        x = (x - 3.5) / 8
        y = (y - 3.5) / 8
    elif mode == "stock":
        # Deliberately construct the actual normalized grid, independently of
        # the phase/tap coefficient representation used by pack_input.
        grid_x = ((x - 3.5) / np.float32(w * 8 - 4.5)) * 2 - 1
        grid_y = ((y - 3.5) / np.float32(h * 8 - 4.5)) * 2 - 1
        x = ((grid_x + 1) / 2) * (w - 1)
        y = ((grid_y + 1) / 2) * (h - 1)
    else:
        raise ValueError(mode)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    result = np.zeros((len(y), len(x), 256), dtype=np.float32)
    for dy, wy in ((0, 1 - fy), (1, fy)):
        for dx, wx in ((0, 1 - fx), (1, fx)):
            ix, iy = x0 + dx, y0 + dy
            valid = ((iy[:, None] >= 0) & (iy[:, None] < h)
                     & (ix[None, :] >= 0) & (ix[None, :] < w))
            weight = (wy[:, None] * wx[None, :] * valid).astype(np.float32)
            values = descriptors[np.clip(iy, 0, h - 1)[:, None], np.clip(ix, 0, w - 1)[None, :]]
            result += values * weight[..., None]
    return result


def prepare(app, workspace, *, height=2, width=3, mode="shared", amplitude=1.0,
            start=0, count=0, compare_stock=False):
    if not np.isfinite(amplitude) or not 0 < amplitude <= np.finfo(np.float32).max:
        raise ValueError("amplitude must be positive, finite, and representable in FP32")
    values = np.random.default_rng(31).uniform(-amplitude, amplitude, (height, width, 256)).astype('<f4')
    params = dict(shape=values.shape, mode=mode, cell_row_start=start,
                  cell_row_count=count or None)
    image, layout = pack_input(values, app, mode=mode,
                              cell_row_start=start, cell_row_count=count or None)

    def check(raw):
        actual = raw.reshape(-1, width * 8, 256)
        error = stock_max = absolute_sum = squared_sum = 0.0
        for row, output in enumerate(actual):
            y = start * 8 + row
            expected = reference_rows(values, mode, y, 1)[0]
            np.testing.assert_allclose(output, expected, atol=1e-5 * amplitude, rtol=1e-5)
            error = max(error, float(np.max(np.abs(output - expected))))
            if compare_stock:
                difference = output.astype(np.float64) - reference_rows(values, "stock", y, 1)[0]
                stock_max = max(stock_max, float(np.max(np.abs(difference))))
                absolute_sum += float(np.sum(np.abs(difference)))
                squared_sum += float(np.sum(difference * difference))
        print(f"{mode}: shape={actual.shape}, {actual.nbytes} bytes, seed=31, "
              f"uniform[-{amplitude:g},{amplitude:g}], max_abs_error={error:.9g}")
        if compare_stock:
            print(f"{mode} vs stock reference: max_abs_error={stock_max:.9g}, "
                  f"mean_abs_error={absolute_sum / actual.size:.9g}, "
                  f"rmse={np.sqrt(squared_sum / actual.size):.9g}")

    return prepared_image(workspace, params, image, layout, check)


DEFAULTS = dict(height=2, width=3, mode="shared", amplitude=1.0,
                start=0, count=0, compare_stock=False)


def make_cases(app):
    run = partial(prepare, app)
    if not app.separable:
        return {
            "default": KernelCase(run, DEFAULTS),
            "stock": KernelCase(run, dict(DEFAULTS, mode="stock")),
            "single_cell": KernelCase(run, dict(DEFAULTS, height=1, width=1)),
            "stock_single_cell": KernelCase(run, dict(DEFAULTS, height=1, width=1, mode="stock")),
        }
    return {
        name: KernelCase(run, dict(DEFAULTS, mode="stock", **options))
        for name, options in {
            "default": {},
            "single_cell": dict(height=1, width=1),
            "single_row": dict(height=1, width=5),
            "single_column": dict(height=5, width=1),
            "interior_band": dict(height=4, width=3, start=1, count=2),
            "last_band": dict(height=3, width=2, start=2, count=1),
            "large_values": dict(amplitude=100.0),
        }.items()
    }
