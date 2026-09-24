"""Cases shared by the max-pool kernels."""
import numpy as np
from ipu_apps.kernel_registry.cases import KernelCase
from ipu_apps.kernel_registry.case_support import random_values, tile_input, tiled_case
from ipu_apps.kernels.pooling.app import Stride2PoolApp


def make_cases(app):
    if issubclass(app, Stride2PoolApp):
        return _stride2_cases(app)
    return _window_cases(app)


def _window_cases(app):
    def prepare(workspace, *, channels, height, width, kernel_size):
        params = dict(shape=(channels, height, width), kernel_size=kernel_size,
                      stride=1, padding=kernel_size // 2)
        layout = app.memory_layout(**params)
        x = random_values(params['shape']) - 2
        floor = np.finfo(np.float32).min
        padded = np.pad(x, ((0, 0), (kernel_size//2,)*2, (kernel_size//2,)*2),
                        constant_values=floor)
        windows = np.lib.stride_tricks.sliding_window_view(padded, (kernel_size, kernel_size), axis=(1, 2))
        expected = windows.max(axis=(-1, -2))
        return tiled_case(workspace, params, layout,
                          tile_input(x, halo=kernel_size // 2, fill=floor), expected,
                          columns=129 - kernel_size, fill=floor)

    def case(c, h, w, k=app.fixed_kernel or 3):
        return KernelCase(prepare, dict(channels=c, height=h, width=w, kernel_size=k))

    cases = {'default': case(1, 4, 16), 'tile_boundary': case(2, 3, 133),
             'single_pixel': case(1, 1, 1)}
    if app.fixed_kernel is None:
        cases.update(identity_window=case(2, 2, 129, 1), large_window=case(1, 1, 3, 127),
                     nms7=case(1, 3, 125, 7), nms9=case(1, 3, 123, 9))
    return cases


def _stride2_cases(app):
    def prepare(workspace, *, channels, height, width):
        params = dict(shape=(channels, height, width), kernel_size=2, stride=2, padding=0)
        layout = app.memory_layout(**params)
        x = random_values(params['shape'])
        oh, ow = height // 2, width // 2
        expected = x[:, :2*oh, :2*ow].reshape(channels, oh, 2, ow, 2).max(axis=(2, 4))
        return tiled_case(workspace, params, layout,
                          tile_input(x, fill=np.finfo(np.float32).min), expected)

    def case(c, h, w):
        return KernelCase(prepare, dict(channels=c, height=h, width=w))

    return {'default': case(1, 4, 16 if app.tail else 256),
            'tile_boundary': case(2, 5, 260 if app.tail else 511)}
