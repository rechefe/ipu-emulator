"""Convolution fixture layout and independent NumPy convolution reference."""
import numpy as np
from ipu_apps.kernel_registry.cases import KernelCase
from ipu_apps.kernel_registry.case_support import random_values, tile_input, tiled_case


def make_cases(app):
    def prepare(workspace, *, channels, height, width, out_channels):
        k = app.kernel_size
        params = dict(shape=(channels, height, width), out_channels=out_channels,
                      kernel_size=k, stride=1, padding=k//2,
                      activation="none" if k == 1 else "relu")
        layout = app.memory_layout(**params)
        x = random_values(params['shape'])
        weights = random_values((out_channels, channels, k, k)) / 4
        bias = random_values((out_channels,))
        group_size = 128 if k == 1 else 14
        groups = (channels + group_size - 1) // group_size

        def weights_and_bias(image):
            for o in range(out_channels):
                for g in range(groups):
                    chunk = weights[o, g*group_size:(g+1)*group_size].ravel()
                    image[layout.crs[4] + o*groups + g, :len(chunk)] = chunk
                image[layout.crs[5] + o, 0] = bias[o]

        padded = np.pad(x, ((0, 0), (k//2,)*2, (k//2,)*2))
        windows = np.lib.stride_tricks.sliding_window_view(padded, (k, k), axis=(1, 2))
        expected = np.einsum('cyxij,ocij->oyx', windows, weights) + bias[:, None, None]
        if k == 3:
            expected = np.maximum(expected, 0)
        return tiled_case(workspace, params, layout,
                          tile_input(x, halo=k//2, guard=not app.single_channel), expected,
                          columns=128 if k == 1 else 126, constants=weights_and_bias)

    def case(c, h, w, o):
        return KernelCase(prepare, dict(channels=c, height=h, width=w, out_channels=o))

    cases = {'default': case(1 if app.single_channel else 3, 3, 8, 2),
             'tile_boundary': case(1 if app.single_channel else 2, 2, 133, 2),
             'single_pixel': case(1, 1, 1, 1)}
    if not app.single_channel:
        cap = 128 if app.kernel_size == 1 else 14
        cases['full_group'] = case(cap, 1, 2, 1)
        cases['partial_group'] = case(cap + 1, 1, 2, 2)
    return cases
