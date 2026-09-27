"""Normalization cases including all-zero columns and partial tiles."""
import numpy as np
from .app import App
from ipu_apps.kernel_registry.cases import KernelCase
from ipu_apps.kernel_registry.case_support import random_values, tile_input, tiled_case


def prepare(workspace, *, channels, columns):
    params = dict(shape=(channels, columns))
    layout = App.memory_layout(**params)
    x = random_values(params['shape'])
    x[:, 0] = 0
    norm = np.sqrt(np.sum(x*x, axis=0))
    expected = np.divide(x, norm, out=np.zeros_like(x), where=norm > 0)
    return tiled_case(workspace, params, layout, tile_input(x[:, None]), expected[:, None])


CASES = {
    'default': KernelCase(prepare, dict(channels=4, columns=16)),
    'tile_boundary': KernelCase(prepare, dict(channels=3, columns=133)),
    'single_channel': KernelCase(prepare, dict(channels=1, columns=129)),
}
