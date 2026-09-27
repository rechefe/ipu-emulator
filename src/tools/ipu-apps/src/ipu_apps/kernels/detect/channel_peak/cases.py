"""Channel maxima and threshold checks, including the single-channel branch."""
import numpy as np
from .app import App
from ipu_apps.kernel_registry.cases import KernelCase
from ipu_apps.kernel_registry.case_support import random_values, tile_input, tiled_case


def prepare(workspace, *, channels, columns, threshold):
    if not np.isfinite(threshold):
        raise ValueError('threshold must be finite')
    params = dict(shape=(channels, columns))
    layout = App.memory_layout(**params)
    x = random_values(params['shape'])
    confidence = x.max(axis=0)
    expected = np.stack((confidence, np.maximum(confidence - np.float32(threshold), 0)))
    def threshold_row(image):
        image[-1] = threshold

    return tiled_case(workspace, params, layout, tile_input(x[:, None]), expected[:, None],
                      constants=threshold_row)


CASES = {
    'default': KernelCase(prepare, dict(channels=4, columns=16, threshold=0.5)),
    'tile_boundary': KernelCase(prepare, dict(channels=3, columns=133, threshold=-0.25)),
    'single_channel': KernelCase(prepare, dict(channels=1, columns=129, threshold=0.0)),
}
