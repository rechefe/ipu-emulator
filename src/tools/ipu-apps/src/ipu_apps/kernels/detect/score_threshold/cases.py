"""Score gates and soft counts with suppressed padding lanes."""
import numpy as np
from .app import App
from ipu_apps.kernel_registry.cases import KernelCase
from ipu_apps.kernel_registry.case_support import random_values, prepared_image, check_close


def prepare(workspace, *, count, threshold, temperature):
    if not np.isfinite(threshold) or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('threshold must be finite and temperature must be finite and positive')
    params = dict(shape=(count,))
    layout = App.memory_layout(**params)
    rows = layout.crs[9]
    x = random_values((count,))
    tau, t = np.float32(threshold), np.float32(temperature)
    image = np.zeros((layout.input_rows, 128), dtype='<f4')
    image[:rows] = tau - 800 / t
    image[:rows].reshape(-1)[:count] = x
    image[layout.crs[6]] = tau
    image[layout.crs[7]] = t * tau
    image[layout.crs[8]] = t
    expected_gate = np.maximum(x - tau, 0)
    z = (t*x - t*tau).astype(np.float64)
    expected_count = np.exp(-np.logaddexp(0, -z)).sum()

    def check(raw):
        check_close(raw[:count], expected_gate)
        check_close(raw[count:rows*128], 0)
        check_close(raw[rows*128], expected_count)

    return prepared_image(workspace, params, image, layout, check)


CASES = {
    'default': KernelCase(prepare, dict(count=16, threshold=0.5, temperature=4.0)),
    'tile_boundary': KernelCase(prepare, dict(count=259, threshold=-0.25, temperature=2.0)),
    'single_score': KernelCase(prepare, dict(count=1, threshold=0.0, temperature=1.0)),
}
