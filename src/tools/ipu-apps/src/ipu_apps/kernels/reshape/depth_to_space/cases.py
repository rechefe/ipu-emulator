"""Pixel shuffle fixtures with exact per-element checks."""
import numpy as np
from .app import App
from ipu_apps.kernel_registry.cases import KernelCase
from ipu_apps.kernel_registry.case_support import tile_input, prepared_image, check_close


def prepare(workspace, *, height, width, upscale_factor):
    r = upscale_factor
    params = dict(shape=(r*r, height, width), upscale_factor=r)
    layout = App.memory_layout(**params)
    x = np.arange(r*r*height*width, dtype='<f4').reshape(r*r, height, width)
    image = tile_input(x)
    expected = x.reshape(r, r, height, width).transpose(2, 0, 3, 1).reshape(height*r, width*r)
    stride = ((width + 127)//128)*128*r
    return prepared_image(workspace, params, image, layout,
                          lambda raw: check_close(raw.reshape(height*r, stride)[:, :width*r], expected))


CASES = {
    'default': KernelCase(prepare, dict(height=2, width=8, upscale_factor=2)),
    'tile_boundary': KernelCase(prepare, dict(height=2, width=129, upscale_factor=4)),
    'superpoint': KernelCase(prepare, dict(height=1, width=17, upscale_factor=8)),
    'identity': KernelCase(prepare, dict(height=1, width=129, upscale_factor=1)),
}
