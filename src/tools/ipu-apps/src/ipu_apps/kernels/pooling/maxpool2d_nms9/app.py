"""maxpool2d_nms9 memory-only harness and registry declaration."""

from ipu_apps.kernel_registry.memory import memory_spec
from ipu_apps.kernels.pooling.app import WindowPoolApp


class App(WindowPoolApp):
    fixed_kernel = 9


SPEC = memory_spec("maxpool2d", App, cost=0)
