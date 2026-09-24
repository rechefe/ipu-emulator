"""maxpool2d_window memory-only harness and registry declaration."""
from ipu_apps.kernels.pooling.app import WindowPoolApp
from ipu_apps.kernel_registry.memory import memory_spec


class App(WindowPoolApp):
    fixed_kernel = None


SPEC = memory_spec("maxpool2d", App, cost=1)
