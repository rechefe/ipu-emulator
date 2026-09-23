"""conv1x1 memory-only harness and registry declaration."""
from ipu_apps.kernels.convolutions.app import ConvApp
from ipu_apps.kernel_registry.memory import memory_spec


class App(ConvApp):
    kernel_size = 1
    single_channel = False


SPEC = memory_spec("conv2d", App, cost=1)
