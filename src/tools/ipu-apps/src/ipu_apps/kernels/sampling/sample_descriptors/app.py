"""sample_descriptors memory-only harness and registry declaration."""
from ipu_apps.kernels.sampling.app import DescriptorApp
from ipu_apps.kernel_registry.memory import memory_spec


class App(DescriptorApp):
    separable = False


SPEC = memory_spec("sample_descriptors", App)
