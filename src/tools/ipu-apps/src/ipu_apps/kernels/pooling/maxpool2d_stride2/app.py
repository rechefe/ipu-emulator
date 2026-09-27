"""maxpool2d_stride2 memory-only harness and registry declaration."""
from ipu_apps.kernels.pooling.app import Stride2PoolApp, stride2_spec


class App(Stride2PoolApp):
    tail = False


SPEC = stride2_spec(App, "stride2_full_pairs", "full-pairs")
