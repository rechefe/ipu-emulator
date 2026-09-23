"""maxpool2d_stride2_tail memory-only harness and registry declaration."""
from ipu_apps.kernels.pooling.app import Stride2PoolApp, stride2_spec


class App(Stride2PoolApp):
    tail = True


SPEC = stride2_spec(App, "stride2_single_half_tail", "single-half-tail",
                    " with a one-XMEM-row final input tail")
