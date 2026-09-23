"""Channel maximum and threshold gate; output contains confidence then keep rows."""
from ipu_apps.kernel_registry.memory import MemoryApp, MemoryLayout, ceildiv, positive_shape, memory_spec


class App(MemoryApp):
    @classmethod
    def memory_layout(cls, *, shape):
        channels, columns = positive_shape(shape, 2)
        tiles = ceildiv(columns, 128)
        tau = channels * tiles
        output = tau + 1
        return MemoryLayout(output, 2 * tiles,
                            {2: 0, 3: output, 4: output + tiles, 5: tau,
                             6: tiles, 7: channels, 10: 128})


SPEC = memory_spec("channel_peak", App)
