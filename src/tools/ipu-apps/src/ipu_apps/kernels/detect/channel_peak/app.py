"""Channel maximum and threshold gate; output contains confidence then keep rows."""
from ipu_apps.kernel_registry import Table
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


def _tables(*, shape):
    """XMEM tables as the Cache Unit sees them (see ``ipu_emu.table_trace``).

    Input row ``c*tiles + t``; each tile walks down all channels, so the next
    tile steps back from the last channel to the first: ``(C-1)*tiles - 1``.
    """
    channels, columns = shape
    tiles = ceildiv(columns, 128)
    return {
        "input": Table(cr=2, rows=channels * tiles,
                       jump_back=max(0, (channels - 1) * tiles - 1) if tiles > 1 else 0),
        "tau": Table(cr=5, rows=1),
        "confidence": Table(cr=3, rows=tiles),
        "keep": Table(cr=4, rows=tiles),
    }


SPEC = memory_spec("channel_peak", App, tables=_tables)
