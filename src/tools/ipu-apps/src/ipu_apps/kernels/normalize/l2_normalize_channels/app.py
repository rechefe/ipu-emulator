"""L2 normalization of columns in a preformatted (channels, columns) matrix."""
from ipu_apps.kernel_registry import Table
from ipu_apps.kernel_registry.memory import MemoryApp, MemoryLayout, ceildiv, positive_shape, memory_spec


class App(MemoryApp):
    @classmethod
    def memory_layout(cls, *, shape):
        channels, columns = positive_shape(shape, 2)
        tiles = ceildiv(columns, 128)
        rows = channels * tiles
        return MemoryLayout(rows + 1, rows, {2: 0, 3: rows + 1, 4: rows, 5: tiles, 6: channels})


def _tables(*, shape):
    """XMEM tables as the Cache Unit sees them (see ``ipu_emu.table_trace``).

    Row ``c*tiles + t``. Each tile walks down all channels twice (sum of
    squares, then scale), stepping back ``(C-1)*tiles`` rows; the output is
    written once per tile, so the next tile steps back ``(C-1)*tiles - 1``.
    """
    channels, columns = shape
    tiles = ceildiv(columns, 128)
    column = (channels - 1) * tiles
    return {
        "input": Table(cr=2, rows=channels * tiles, jump_back=column),
        "norm": Table(cr=4, rows=1),
        "output": Table(cr=3, rows=channels * tiles,
                        jump_back=max(0, column - 1) if tiles > 1 else 0),
    }


SPEC = memory_spec("l2_normalize", App, tables=_tables)
