"""Score gate; output contains selected rows followed by the count row (lane 0)."""
from ipu_apps.kernel_registry import Table
from ipu_apps.kernel_registry.memory import MemoryApp, MemoryLayout, ceildiv, positive_shape, memory_spec


class App(MemoryApp):
    @classmethod
    def memory_layout(cls, *, shape):
        (count,) = positive_shape(shape, 1)
        rows = ceildiv(count, 128)
        # Scores, staged sigmoid rows, tau, T*tau, T, selected, count.
        tau = 2 * rows
        output = tau + 3
        return MemoryLayout(output, rows + 1,
                            {2: 0, 3: output, 4: rows, 5: output + rows,
                             6: tau, 7: tau + 1, 8: tau + 2, 9: rows, 10: 128})


def _tables(*, shape):
    """XMEM tables as the Cache Unit sees them (see ``ipu_emu.table_trace``).

    The sigmoid rows are staged (written) and then read back: two traversals.
    """
    (count,) = shape
    rows = ceildiv(count, 128)
    return {
        "scores": Table(cr=2, rows=rows),
        "staged": Table(cr=4, rows=rows, repetitions=2),
        "tau": Table(cr=6, rows=1),
        "t_tau": Table(cr=7, rows=1),
        "temperature": Table(cr=8, rows=1),
        "selected": Table(cr=3, rows=rows),
        "count": Table(cr=5, rows=1),
    }


SPEC = memory_spec("score_threshold", App, tables=_tables)
