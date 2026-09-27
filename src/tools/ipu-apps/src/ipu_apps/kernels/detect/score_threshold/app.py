"""Score gate; output contains selected rows followed by the count row (lane 0)."""
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


SPEC = memory_spec("score_threshold", App)
