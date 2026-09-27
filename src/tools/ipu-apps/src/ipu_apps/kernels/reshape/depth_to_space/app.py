"""Single-output-channel pixel shuffle; output rows retain input-tile padding."""
from ipu_apps.kernel_registry.memory import MemoryApp, MemoryLayout, ceildiv, positive_shape, memory_spec


def packed_bytes(values):
    return int.from_bytes(bytes(values), "little")


class App(MemoryApp):
    @classmethod
    def memory_layout(cls, *, shape, upscale_factor):
        channels, h, w = positive_shape(shape, 3)
        r = upscale_factor
        if type(r) is not int or r not in (1, 2, 4, 8) or channels != r * r:
            raise ValueError("requires upscale_factor in (1, 2, 4, 8) and channels=upscale_factor**2")
        tiles = ceildiv(w, 128)
        plane = h * tiles
        rows = channels * plane
        return MemoryLayout(rows, rows,
                            {2: 0, 3: rows, 4: plane, 5: r * plane, 6: tiles,
                             7: packed_bytes(range(4*r, 8*r, r)), 8: r, 9: 128 // r,
                             10: 8*r, 11: 16 // r - 1,
                             12: packed_bytes(range(4)), 13: packed_bytes(range(4, 8)),
                             14: packed_bytes(range(0, 4*r, r))})


SPEC = memory_spec("depth_to_space", App)
