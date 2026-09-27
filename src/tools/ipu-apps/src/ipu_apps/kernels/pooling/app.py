"""Memory geometry shared by the max-pool kernels."""
from ipu_emu.ipu import LANES, R_ACC_SIZE
from ipu_emu.xmem import XMEM_SIZE_BYTES

from ipu_apps.kernel_registry import OUTPUT, ShapeBundle
from ipu_apps.kernel_registry.memory import (
    MemoryApp, MemoryLayout, ceildiv, memory_spec, positive_shape,
)


class WindowPoolApp(MemoryApp):
    """Centred, stride-1 max-pool; ``fixed_kernel`` pins an unrolled window."""
    fixed_kernel = None

    @classmethod
    def memory_layout(cls, *, shape, kernel_size, stride, padding):
        c, h, w = positive_shape(shape, 3)
        k = kernel_size
        if type(k) is not int or k < 1 or k > 127 or k % 2 != 1:
            raise ValueError("kernel_size must be odd and between 1 and 127")
        if stride != 1 or padding != k // 2:
            raise ValueError("requires stride=1 and padding=kernel_size//2")
        if cls.fixed_kernel is not None and k != cls.fixed_kernel:
            raise ValueError(f"requires kernel_size={cls.fixed_kernel}")
        tiles = ceildiv(w, 129 - k)
        plane = (h + k - 1) * tiles
        rows = c * plane
        general = cls.fixed_kernel is None
        output = rows + int(general)
        crs = {2: 0, 3: output, 4: rows if general else 128,
               5: tiles, 6: plane, 7: h, 8: c}
        if general:
            crs.update({9: k, 10: 384, 11: k - 1})
        return MemoryLayout(output, c * h * tiles, crs)


STRIDE2_SCRATCH_ROWS = 2


class Stride2PoolApp(MemoryApp):
    """2x2, stride-2 FP32 max-pool.

    ``input_path`` must already contain the kernel's XMEM rows: channel-major,
    then spatial-row-major, ``ceil(width / LANES)`` XMEM rows per spatial row,
    real columns first and ``-FLT_MAX`` padding in the final XMEM row. The
    output file is the raw tiled result: ``ceil((width // 2) / LANES)`` XMEM rows
    per output row. Two scratch rows follow the output (CR4).

    ``tail`` selects the kernel for a final output XMEM row backed by only one
    input XMEM row; exactly one of the two kernels accepts any given shape.
    """
    tail = False

    @classmethod
    def memory_layout(cls, *, shape, kernel_size, stride, padding):
        channels, height, width = positive_shape(shape, 3)
        if height < 2 or width < 2:
            raise ValueError(f"a 2x2 window does not fit shape {tuple(shape)}")
        if kernel_size != 2 or stride != 2:
            raise ValueError(f"implements kernel_size=2, stride=2; got "
                             f"kernel_size={kernel_size}, stride={stride}")
        if padding != 0:
            raise ValueError(f"implements padding=0; got padding={padding}")
        in_tiles, out_tiles = ceildiv(width, LANES), ceildiv(width // 2, LANES)
        if (in_tiles < 2 * out_tiles) != cls.tail:
            raise ValueError("every output XMEM row has two input XMEM rows" if cls.tail
                             else "the final output XMEM row has only one input XMEM row")
        input_rows = channels * height * in_tiles
        output_rows = channels * (height // 2) * out_tiles
        scratch = input_rows + output_rows
        if (scratch + STRIDE2_SCRATCH_ROWS) * R_ACC_SIZE > XMEM_SIZE_BYTES:
            raise ValueError(f"memory image needs {scratch + STRIDE2_SCRATCH_ROWS} XMEM rows, "
                             f"exceeding the IPU capacity of {XMEM_SIZE_BYTES // R_ACC_SIZE} rows")
        crs = {2: 0, 3: input_rows, 4: scratch, 5: in_tiles, 6: 2 * in_tiles,
               7: out_tiles, 8: height // 2, 9: channels, 10: LANES, 11: height * in_tiles}
        if cls.tail:
            crs[12] = out_tiles - 1
        return MemoryLayout(input_rows, output_rows, crs)


def _stride2_output_shape(params):
    channels, height, width = params["shape"]
    return (channels, height // 2, width // 2)


def stride2_spec(app_class, variant, tag, detail=""):
    """SPEC for a :class:`Stride2PoolApp` kernel."""
    return memory_spec(
        "maxpool2d", app_class,
        variant=variant,
        explain=lambda **p: (f"a 2x2 stride-2 max-pool maps {tuple(p['shape'])} "
                             f"to {_stride2_output_shape(p)}{detail}"),
        bundle=lambda **p: ShapeBundle.of(input=tuple(p["shape"])).with_shapes(
            derived={OUTPUT: _stride2_output_shape(p)}),
        caveats=lambda **p: ("input and output files use the kernel's raw tiled XMEM layout",),
        tags=("fp32-wide", "strided", tag),
    )
