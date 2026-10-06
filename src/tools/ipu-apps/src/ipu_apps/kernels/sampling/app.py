"""Dense descriptor interpolation in HWC order; no L2 normalization."""

from ipu_apps.kernel_registry import Table
from ipu_apps.kernel_registry.memory import MemoryApp, MemoryLayout, positive_shape


class DescriptorApp(MemoryApp):
    """``separable`` selects the compact stock-mode kernel's layout."""
    separable = False

    @classmethod
    def memory_layout(cls, *, shape, mode="shared", cell_row_start=0, cell_row_count=None):
        h, w, c = positive_shape(shape, 3)
        if c != 256:
            raise ValueError("sample_descriptors requires 256 channels in HWC order")
        if mode not in ("shared", "stock"):
            raise ValueError("mode must be 'shared' or 'stock'")
        if cls.separable and mode != "stock":
            raise ValueError("separable interpolation requires stock mode")
        if cell_row_count is None:
            cell_row_count = h - cell_row_start
        if (type(cell_row_start) is not int or type(cell_row_count) is not int
                or cell_row_start < 0 or cell_row_count < 1
                or cell_row_start + cell_row_count > h):
            raise ValueError("coarse-row band must be nonempty and inside the input")
        stride = (w + 2) * 2
        descriptor_rows = (h + 2) * stride
        if cls.separable:
            scratch_base = descriptor_rows + h + w
            input_rows = scratch_base + 8 * stride
            return MemoryLayout(input_rows, cell_row_count * w * 64 * 2, {
                2: cell_row_start * stride, 3: input_rows,
                4: descriptor_rows + cell_row_start, 5: descriptor_rows + h,
                6: scratch_base, 7: stride, 8: w, 9: cell_row_count,
                10: 8, 11: 2, 12: 128, 13: w * 2,
            })
        weight_step = 0 if mode == "shared" else 9
        weight_rows = 9 if mode == "shared" else h * w * 9
        input_rows = descriptor_rows + weight_rows
        return MemoryLayout(input_rows, cell_row_count * w * 64 * 2, {
            2: cell_row_start * stride,
            3: input_rows,
            4: descriptor_rows + cell_row_start * w * weight_step,
            5: stride, 6: stride - 4,
            7: w, 8: cell_row_count, 9: 2, 10: 8,
            11: w * 16 - 16, 12: w * 16 * 7, 13: weight_step,
        })

    @classmethod
    def tables(cls, *, shape, mode="shared", cell_row_start=0, cell_row_count=None):
        """XMEM tables as the Cache Unit sees them (see ``ipu_emu.table_trace``).

        ``S = 2*(W+2)`` rows per padded input row (two channel tiles per
        cell); the input table starts at the band's top halo row and holds
        ``n + 2`` padded rows for a band of ``n`` coarse rows.
        """
        h, w, _ = shape
        n = h - cell_row_start if cell_row_count is None else cell_row_count
        stride = (w + 2) * 2
        output_rows = n * w * 64 * 2
        if cls.separable:
            # Vertical pass: per phase a, each tile reads rows +0, +S, +2S, so
            # the next phase steps back from the last tile's +2S row to the
            # first tile. Kx is re-walked for every phase; scratch T is written
            # then read once per coarse row; the output's two channel tiles
            # interleave over 8 phases (13 rows back).
            return {
                "input": Table(cr=2, rows=(n + 2) * stride, jump_back=2 * stride + 2 * w - 1),
                "ky": Table(cr=4, rows=n),
                "kx": Table(cr=5, rows=w, repetitions=8 * n),
                "scratch": Table(cr=6, rows=8 * stride, repetitions=2 * n, jump_back=3),
                "output": Table(cr=3, rows=output_rows, jump_back=13),
            }
        # Direct: each (phase, tile) reads a 3x3 window of descriptor rows
        # (+0..+4, +S.., +2S..), stepping back 2S + 5 to the next one; its nine
        # weight rows are re-read for every (phase, tile). A cell's 8x8 output
        # block spans 8 phase rows, so the next cell steps back 7 of them.
        weight_rows = 9 if mode == "shared" else n * w * 9
        return {
            "input": Table(cr=2, rows=(n + 2) * stride, jump_back=2 * stride + 5),
            "weights": Table(cr=4, rows=weight_rows, jump_back=8),
            "output": Table(cr=3, rows=output_rows, jump_back=7 * 16 * w - 1 if w > 1 else 0),
        }
