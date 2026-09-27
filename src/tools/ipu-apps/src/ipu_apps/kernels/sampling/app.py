"""Dense descriptor interpolation in HWC order; no L2 normalization."""

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
