"""Convolution CR maps and row counts, matching the adjacent assembly headers."""
from ipu_apps.kernel_registry.memory import MemoryApp, MemoryLayout, ceildiv, positive_shape


class ConvApp(MemoryApp):
    kernel_size = 1
    single_channel = False

    @classmethod
    def memory_layout(cls, *, shape, out_channels, kernel_size, stride, padding, activation):
        c, h, w = positive_shape(shape, 3)
        positive_shape((out_channels,), 1)
        k = cls.kernel_size
        if kernel_size != k or stride != 1 or padding != k // 2:
            raise ValueError(f"requires kernel_size={k}, stride=1, padding={k // 2}")
        expected_activation = "none" if k == 1 else "relu"
        if activation != expected_activation:
            raise ValueError(f"requires activation={expected_activation}")
        if cls.single_channel and c != 1:
            raise ValueError("requires one input channel")
        tiles = ceildiv(w, 128 if k == 1 else 126)
        groups = ceildiv(c, 128 if k == 1 else 14)
        weights = (c + int(not cls.single_channel)) * (h + k - 1) * tiles
        bias = weights + out_channels * groups
        output = bias + out_channels
        if cls.single_channel:
            crs = {2: 0, 3: output, 4: weights, 5: bias, 6: tiles,
                   7: h, 8: out_channels, 9: 126, 10: 128, 11: tiles - 1}
        else:
            crs = {2: 0, 3: output, 4: weights, 5: bias, 6: h * tiles,
                   7: tiles, 8: h, 9: c, 10: out_channels, 11: groups, 12: 128}
            if k == 3:
                crs.update({13: 13, 14: 126})
        return MemoryLayout(output, out_channels * h * tiles, crs)
