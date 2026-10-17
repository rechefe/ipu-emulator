"""depthwise_conv_tpl's golden: ``filters`` holds each channel's bias, then its
9 taps (one region, as the program reads them), so split it and take the
family's shared conv2d with groups = channels. The bias is added only when
``has_bias``: a seeded program without one reads the zero the pack writes in
place of the bias column."""

from ipu_kernels.reference.ops import conv2d


def reference(x, filters, **params):
    """``act(depthwise conv2d(x, w, stride 1, padding 1) + b)``, w and b from ``filters``."""
    w = filters[:, 1:].reshape(filters.shape[0], 1, 3, 3)
    return conv2d(x, w, filters[:, 0] if params["has_bias"] else None, **params)
