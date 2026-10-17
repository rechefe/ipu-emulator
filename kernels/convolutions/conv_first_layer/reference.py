"""conv_first_layer's golden: ``filters`` holds each filter's bias, then its
27 taps (one region, as the program reads them), so split it and take the
family's shared conv2d."""

from ipu_kernels.reference.ops import conv2d


def reference(x, filters, **params):
    """``relu(conv2d(x, w, stride 2, padding 1) + b)``, w and b from ``filters``."""
    w = filters[:, 1:].reshape(filters.shape[0], x.shape[0], 3, 3)
    return conv2d(x, w, filters[:, 0], **params)
