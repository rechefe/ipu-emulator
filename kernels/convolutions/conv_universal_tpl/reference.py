"""conv_universal_tpl's golden: ``filters`` holds each filter's 9 * in_channels
weights, then its bias (one region, as the program reads them), so split it and
take the family's shared conv2d. The bias is added only when the program reads
it: when seeded (has_bias or apply_relu), as the .asm and the pack decide."""

from ipu_kernels.reference.ops import conv2d


def reference(x, filters, **params):
    """``act(conv2d(x, w, stride 1, padding 1) + b)``, w and b from ``filters``."""
    w = filters[:, :-1].reshape(filters.shape[0], x.shape[0], 3, 3)
    seeded = params["has_bias"] or params["apply_relu"]
    return conv2d(x, w, filters[:, -1] if seeded else None, **params)
