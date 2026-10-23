"""pointwise_conv_tpl's golden: the family's shared conv2d. The bias is added
only when ``has_bias``: the seeded program without one (ReLU, no bias) reads
the zero rows the pack writes in place of ``b``."""

from ipu_kernels.reference.ops import conv2d


def reference(x, w, b=None, **params):
    """``act(conv2d(x, w) + b)``, ``w`` ``[out, in, 1, 1]``."""
    return conv2d(x, w, b if params["has_bias"] else None, **params)
