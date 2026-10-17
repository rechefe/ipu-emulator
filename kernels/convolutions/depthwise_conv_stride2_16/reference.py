"""The stride-2 depthwise golden: depthwise_conv_tpl's split of ``filters``, the
shared conv2d at the case's stride (2) and groups; the output is ``out`` (``y``
is stage 1's full-resolution result, a scratch region)."""

from ipu_kernels.reference.ops import conv2d


def reference(x, filters, **params):
    """``depthwise conv2d(x, w, stride 2, padding 1)`` (no bias, no ReLU)."""
    w = filters[:, 1:].reshape(filters.shape[0], 1, 3, 3)
    return {"out": conv2d(x, w, None, **params)["y"]}
