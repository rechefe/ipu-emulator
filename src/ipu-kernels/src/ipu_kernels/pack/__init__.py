"""Named pack transforms shared across kernel families.

Each transform reshapes one declared tensor into the layout a kernel reads
(ADR 0006); building whole XMEM images is the harness's job, not a pack's.
``pack = "<name>"`` in ``kernel.toml`` finds these by attribute name, after the
kernel's own ``pack.py``, its family folder's ``kernels/<family>/pack.py`` and
its family's ``ipu_kernels.<family>.pack``.
"""

from ipu_kernels.pack.layouts import channel_major, chunk_interleaved, output_major_weights

__all__ = ["channel_major", "chunk_interleaved", "output_major_weights"]
