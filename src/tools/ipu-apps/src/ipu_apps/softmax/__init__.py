"""Softmax applications (row / column variants).

Five apps split across two axes -- which direction the reduction runs, and how
the reduced length relates to the 128-element datapath. Use :func:`lookup` to pick
one for a given shape (or be told no app covers it), and :func:`catalog` for the
whole coverage table::

    from ipu_apps.softmax import lookup
    v = lookup(axis="rows", n=300, rows=8)
    if v:
        print(v.app_class, v.kwargs)

If you already have the query in PyTorch terms, :func:`lookup_torch` takes the
tensor shape and ``dim`` exactly as you'd write the ``torch.softmax`` call::

    from ipu_apps.softmax import lookup_torch
    lookup_torch((32, 300), dim=1)      # softmax over each row of a 32x300

The same answers are available from the command line::

    python -m ipu_apps.softmax --axis rows --n 300 --rows 8
    python -m ipu_apps.softmax --shape 32,300 --dim 1
    python -m ipu_apps.softmax --catalog
"""

from ipu_apps.softmax.registry import (
    GAPS,
    Verdict,
    catalog,
    lookup,
    lookup_torch,
)

__all__ = ["GAPS", "Verdict", "catalog", "lookup", "lookup_torch"]
