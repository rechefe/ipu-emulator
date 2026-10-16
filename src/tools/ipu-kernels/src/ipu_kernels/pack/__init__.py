"""Named pack transforms shared across kernel families.

Each transform reshapes one declared tensor into the layout a kernel reads
(ADR 0006); building whole XMEM images is the harness's job, not a pack's.
"""
