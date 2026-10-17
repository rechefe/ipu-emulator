# A kernel is its assembly plus a declaration; the harness owns memory

_Status: accepted (2026-10-15). Implementation is planned in `planning/architecture-deepening/`._

A kernel is a `.asm` program (or a variant entry over a family template) plus a `kernel.toml` that declares its parameters, inputs, outputs, ABI, element formats and cases, and optionally a PyTorch `reference.py`. The harness places regions in XMEM, packs data per element format, programs CRs, assembles, runs and compares against the reference or user-supplied data. Kernels never build XMEM images themselves; the whole-image (`MemoryApp`) path is removed, and `pack.py` may only transform one declared tensor.

This replaces about 59 hand-written Python harnesses (roughly 15k lines, 80% of them duplicated across kernels) with declarations, so that writing a kernel is mostly writing assembly.
