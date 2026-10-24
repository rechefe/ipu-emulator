# Element formats replace wide-vector mode

_Status: accepted (2026-10-15). The format list (INT8, E1–E7, INT32) and the single per-run format are superseded by [ADR 0007](0007-operational-aaq-and-nx8.md); lane encoding still goes through one codec per format. Where ADR 0007 makes the two modes behave differently by architecture (the AaQ store's width and quantization, LOAD's metadata copy, MULT's QF expansion), those handlers branch on `fp32_mode`; no other handler does._

One element-format setting selects how lanes are encoded: INT8 and E1–E7 on hardware, FP32 and INT32 in the emulator only. Element width, row width and accumulator type derive from it. Every instruction is written once as decode → compute → encode through the format's codec, with no `if wide:` branches. "Wide mode" is simply the FP32 format, in which XMEM rows are four times wider while vectors keep 128 elements.

Kernels do not depend on the format: the same program runs in every format it declares, and the harness packs data per format. CR scalars are decoded in the active format. The earlier design kept a separate `dtype` plus three wide-mode flags and duplicated most handlers; that is what this replaces.
