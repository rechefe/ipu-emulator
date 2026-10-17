# Element formats replace wide-vector mode

_Status: accepted (2026-10-15). Implementation is planned in `planning/architecture-deepening/`._

One element-format setting selects how lanes are encoded: INT8 and E1–E7 on hardware, FP32 and INT32 in the emulator only. Element width, row width and accumulator type derive from it. Every instruction is written once as decode → compute → encode through the format's codec, with no `if wide:` branches. "Wide mode" is simply the FP32 format, in which XMEM rows are four times wider while vectors keep 128 elements.

Kernels do not depend on the format: the same program runs in every format it declares, and the harness packs data per format. CR scalars are decoded in the active format. The earlier design kept a separate `dtype` plus three wide-mode flags and duplicated most handlers; that is what this replaces.
