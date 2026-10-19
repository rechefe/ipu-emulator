# A small stable reference-model API; everything else may change

_Status: accepted (2026-10-15). Implementation is planned in `planning/architecture-deepening/`._

The emulator exposes one versioned, non-breaking API for RTL verification (driven by a cocotb-style testbench): reset with an element format, load a program, read and write registers by their ISA names and XMEM, step one instruction word, run. `step()` returns a commit record. The commit record and trace format are a language-neutral JSON schema in `ipu_isa`, and the conformance suite (programs plus expected traces) is stored as data, so another implementation such as an ETISS-based simulator can replace the Python emulator later.

All other Python interfaces are internal and change without deprecation shims. Breaking the reference API needs a major version bump.
