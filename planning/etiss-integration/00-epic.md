# Epic: IPU as a native ETISS architecture, Python emulator as golden reference

## Goal

Re-implement the IPU emulator **over ETISS** (<https://github.com/tum-ei-eda/etiss>):
the IPU becomes an ETISS architecture plugin (`etiss::CPUArch`) whose
translate-to-C JIT executes assembler `--format bin` images directly. The
existing Python emulator (`ipu_emu`) stays as the golden reference; a
differential test suite keeps the two bit-exact.

Full design: [`docs/content/specs/etiss-integration.md`](../../docs/content/specs/etiss-integration.md).

## Why

- Speed: a JIT-compiled C datapath instead of a per-lane Python interpreter.
- Ecosystem: GDB remote debugging, instruction tracing, memory-mapped
  peripherals, SystemC-style system integration, and ETISS's own RISC-V core
  as the future IPU host (replacing the Unicorn plan).
- Same single source of truth: decode tables and register layout are
  **generated** from `instruction_spec.py` / `registers.py`.

## Scope

- ETISS architecture plugin `IPU` built from this repository (out-of-tree,
  `FIND_PACKAGE(ETISS)` like `ArchImpl/RV32IMACFD`).
- One 224-bit ETISS instruction per VLIW word (186 payload bits, 28-byte
  stride as produced by the assembler). Slot opcodes decoded in the callback.
- `struct IPU` and all bit offsets generated from the Python spec; semantics
  hand-written in C (`IPUFuncs.c`), a port of `ipu.py` + `ipu_math.py` +
  `activations.py`.
- A runner binary driven from Python (`run_test(..., backend="etiss")`), with
  files as the contract: ini config + IMEM/XMEM images in, XMEM image +
  register/stat JSON out.
- Bazel targets for ETISS (pinned commit, `rules_foreign_cc`), the plugin,
  the runner, the generator, and all tests; CI updated.

## Out of scope

- Wide-vector debug mode (stays Python-only).
- Cycle-accurate pipeline timing (one VLIW = one cycle, as today).
- In-process Python bindings (subprocess first; pybind11 is a later option).
- The RISC-V host virtual platform (issue 7, follow-up).

## Constraints

- **Never assign opcodes manually; never duplicate instruction metadata.**
  Everything the C++ side needs comes out of `gen_etiss.py`.
- Existing Python APIs (`run_until_complete`, `run_with_debug`, debug CLI,
  `IpuApp`) keep working unchanged; the ETISS path is additive.
- `bazel test //...` stays green; the build is the only entry point.

## Sub-issues

- [x] #1 — Spike: minimal `IPU` ETISS architecture executing a 224-bit NOP program
- [~] #2 — Build ETISS and the `IPU` plugin under Bazel; CI wiring (CMake + script done; Bazel fetch of ETISS open)
- [x] #3 — `gen_etiss.py`: generate struct, bit layout, and decode callback from the spec
- [x] #4 — Port instruction semantics and numerics to `IPUFuncs.c`
- [x] #5 — Python `etiss` backend for `run_test` / `IpuApp` and differential test suite
- [~] #6 — Debug (GDB), run stats, tracing, and user documentation (stats and docs done; GDB workflow open)
- [ ] #7 — Follow-up: RISC-V host + IPU in one ETISS virtual platform

Dependency order: **1 → 2 → 3 → 4 → 5 → 6**, then 7.

## Acceptance Criteria

- [ ] `bazel run //src/tools/ipu-etiss:ipu_etiss_run -- --imem prog.bin …`
      executes any program the Python emulator executes.
- [ ] For every program in `ipu-emu-py/test` and every kernel in `ipu-apps`,
      the ETISS backend produces an identical register file, XMEM image, and
      cycle count to the Python emulator (differential tests in CI).
- [ ] No opcode, bit offset, or register size is typed by hand in C/C++.
- [ ] Adding an instruction still follows the `CLAUDE.md` checklist, plus one
      C handler; a missing handler fails the build.
- [ ] The fully-connected kernel runs at least 10× faster on ETISS (TCC JIT)
      than on the Python emulator (measured, reported in docs).
- [ ] User docs describe how to run, debug (GDB), and extend the ETISS backend.
