# Epic: Emulated RISC-V host drives the IPU via a SystemRDL control register block

## Goal

Introduce an emulated **RISC-V host core** (via the [Unicorn](https://www.unicorn-engine.org/)
framework, embedded in the Python environment) that configures and sequences the
IPU emulator through a **memory-mapped control/configuration register block**.
The register block is authored once in **SystemRDL** and is the single source of
truth for the Python emulator model, the on-device **Rust** driver, and the
generated documentation.

This realizes, in the emulator, the host model already described in the
[Control Stage spec](../../docs/content/specs/stage-control.md) and the
[Cache Unit spec](../../docs/content/specs/cache-unit.md): an external RISC-V host
that configures the IPU, programs instruction memory, and starts/stops the core
over a host bus.

The full design is captured in
[`docs/content/specs/riscv-host-integration.md`](../../docs/content/specs/riscv-host-integration.md).

## Scope

The register block must expose **all current configuration** plus the execution
controls below.

Configuration (parity with today's Python harness):

- `dtype` (arithmetic data type)
- `dstructure` (`valid_elements` + `partition`, i.e. `CR15`)
- `CR2`–`CR14` application constants
- activation `elu_alpha`

Execution-flow control:

1. **Start** execution
2. **Halt** execution
3. **Reprogram** the IPU instruction memory
4. **Read and write** the program counter
5. **Reset** the IPU — resets everything **except** instruction memory

## Constraints

- **Single source of truth.** No hand-duplicated register metadata; everything
  generated from the SystemRDL source (consistent with `instruction_spec.py` /
  `registers.py`).
- **Rust, not C**, for the on-device driver/firmware.
- The existing direct-Python path (`run_until_complete`, debug CLI, app
  harnesses) must keep working — the host path is additive.
- Builds stay hermetic under Bazel.

## Sub-issues

- [ ] #1 — Define the IPU host-control register block in SystemRDL + codegen
- [ ] #2 — Emulator MMIO control-register model mapped onto `IpuState`
- [ ] #3 — Refactor the run loop into a host-drivable, steppable execution engine
- [ ] #4 — Integrate an emulated RISC-V core (Unicorn) with the IPU MMIO bridge
- [ ] #5 — Rust `no_std` firmware + generated driver for the IPU host
- [ ] #6 — End-to-end firmware-driven app run, parity tests, and docs

Dependency order: **1 → {2, 3} → 4 → 5 → 6**.

## Acceptance Criteria

- [ ] A SystemRDL register block is the single source for Python, Rust, and docs.
- [ ] A Rust `no_std` firmware image, running on an emulated RISC-V core, can
      configure the IPU, load a program into the mapped instruction memory, start
      it, poll
      for completion, read/write the PC, and reset it — all over MMIO.
- [ ] An end-to-end test runs an existing app (e.g. fully-connected) through the
      host path and asserts identical results to the direct-Python path.
- [ ] All new build dependencies (`rules_rust` + bare-metal RISC-V toolchain,
      `unicorn`, `peakrdl*`) are wired into Bazel, and `bazel test //...` passes.
- [ ] User-facing docs describe the host path and register map.
