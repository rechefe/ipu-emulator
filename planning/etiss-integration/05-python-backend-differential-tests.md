# Python `etiss` backend for `run_test` / `IpuApp` and differential test suite

Part of the [ETISS integration epic](00-epic.md). Depends on #2 and #4.

## Goal

Let every existing harness choose the backend with one argument, and make
"Python and ETISS agree" a permanent CI property.

## Implementation

- New package `ipu_emu/etiss/`:
  - `layout.py` — generated address/word constants (#3).
  - `runner.py` — `EtissRunner` wrapping the Bazel-built `ipu_etiss_run`
    binary (path via `runfiles`), building a temp working directory with
    `config.ini`, `imem.bin` (program image padded to 1024 words with the
    canonical NOP word), `xmem.bin` (serialised `state.xmem`), and reading
    back `xmem.bin`, `regs.json`, `stats.json`.
  - `backend.py` — `run_state_on_etiss(state, max_cycles, break_mode)`:
    serialises an `IpuState` prepared by `setup()` (CRs, dtype, `elu_alpha`,
    XMEM, program), runs, then writes results back into the **same**
    `IpuState` (regfile, XMEM, PC, `stats`) so `teardown()` and existing
    assertions work unchanged. Rejects `wide_vector_debug=True` with a clear
    error.
- `emulator.run_test(..., backend="python" | "etiss")` and
  `IpuApp.run(backend=...)`, default `"python"`; `IPU_EMU_BACKEND` env var as
  an override for CI matrices.
- `debug_callback` is only supported by the Python backend in v1 (ETISS
  interactive debugging goes through GDB, issue 6); passing both raises.
- Cycle-count contract: ETISS `total_cycles` = instructions executed =
  Python `cycles` (including run-off NOP cycles up to `INST_MEM_SIZE`).
- `max_cycles` → `etiss.max_instructions`-style limit in the runner; exceeding
  it maps to the same `RuntimeError` message as the Python loop.

## Tests

- [ ] `test_backend_parity.py` (in `ipu-emu-py/test`): a pytest fixture
      `backend` parametrised over both; the existing `_run()` helper gains a
      backend parameter so **every** test in `test_execute.py` runs twice.
      Compare `regfile.to_dict()`, XMEM (full 8 MB, hashed), PC, and
      `stats`.
- [ ] `ipu-apps` tests parametrised over backends (FC, identity, all softmax
      kernels) with golden outputs.
- [ ] Randomised differential test: assemble random-but-valid single-slot
      programs from `INSTRUCTION_SPEC` (operand ranges from the union layout),
      run both, compare — catches divergence for instruction combinations no
      kernel uses.
- [ ] Error parity: out-of-range XMEM row, bad cyclic index, `max_cycles`
      exceeded → same exception type and message prefix on both backends.

## Acceptance Criteria

- [ ] `run_test(backend="etiss")` runs every app kernel; outputs and cycle
      counts match Python exactly.
- [ ] Differential suite runs in CI on every push.
- [ ] Measured speed-up on `fully_connected` recorded in
      `docs/content/specs/etiss-integration.md`.
