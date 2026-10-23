# Stable reference-model API v1

Part of [Epic E04: Emulator core](00-epic.md) · ID E04-30 · Size M · Depends on: E04-17, E04-21, E04-22, E04-29 · Behaviour change: none (new API; legacy `execute.py` wrappers removed)

## Goal

Give RTL verification a small, fixed, non-breaking Python interface to the emulator (D5, Q10, Q11): a cocotb-style testbench resets the model, loads a program, sets registers and XMEM, steps cycle by cycle and compares each commit record with the RTL. Everything else in `ipu_emu` stays free to change.

## Follow-up

- 1.1.0 (additive) added `ReferenceModel.stats() -> Stats`, the run statistics lane B's E06-11 runner needs (effective MAC utilization); see the decisions row.

## Background

- Today callers combine `IpuState`, `load_program`, `run_until_complete`, `Ipu(state).execute_vliw_cycle()` and register accessors with names like `get_lr`, `raw("r_acc")` and `get_r_wide_debug`. Nothing is marked stable.
- `execute.py:97-121` has legacy wrappers (`execute_next_instruction`, `execute_instruction_skip_break`) that build a new `Ipu` each cycle.
- The [steppable engine](../../riscv-host-integration/03-steppable-engine.md) will own run control (start, halt, reset, PC). This API must not duplicate it: until the engine lands, the API uses `RunControl` (E04-21); afterwards it delegates to `Engine`.

## Implementation

New module `ipu_emu/reference.py`, re-exported as `ipu_emu.reference`:

```python
API_VERSION = "1.0.0"

class ReferenceModel:
    def reset(self, fmt: ElementFormat) -> None                 # clears registers, XMEM, faults, flags, stats
    def load_program(self, words: bytes | Sequence[int]) -> None  # binary words, decoded by the ipu_isa codec
    def read_register(self, name: str, index: int = 0) -> int | bytes   # names as in ipu_isa
    def write_register(self, name: str, value: int | bytes, index: int = 0) -> None  # only while halted
    def read_xmem(self, address: int, size: int) -> bytes
    def write_xmem(self, address: int, data: bytes) -> None     # only while halted
    def set_fault_enable(self, fault: FaultId, enabled: bool) -> None
    def step(self) -> CommitRecord                               # one cycle
    def run(self, max_cycles: int | None = None, *, trace: TraceWriter | None = None) -> RunResult
    @property
    def status(self) -> Status                                   # RUNNING, HALTED, FAULTED (with fault ID, PC)
```

Rules:

1. Names come from `ipu_isa`, never from emulator internals. Register writes that hardware forbids (CR0, CR1; faults.md "Illegal host accesses") raise `ConfigurationError`.
2. After a fault halt, `step()`, `run()` and register/PC writes raise `ConfigurationError` until `reset()` (IPU-FLT-7).
3. `step()` never stops at the EMULATOR-only BREAK slot (Q39); it retires the word. HALT ends the run with `Status.HALTED`.
4. Delete `execute_next_instruction` and `execute_instruction_skip_break` and migrate their callers (test_execute.py:1913-1952) to the API or `Cycle`.
5. Contract test `test_reference_api_contract.py`: pins the public names and signatures (via `inspect.signature`) against a committed snapshot, and the behaviour of each method. Changing the snapshot requires a version bump; a breaking change requires a major bump (Q11).
6. Strict typing: the module passes the type checker configured in E01 (Q65 puts the reference API first): a `mypy_test(name = "mypy_reference", lib = <the API library>, strict = True)` from `//tools:typecheck.bzl` (E01-11), with the API module in a py_library of its own if the package's other modules are not yet strict-clean.
7. Short usage page for cocotb users (E09 owns the docs section; this PR adds the page content).

## Tests

- [ ] Contract snapshot test.
- [ ] `step()` sequence over a kernel equals the records collected during `run()`.
- [ ] `reset(fmt)` for every format gives the declared register widths.
- [ ] Writes while running, writes to CR0/CR1, and use after a fault halt raise `ConfigurationError`.
- [ ] A kernel run through the API produces the same outputs as `run_until_complete` (E02 kernel digests, three kernels).

## Acceptance Criteria

- [ ] `ipu_emu.reference` exposes only the names above, with `API_VERSION`.
- [ ] No API method returns an internal object (`IpuState`, `RegFile`, `Ipu`).
- [ ] The steppable-engine issue references this API as a consumer of `Engine`.
- [ ] A strict `mypy_test` (E01-11) over the API module passes in the default tier.
