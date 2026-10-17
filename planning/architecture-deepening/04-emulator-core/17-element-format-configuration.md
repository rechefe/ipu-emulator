# Element format replaces `dtype` and the three wide flags

Part of [Epic E04: Emulator core](00-epic.md) · ID E04-17 · Size M · Depends on: E04-15, E04-16, E03 (element-format enum) · Behaviour change: none

## Goal

`IpuState` takes one element format (Q47) instead of `dtype`, `wide_vector_debug`, `wide_vector_arithmetic` and `wide_vector_quantize_output`. The format is chosen at construction or reset and is a configuration value: an invalid format raises `ConfigurationError` at setup time, never mid-run (Q59). Any kernel can be simulated in any format it supports (Q44, Q50).

## Background

- `IpuState.__init__` has nine keywords (ipu_state.py:65-77), four of them about formats.
- Tests and apps set `state.dtype = DType.INT8` after construction (for example test_execute.py:3129), and apps pass `wide_vector_debug=True` (fully_connected, softmax, reshape and others).
- After E04-15 and E04-16, `active_format(state)` is the only reader of these flags in the datapath. Readers remain in `alias_observer.py` and the debuggers (`debug_tui.py`, 11 sites per plan 02).
- `ACTIVATE.QUANTIZE` behaviour in wide mode depends on `wide_vector_quantize_output`. AaQ is out of scope (Q35); this flag needs a home that does not reopen AaQ.

## Implementation

1. `IpuState(*, fmt: ElementFormat = ElementFormat.INT8, activation: ActivationParams | None = None, prototyping: bool | None = None, alias_profile=None)`. `fmt` is read-only after construction; changing it means building a new state or calling `reset(fmt)` (added by E04-30's API layer, implemented here as `IpuState.reset(fmt)`).
2. Delete `dtype`, `wide_vector_debug`, `wide_vector_arithmetic` and `active_format()`. `Ipu` reads `state.fmt`.
3. `wide_vector_quantize_output` moves into an `AaqOptions` value held by `IpuState` and read only by the AAQ handler. No behaviour change. Record in E09's open questions that the AaQ placeholder still carries this option.
4. Migrate every call site in `src/tools/ipu-apps` and the tests: `wide_vector_debug=True, wide_vector_arithmetic=FP32` → `fmt=ElementFormat.FP32`; INT32 likewise; `dtype=DType.E4` → `fmt=ElementFormat.E4`. Use a scripted rewrite; list the files in the PR.
5. `alias_observer` and the debuggers read `state.fmt` and the codec's `element_bytes` (debugger changes coordinated with E08).
6. Update `docs/content/wide-vector-debug-mode.md` and `ipu-configuration.md` to describe formats (E09 owns the rewrite; this PR fixes facts only).

## Tests

- [ ] New: `IpuState(fmt=...)` for every format sets element bytes, row bytes and accumulator type as declared in `ipu_isa`.
- [ ] New: assigning `state.fmt` raises; `IpuState(fmt="bogus")` raises `ConfigurationError`.
- [ ] Output-identical proof: E02 kernel output digests (every kernel in its current format), `test_wide_vector_debug.py` rewritten to `fmt=FP32`/`INT32` with identical expectations, `test_prototyping_equivalence.py` under `--define=ipu_proto=1`.

## Acceptance Criteria

- [ ] `grep -rn "wide_vector_debug\|wide_vector_arithmetic\|DType" src/` finds nothing outside the AaQ option and E09-tracked docs.
- [ ] One format value configures everything format-dependent.
