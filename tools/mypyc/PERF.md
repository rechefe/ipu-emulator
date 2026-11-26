# Emulator speed: the mypyc build and `mypyc-build-2`

What each change on this branch does for the emulator's speed, which
behaviour it changes, and which tests changed with it. Every change keeps
results bit-identical: `tools/mypyc/build.sh` checks that matmul_384x192_x128,
fully_connected e4 and fully_connected dynamic end in the same state compiled as
pure, with the same fingerprints as before the branch (`a75173d8`, `0973b50f`,
`f5f748c7`).

## How it was measured

- **Workload:** 100K emulated cycles (one `matmul_384x192_x128.default` plus nine
  `fully_connected.default` e4), in kernelgen's live fast-path mode: prototyping
  on, no commit collector, `np.errstate` a no-op.
- **Method:** ABBA on one pinned CPU of the L40, configs run back to back.
- **Noise:** the machine is shared, and a CPU's speed moves with the load on its
  sibling thread. So only ratios between neighbouring runs are quoted.
  Absolute rates are low because those cores are contended.
- **Pure:** the original pure-Python fast path (`mypyc-build`'s sources,
  uncompiled), 7.4k-8.2k cycles/s on these CPUs.
- **Live build:** the compiled `mypyc-build` (`b350178f`), 1.19x pure.

## Increments

| Commit | Change | Gain over the previous | vs pure |
|---|---|---|---|
| `2fadfd88` `256ff3ea` | per-op read steps in the resolver; float32 ACC/MULT while every lane is finite | | 1.59x |
| `154800a6` `44ee7a2f` | one Ra element for an LR scalar; `Final` constants, fewer generic lookups | | |
| `4f520c6a` | resolver views, alias fast path, buffer pairs | | 1.75x |
| `60064fad` `359cd69b` | cheaper float32 checks; element width read once per state | +8% | 1.89x |
| `b4eb0dae` | `RegFile.mutable` without a method call | +2% | |
| `01b73782` | single-buffer `RegFile` (snapshot copy is one memcpy) | +8.7% | 2.13x |
| `eec14098` | fused MULT.RC.VE -> ACC hand-over of the float32 product | +2.6% | 2.12x* |
| `18948e97` | native `Ipu` and `Cycle`, explicit close | +9.1% | |
| `2dec7275` | native `IpuState` | +6.0% | |
| `3d389361` | NX8 MULT window decodes only the ring rows it covers | +5.3% (NX8 alone +33%) | 2.58x |
| `0c116f56` | resolver: native step objects, LR/CR views as a tuple | +5.2% | 2.75x |
| `90344e69` | native `RegFile` | +13.7% | **3.0x** |

\* The ratios in the "vs pure" column come from different measurement sets, so
neighbouring rows are not strictly comparable. The final check measured
`90344e69` against pure back to back, 3 rounds on each of two CPUs: 2.92x to
3.13x per round, 3.02x on average.

## Behaviour changes

- **Explicit close** (`18948e97`, `2dec7275`). mypyc cannot weakly reference a
  native class, so the weak references between the engine objects became strong
  ones that `close()` breaks:
  - `Ipu.close()` and `Cycle.close()` remove the IMEM write observer and release
    the engine.
  - `run_until_complete` and `run_with_debug` close the engine they build.
  - `ReferenceModel.close()` (API 2.1.0) closes the engine and the commit
    recording, and detaches the alias profile, which keeps its counts. A closed
    model can still read and write registers and memory; `step`, `run` and
    `debug_run` raise `ConfigurationError("the model is closed")`.
  - `ipu_harness.run_kernel` closes a model it built itself.
  - An alias profile holds its state until it is detached.
  - Without close, the engine and its state wait for the cyclic garbage
    collector.
- **`RegFile.raw()` returns a view** (`01b73782`). The deprecated `raw()` now
  returns the register's writable `memoryview` instead of a `bytearray`: the
  same bytes, and writes through it still reach the register.
- **`RegisterSizeError`** (`01b73782`). Setter data whose size would have resized
  a register's storage now raises `RegisterSizeError`; before, it silently
  changed the register layout. Writes otherwise convert or refuse data exactly
  as a `bytearray` slice assignment did.
- **Native `RegFile` with a custom-schema warning** (`90344e69`). A compiled
  `RegFile` has no per-instance attributes, so a custom schema's named accessors
  are not attached. Building one warns (`UserWarning`) and names the generic
  methods: `get_scalar`, `set_scalar`, `get_register_bytes`,
  `set_register_bytes`, `view`, `mutable`. The default schema keeps every named
  accessor.
- **`RegFile.view()` is cached** (lever 3, approved). `view(name)` returns the
  same read-only `memoryview` object on every call, built once per register
  file (and once per snapshot) from its register's slice of the single buffer.
  It still sees every later write. **Do not `release()` it, and do not use it
  as a context manager** (`with rf.view(...)` releases it on exit): either
  breaks every later reader of that register. Copy (`bytes(...)`) whatever has
  to outlive a write. The cache never goes stale: a register's slice never
  changes identity, because the buffer is only allocated in `__init__` and
  `snapshot()` (each builds its own views) and is otherwise written in place
  (`copy_into` included).

## Tests changed, and why

- `test_cycle.test_imem_does_not_keep_a_cycle_alive` took a weakref to a
  `Cycle`, which a native class cannot have. It is replaced by two tests:
  - `test_closing_a_cycle_removes_its_imem_observer`
  - `test_a_closed_engine_is_freed_without_the_cycle_collector`
- `test_alias_profile.test_profiled_state_is_freed_without_the_cycle_collector`
  took a weakref to an `IpuState`, and the profile now holds its state. It
  becomes `test_a_detached_profile_frees_its_state_without_the_cycle_collector`:
  detach, then check that the state's XMEM is freed with the collector off.
- `test_alias_profile.test_profiled_xmem_is_freed_without_the_cycle_collector`
  becomes `..._once_detached`: it detaches the profile first.
- `test_reference_api_contract` gets the 2.1.0 snapshot (2.0.0 plus `close`)
  and `test_a_closed_model_is_freed_on_release_and_runs_no_more`.
- New tests:
  - `test_fused_mult_acc`: the fused and the general path give the same state,
    in FP32 and NX8, with NaN, inf, overflow and subnormal lanes; and the
    hand-over reaches only the right ACC.
  - `test_mult_window`: the float32 window equals the float64 one at every
    start element.
  - `test_regfile.test_a_custom_schema_warns_and_works_through_the_generic_methods`
  - `test_regfile.test_view_is_one_cached_object_that_sees_later_writes`

Memory, 200 back-to-back `run_kernel` calls in one process with the cyclic
collector off:

| Build | Growth |
|---|---|
| before `18948e97` | +57 MB (each run's model waited in reference cycles) |
| after, and on `90344e69` | +6.4 MB |
