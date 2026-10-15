# Epic E04: Emulator core — one cycle module, format codecs, faults and a stable reference-model API

ID E04 · Folder `04-emulator-core` · Runs after E03 (alongside E05) · Feeds E06, E07, E08 and the [steppable engine](../../riscv-host-integration/03-steppable-engine.md)

## Goal

Turn `ipu_emu` into a small set of deep modules with narrow interfaces:

- **One cycle module.** One VLIW word has exactly one implementation. Slot order and register timing come from the declared timing contract in `ipu_isa` (E03), not from call order in `ipu.py`. Prototyping mode is an adapter on that module, not a second cycle.
- **Handlers written once.** One module per slot. Each handler decodes lanes through the active element-format codec, computes, and encodes. No `if wide:` branches.
- **Faults and arithmetic events as specified.** Faults F-1 to F-9 have IDs, a halt record (code and PC) and an enable bit. Accumulator overflow and NaN give the IEEE result and set a sticky flag. They never halt.
- **A stable reference-model API** for a cocotb-style RTL testbench: `reset(format)`, `load_program`, registers by spec name, XMEM, `step()` returning a commit record, and `run`. The commit record and trace format is a versioned, language-neutral JSON schema in `ipu_isa`, and the conformance suite is stored as data.
- **Fast enough to keep the dev loop short.** Paged sparse XMEM, per-word execution plans cached by PC, vectorised lane codecs, all in pure Python and numpy.

`Ipu` shrinks to orchestration. `ipu_emu` depends only on `ipu_isa`.

## Background

Verified at HEAD 694a8a8 (`src/tools/ipu-emu-py/src/ipu_emu/`):

- `ipu.py` is 2351 lines. It holds every handler, operand resolution, MAC-identity probing, alias attribution and the cycle loop.
- The cycle is written three and a half times: `execute_vliw_cycle` (ipu.py:2257-2309), `execute_vliw_cycle_skip_break` (ipu.py:2311-2346), the prototyping `_cycle_plan`/`_execute_cycle_fast` (ipu.py:2139-2255), and the break-then-resume calling pattern in `emulator.py:63-169`. The slot order is hard-coded twice (ipu.py:2297-2304, 2336-2343) and once more as `_SLOT_ORDER` (ipu.py:2134).
- Register timing is decided per handler, by reading either `self._snapshot_buffer` or `self.state.regfile`. Several of these choices contradict the RTL docs (see plan 01, table "What the interface really is today").
- Wide-vector mode is 21 `_wide_vector_active()` checks and 18 `WideVectorArithmetic` references in `ipu.py` alone, plus three `IpuState` constructor flags (ipu_state.py:65-77).
- `XMem` allocates 512 MiB per `IpuState` (xmem.py:37). `clear()` and `__setitem__` report the whole buffer to the observer (xmem.py:111-126).
- `RegFile.raw()` (regfile.py:101-109) fires the write observer and drops constant-ones provenance on every call, including pure reads.
- Errors are a mix of `RuntimeError` (LR conflict, cycle limit), `ValueError` (XMEM bounds), `EmulatorError` (errors.py, a `RuntimeError` subclass), `IndexError` (undefined opcode) and `assert` (13 in regfile.py, 7 in ipu.py).
- `alias_observer.py` imports `_SLOT_PLANS` and reaches about seven `Ipu` privates (alias_observer.py:73, 137-145, 225-279).
- `execute.py:30` imports `ipu_as.compound_inst`, so `ipu_emu` depends on the assembler.

## Decisions applied

| Decision | Applied as |
|---|---|
| D3 | Refactor issues are output-identical and proven by E02 harnesses. Behaviour changes are separate issues, labelled. |
| D5, Q10-Q12, Q60 | E04-29 to E04-31: versioned commit-record/trace JSON schema in `ipu_isa`, thin Python API, conformance suite as data. ETISS migration stays possible because the contract is language-neutral (Q56). |
| D7, Q7 (plan 01 Q5a), Q72 | E04-08: prototyping is an adapter on the single cycle module, with identical results including partial state after a fault. The faithful path also gets per-word plans cached by PC, invalidated on IMEM write. |
| Q32 | E04-06, E04-12 consume the declared contract. The flips are separate behaviour issues: E04-25 (AGG `dest_slot`, ACC.RESHAPE `source`/`dest` post-LR-write, fix 3 `attn_v_*` kernels), E04-26 (RESHAPE reads this cycle's MULT_RES), E04-27 (MULT reads R_MASK at cycle start). |
| Q43, Q44, Q47, Q48 | E04-13 to E04-17: one element-format enum from `ipu_isa`, handlers decode → compute → encode through a per-format codec, no format branches. |
| Q49 | E04-28: CR scalars decoded in the active format, after a kernel audit, documented. |
| Q52, Q53, Q59 | E04-19 to E04-23: faults F-1 to F-9 with IDs and enable bits, all enabled at reset; arithmetic events as sticky flags; configuration errors at setup only; `SimulationLimit` dropped; F-8 "not modelled". |
| Q63 | Pure Python and numpy. No compiled extensions. |
| Q67 | E04-14: one module per slot; `Ipu` is orchestration only. |
| Q69 | E04-18: `ipu_emu` depends only on `ipu_isa`. Debuggers leave `ipu_emu` in E08. |
| Q71 | E04-02: tracked targets (`IpuState()` < 5 ms and < 5 MB, >= 50k cycles/s on the matmul benchmark, emulator unit tests < 30 s). Tracked, not gated. |
| L1-L6 (Q29) | L1 asserts → exceptions (E04-03), L2 paged 512 MiB sparse XMEM (E04-05), L3 `max_cycles` default 1M (E04-21), L4 stats reset per run (E04-21), L5 bit-identical vectorised FP8 proven by an exhaustive test (E04-13), L6 ranged observer events (E04-05). |
| Q25, Q35 | AaQ is out of scope. The AAQ handler is only moved and given codec I/O. Every AaQ item is recorded as an open question for E09. |

## Scope

- Test support (`conftest.py`, shared helpers), emulator benchmarks.
- Error hierarchy below `IpuError`; asserts replaced.
- `RegFile` read/write seam; paged sparse XMEM with ranged events and registered observers.
- Cycle module, execution plans cached by PC, issue step and fact records, operand resolver seam, cycle view, observers on fact records.
- Element-format codecs (numerics module) and handler modules per slot.
- Fault model, precise faults, cycle limit (F-9), fault enable bits, arithmetic sticky flags.
- Timing behaviour changes decided in Q32; CR scalar decode (Q49).
- Reference-model API, commit-record/trace schema, conformance suite.

## Out of scope

- The AaQ stage model, activation functions, LUT, block quantization, reciprocal/rsqrt(0), exp2 overflow, `ACTIVATE.QUANTIZE` saturation. Record each as an open question in E09 (Q25, Q35).
- Declaring the timing contract, the fault list, the element-format enum, the binary codec and `IpuError` in `ipu_isa`. E03 owns these; E04 consumes them.
- Debugger, TUI and CLI code. E08 owns `ipu_debug`; E04 gives it the public seams it needs.
- Kernel-side work beyond the 3 `attn_v_*` kernels touched by E04-25 (E07).
- The run-control engine itself (start/halt/reset/PC, MMIO). That is [`03-steppable-engine.md`](../../riscv-host-integration/03-steppable-engine.md).

## Relationship to the steppable engine

[`planning/riscv-host-integration/03-steppable-engine.md`](../../riscv-host-integration/03-steppable-engine.md) needs one primitive: "advance one cycle and report". **E04-07 (cycle module) is a prerequisite of that issue, and the engine sits on top of it.** The split is:

- **`Cycle` (E04-07)** owns what one word does: break evaluation, the snapshot, slot order, the LR conflict check, the idle-word path and observer hooks. It knows nothing about run control.
- **`Engine` (#3)** owns run control: `run(max_cycles)`, `halt()`, `reset()`, `get_pc`/`set_pc`, status flags and cycle counting. `Engine.step()` is `Cycle.step(honour_break=...)` plus bookkeeping.

If #3 landed first, `Engine.step()` would freeze today's `check_break` / `execute_vliw_cycle` / `execute_vliw_cycle_skip_break` calling convention into a public API. E04-21 puts cycle counting and stats reset into one small run-control helper so the engine can absorb it. E04-30's reference API delegates run control to the engine once #3 lands. `Cycle.reset()` clears the plan cache so #3's `reset(preserve_imem=True)` stays correct.

## Issue index

| ID | File | Title | Size | Depends on | Behaviour change? |
|---|---|---|---|---|---|
| E04-01 | [01-test-support.md](01-test-support.md) | Emulator test support: `conftest.py` and one helper module | S | — | none |
| E04-02 | [02-emulator-benchmarks.md](02-emulator-benchmarks.md) | Emulator benchmarks for the Q71 targets | S | E01 (benchmark tracking), E04-01 | none |
| E04-03 | [03-error-hierarchy-and-asserts.md](03-error-hierarchy-and-asserts.md) | Error hierarchy under `IpuError`; asserts become exceptions | S | E03 (`IpuError`) | none for programs |
| E04-04 | [04-regfile-read-write-seam.md](04-regfile-read-write-seam.md) | `RegFile` read view and write view | S | E04-01 | none |
| E04-05 | [05-paged-sparse-xmem.md](05-paged-sparse-xmem.md) | Paged sparse XMEM with ranged events and registered observers | M | E04-01 | none |
| E04-06 | [06-declared-slot-order.md](06-declared-slot-order.md) | Emulator consumes the declared slot execution order | S | E03 (timing contract), E02 (timing characterization) | none |
| E04-07 | [07-cycle-module.md](07-cycle-module.md) | Extract the single `Cycle` module | M | E04-06 | none |
| E04-08 | [08-execution-plans-cached-by-pc.md](08-execution-plans-cached-by-pc.md) | Per-word execution plans cached by PC; prototyping as an adapter | M | E04-07 | none |
| E04-09 | [09-issue-facts.md](09-issue-facts.md) | Issue step produces one `IssueFacts` record | M | E04-07 | none |
| E04-10 | [10-operand-resolver-seam.md](10-operand-resolver-seam.md) | One operand resolver interface, two adapters | S | E04-04, E04-08, E04-09 | none |
| E04-11 | [11-observers-on-fact-records.md](11-observers-on-fact-records.md) | Observers consume fact records, not `Ipu` privates | M | E04-05, E04-09 | none |
| E04-12 | [12-cycle-view.md](12-cycle-view.md) | Handlers read implicit registers through `CycleView` | M | E04-06, E04-10 | none |
| E04-13 | [13-element-codecs.md](13-element-codecs.md) | Element-format lane codecs with exhaustive bit-identity tests | M | E03 (element formats), E00 (FP8 encode fix) | none |
| E04-14 | [14-handler-modules-per-slot.md](14-handler-modules-per-slot.md) | One handler module per slot; `Ipu` becomes orchestration | M | E04-12 | none |
| E04-15 | [15-mult-on-codec.md](15-mult-on-codec.md) | MULT handlers written once against the codec | M | E04-13, E04-14 | none |
| E04-16 | [16-acc-load-store-on-codec.md](16-acc-load-store-on-codec.md) | ACC, AGG, load and store handlers written once against the codec | M | E04-13, E04-14 | none |
| E04-17 | [17-element-format-configuration.md](17-element-format-configuration.md) | Element format replaces `dtype` and the three wide flags | M | E04-15, E04-16, E03 | none |
| E04-18 | [18-depend-only-on-ipu-isa.md](18-depend-only-on-ipu-isa.md) | `ipu_emu` depends only on `ipu_isa` | S | E03 (codec), E08 (debug move) | none |
| E04-19 | [19-fault-model.md](19-fault-model.md) | Fault model: F-1 to F-9 with IDs and a halt record | M | E04-03, E04-07, E03 (fault list) | yes |
| E04-20 | [20-precise-faults.md](20-precise-faults.md) | Precise faults: detect before any slot writes | M | E04-12, E04-19 | yes |
| E04-21 | [21-cycle-limit-and-stats-reset.md](21-cycle-limit-and-stats-reset.md) | Cycle limit as F-9 (default 1M) and stats reset per run | S | E04-19 | yes |
| E04-22 | [22-fault-enable-bits.md](22-fault-enable-bits.md) | Per-fault enable bits and sticky fault flags | S | E04-19 | yes |
| E04-23 | [23-arithmetic-sticky-flags.md](23-arithmetic-sticky-flags.md) | Accumulator overflow and NaN: IEEE result plus sticky flag | S | E04-16, E04-19 | yes |
| E04-24 | [24-reads-keep-provenance.md](24-reads-keep-provenance.md) | Register reads keep constant-ones provenance | S | E04-04, E04-11 | yes (statistics only) |
| E04-25 | [25-lr-operands-post-lr-write.md](25-lr-operands-post-lr-write.md) | AGG `dest_slot` and RESHAPE `source`/`dest` read post-LR-write | S | E04-12, E03 | yes |
| E04-26 | [26-reshape-reads-this-cycle-mult-res.md](26-reshape-reads-this-cycle-mult-res.md) | ACC.RESHAPE reads this cycle's MULT_RES | S | E04-12, E03 | yes |
| E04-27 | [27-mult-reads-r-mask-at-cycle-start.md](27-mult-reads-r-mask-at-cycle-start.md) | MULT reads R_MASK at cycle start | S | E04-12, E03 | yes |
| E04-28 | [28-cr-scalars-in-active-format.md](28-cr-scalars-in-active-format.md) | CR scalars decoded in the active element format | S | E04-15, E04-17 | yes |
| E04-29 | [29-commit-record-schema.md](29-commit-record-schema.md) | Commit record and trace: versioned JSON schema in `ipu_isa` | M | E04-11, E04-19 | none (new output) |
| E04-30 | [30-reference-model-api.md](30-reference-model-api.md) | Stable reference-model API v1 | M | E04-17, E04-21, E04-29 | none (new API) |
| E04-31 | [31-conformance-suite.md](31-conformance-suite.md) | Conformance suite stored as data | M | E04-30 | none |

## Suggested order

```
E04-01 ──▶ E04-02 (measure before anything moves)
   │
   ├──▶ E04-04 ──────────────────────────────┐
   ├──▶ E04-05 ──────────────────┐            │
   │                             │            │
E04-03                           │            │
                                 │            │
E04-06 ──▶ E04-07 ──▶ E04-08 ──┐ │            │
              │                ├─┼──▶ E04-10 ◀┘
              └──▶ E04-09 ─────┘ │      │
                     └──▶ E04-11 ◀┘      │
                                         ▼
                                      E04-12 ──▶ E04-14 ──┬──▶ E04-15 ──┐
E04-13 ───────────────────────────────────────────────────┴──▶ E04-16 ──┴──▶ E04-17
E04-18 (when E03 codec + E08 move land)

                    ── refactors above are output-identical ──
                    ── behaviour changes below, one per PR ──

E04-19 ──▶ E04-20, E04-21, E04-22, E04-23
E04-24          E04-25   E04-26   E04-27   E04-28

E04-11 + E04-19 ──▶ E04-29 ──▶ E04-30 ──▶ E04-31
```

E04-07 is the earliest point at which the [steppable engine](../../riscv-host-integration/03-steppable-engine.md) can start. E04-13 can start as soon as E03 lands the element-format enum. The reference-model track (E04-29 to E04-31) can run in parallel with the timing flips (E04-25 to E04-27); each flip regenerates the affected conformance traces.

## Tracked targets (Q71)

| Target | Today | Owner issues |
|---|---|---|
| `IpuState()` construction | about 0.27 s and 550 MB (plan 02) | E04-05 |
| Matmul benchmark | to be measured in E04-02 | E04-08, E04-15, E04-16 |
| Emulator unit tests (`bazel test //src/tools/ipu-emu-py/...`) | about 57.6 s (plan 02) | E04-01, E04-05 |
