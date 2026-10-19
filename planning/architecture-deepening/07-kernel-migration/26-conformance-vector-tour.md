# Conformance: `vector_tour` onto `ipu_harness`

Part of [Epic: Kernel migration](00-epic.md) · ID E07-26 · Size S · Depends on: E07-01, E07-02, the runner's trace-context hook (E08-16's revisit, IMPLEMENTATION-DECISIONS "Inspector trace recorder") · Behaviour change: none

## Goal

Migrate `vector_tour`, the Cycle Inspector's curated conformance program (E08-16), to the E06
format. It is the 90th registered kernel and was added after this epic was planned. It is the
only kernel that already runs in four element formats (INT8, E3, E4, E5), so it is also the
first user of per-format goldens on an accumulator output.

## Background

Paths are under `src/tools/ipu-apps/src/ipu_apps/kernels/conformance/vector_tour/`.

- One program, `vector_tour.asm`, already upper case. It loads a mask, two R_CYCLIC slots and
  R0/R1, then runs MULT.VE, MULT.RC.VV/VS/VE and MULT.EE into ACC.ADD/SUB/MAX, masked
  multiplies, and a 4-row matrix-vector loop with AGG.SUM.FIRST. INT8 also stores
  ACTIVATE.QUANTIZE RELU through ST.AAQ. A run-time branch on CR12 (1 in INT8) skips that tail
  in the float formats.
- `cases.py`: cases `default` (INT8), `int8`, `e4`, `e5`, `e3`. Inputs come from
  `default_rng(16)` and are encoded with `ipu_isa.formats`. The NumPy reference is independent
  of the emulator: INT8 is compared exactly, the float formats with `check_close`.
- `app.py` writes CR2..CR12 and three R_MASK slots (raw bytes: all lanes, even lanes, low half).
- Its Inspector traces (`docs/inspector` CURATED `tour_int8`, `tour_e4`, `tour_e5`, `tour_e3`)
  are recorded through `//src/tools/ipu-apps:trace_case` until this issue.
- Read-only prep (lane D, at 56bba306) drafted its `kernel.toml`. Setup is register-identical
  in all four formats and the binary is unchanged.

## Implementation

1. `kernels/conformance/vector_tour/`: the program, a `kernel.toml` and a local
   `reference.py` (from `cases.py:reference`).
   - `two` is a float parameter: CR9 is the element scalar 2.0, encoded per format (E06-06).
   - The mask slots are a scratch region with `bytes` content (E06 gap G2).
   - Its first row stays where the old harness put it, with a `row` pin or `after` (gap G7).
   - The INT8-only tail: CR12 and the extra AAQ output rows. Either use an `int8_tail`
     parameter set per case, or a `when` condition on a bool parameter (gap G6). Record the
     choice in the PR.
2. Cases: `default`, `int8`, `e4`, `e5`, `e3`, each limited to its format. A `generator`
   replays `default_rng(16)`, so the inputs stay bit-identical.
3. `[golden]`: the local reference, `tolerance = { int32 = "EXACT", fp32 = "EXACT" }` for the
   accumulator output. Measure the float formats and record any tier other than `EXACT` with a
   `why`.
4. Keep literal register names (`CR2`) in the program rather than `{{ cr.a }}`. The program then
   stays plain assembly: the Inspector shows it as source, and its labels come from assembling
   the source alone. Move the header's CR map into `[abi]`.
5. Move the four `tour_*` Inspector traces onto the runner (`kernel_trace` over
   `ipu_harness.cli --trace` plus the runner's context hook).
6. Apply the per-family checklist, then delete the old folder.

## Tests

- [ ] `compare_setup` shows identical CRs, LRs and XMEM after setup in all four formats.
- [ ] `compare_legacy conformance` shows the binary equal to the E02 asm digest, every case's
  raw output equal, and cycles equal.
- [ ] The four `tour_*` traces (`trace.jsonl`) are byte-equal before and after. Their page JSON
  differs only in `source_path` and in the `source` header lines that moved to `[abi]`, and the
  PR names both.

## Acceptance Criteria

- [ ] `kernels/conformance/vector_tour/` holds the program, `kernel.toml` and `reference.py`;
  no `app.py`, `cases.py` or `test.py` remains.
- [ ] The Inspector's `tour_*` kernels are recorded through the runner.
- [ ] `bazel test //...` is green.
