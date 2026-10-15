# Bazel targets that record Cycle Inspector traces

Part of [Epic: Debugger, generators and editor](00-epic.md) · ID E08-16 · Size M · Depends on: E04-29 (commit record and trace JSON schema), E04-30 (reference-model API), E08-15 (`ipu-gen vscode-data` for `isa.json`) · Behaviour change: none (new targets)

## Goal

The traces the Cycle Inspector replays are build outputs, not hand-recorded files (Q55, phase 1). A Bazel rule runs a curated kernel case through the E04 reference model with its trace writer enabled. An `ipu_gen` adapter then converts the language-neutral trace into the viewer's compact per-kernel JSON. The docs build regenerates them every time, so the page can never show a stale ISA or a stale kernel.

## Background

- The Cycle Inspector artifact (https://claude.ai/artifact/4sTeRLzw27gc3z8JmhhMVB) fetches `data/isa.json` and `data/<id>.json`. Each trace has:
  - `meta` (`dtype`, `wide`, optional `lut`);
  - `program`, per word: `label`, hex `word`, and `slots[{slot, op, args}]`;
  - `layout`: `[field name, width]` MSB → LSB;
  - `init`: hex image per register;
  - `cycles`, per cycle: `pc`, register deltas `d`, lanes used `L`;
  - `total_cycles`, `truncated`, `source`.

  It covers 14 cases, including fully_connected INT8/E4/E5, identity, softmax_rows and conv_universal, plus four hand-written "vector tour" demos that are not in the repo. Nothing in the repo produces these files today.
- Q60: the commit record and trace are a versioned JSON schema in `ipu_isa`, written by the E04 reference model. The viewer format is a presentation of that trace, not a second contract.
- Q55: curated kernels, regenerated on each docs build.

## Implementation

1. `ipu_gen/inspector_trace.py` is an adapter from the E04 trace (schema version pinned) to the viewer JSON above:
   - decoded slots come from the `ipu_isa` codec (`DecodedWord.slots()`);
   - `layout` comes from `WORD_LAYOUT` (E05-03);
   - register deltas come from the commit records' register writes;
   - `L` (lanes used) comes from the MULT slot's mask, or from a field the E04 trace carries. Open question: does the E04 trace record lanes used, or does the adapter derive it from `R_MASK`?
   - a `max_cycles` cap sets `truncated`.

   Add a `schemaVersion` integer to the viewer JSON (L40 style), so the page can reject a trace it does not understand.
2. A `kernel_trace` macro in `docs/inspector.bzl`:

   ```python
   kernel_trace(name, kernel, case = "default", format = None, max_cycles = 5000)
   # genrule: run the harness case with the trace writer on (E06 runner flag, e.g. `--trace out.json`),
   #          then `ipu-gen inspector-trace in.json out/<name>.json`
   ```

   Running the case uses the E06-11 runner's trace option. Until E06-11 lands, call the E04 reference API directly from a small `py_binary`.
3. `docs/inspector/BUILD.bazel` lists the curated set as data: one `kernel_trace` per case, with title and description in a `curated.toml` read by the page build. Start with the in-repo cases from the artifact: fully_connected (INT8, E4, E5), identity, channel_peak, softmax_rows, depth_to_space, conv_universal and residual_add. Use the E07 format names (`fp8_e4`, `e4`) once that rename lands. The vector-tour demos join only if they are added to the repo as test programs. Open question: keep them?
4. `filegroup(name = "inspector_data")` contains every trace plus `isa.json` from `//vscode-ipu-asm:generate_grammar` (or `ipu-gen vscode-data`).
5. Tag the targets `docs` so they build with the docs and not in the default CI tier, which has a 5-10 minute budget (E01).

## Tests

- [ ] An adapter unit test on a tiny program (3 words, one loop) checks `program`, `layout`, `cycles[*].pc` and register deltas against hand-written expectations.
- [ ] Replaying `init` + all deltas reproduces the final register state from the reference model (sampled registers).
- [ ] `bazel build //docs/inspector:inspector_data` succeeds. Each output validates against `ipu_gen/schemas/inspector-trace.schema.json`.
- [ ] `bazel test //...` green; the new tests are not tagged `docs`.

## Acceptance Criteria

- [ ] Every trace the page loads is a Bazel output. No trace JSON is checked in.
- [ ] Adding a curated case is one entry in `docs/inspector/BUILD.bazel` plus one entry in `curated.toml`.
