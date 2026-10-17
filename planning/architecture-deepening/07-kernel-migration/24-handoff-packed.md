# E07-24 hand-off: the packed full-layer chains (lane B, 2026-10-17)

A working note for whoever ports `test_full_layer_l{4,5}_packed.py`. It
complements lane C's `24-handoff.md` (the seams, done; the unpacked chains).
Delete both in the commit that finishes E07-24.

## Where things stand

- **The kernels exist** (E07-23, lane B, `b3dd69e5`). They are registered
  and byte-identical to the old programs:

  | Old program | Entry |
  |---|---|
  | `asm_packed_layernorm_<shape>` | `packed_layernorm_<shape>` (`kernels/normalize`) |
  | `asm_packed_pack_<shape>` | `packed_pack_<shape>` (`kernels/reshape`) |
  | `asm_packed_unpack_<shape>` | `packed_unpack_<shape>` (`kernels/reshape`) |
  | `asm_packed_output_linear_generic`, `_silu` | `packed_output_linear{,_silu}_240` (`kernels/projections`, `k` a parameter) |
  | `asm_packed_output_linear_generic_p4`, `_silu_p4` | `packed_output_linear{,_silu}_192` (`kernels/projections`, `k` a parameter) |
  | `asm_packed_residual_add_<shape>` | `packed_residual_add_<shape>` (`kernels/elementwise`, `residual_add_rows` with `per_row` 2 / 8) |

  The layer shapes are L4 192x64 and L5 240x16.
- **The tests already run them, and pass by name.** Today the two tests load
  the registered programs (`fixture_kernel_support.kernel_program`) but still
  stage XMEM and set the 43 CRs by hand on one `IpuState`. They passed on
  b3dd69e5 with `--config=full`:
  - `test_full_l4_layer_packed_end_to_end` (137 s);
  - L5 `test_stage1_layernorm_qkv_unpack` (25 s);
  - L5 `test_stage2_attention` and `test_stage3_outproj_residual_ffn_residual` (79 s together).
  - Not run: `test_full_l5_layer_packed_end_to_end`. These are much faster
    than the unpacked chains' 80-90 minutes.
- **Lane C's 2bffcdcf predates E07-23.** It still has the 12
  `test/mobilevit/asm_packed_*.asm` and the 4 `test_packed_*.py`. Merge lane
  B (b3dd69e5 or later) into it first.
  - Expect conflicts in `DELETIONS.md`, `fixture_kernel_support.py`
    (lane C removed the direct-handoff helpers; lane B added
    `kernel_program` and moved `run_layernorm_capture_xmem` onto
    `fixture_normalize`) and `test_template_lint.py` / `test_format.py`
    (lane B dropped `test/mobilevit` from their roots).
  - Take lane C's deletions and lane B's E07-23 rows.

## Mapping onto a Chain (one `Chain(FP32)` per test)

Each `run_*` helper in the packed tests becomes one `chain.run` (or a loop
of them). Every packed tensor is declared as its rows, `[channels * tokens /
128, 128]`, pack `None`. So a packed output passes straight to the next
packed input as a `Resident`, with no repack and no relabel.

| Old helper | New step |
|---|---|
| `run_packed_layernorm(state, x_base, gamma, beta, ...)` | `chain.run("packed_layernorm_<shape>", {}, {"x": x, "gamma": gamma, "beta": beta})`. `gamma`/`beta` are `[channels]` (pack `channel_tile` tiles them). Masks, `-1/n` and `1/n` rows are scratch, placed by the step. |
| `run_packed_output_linear(..., k, n_out, weight_slices, output_base_row)` (one call per packed output row: L5 8 outputs, L4 2) | `whole = chain.reserve(n_out // per_call, pack=None)`, then for each group `g`: `chain.run(entry, {"k": k}, {"b": x_resident, "a": W[g*per:(g+1)*per]}, at={"y": whole.slice(g, 1)})`. Write the slices in row order (ST.AAQ spill). Hold `x_resident` across the loop (`chain.hold`), or a later group's scratch may land on it. |
| `run_unpack(state, packed_base, out_base)` over a third of the QKV output | `chain.run("packed_unpack_<shape>", {}, {"x": qkv.slice(third * rows, rows)})` |
| `run_pack` | `chain.run("packed_pack_<shape>", {}, {"x": unpacked})` |
| `run_packed_residual_add` (`a` read through CR0, so it is pinned at row 0) | Copy `a`'s rows to row 0 first: `chain.write(chain.read(a), at=0)`. The old tests did this with `state.xmem.write_address(0, ...)` (L5 `:574`, `:649`). Then `chain.run("packed_residual_add_<shape>", {}, {"a": row0, "b": b}, reset=("r_mask",))`. |
| attention (`qk_scores_*`, `softmax_rows_partial`, `attn_v_*`) on separate `IpuState`s with files | The migrated kernels on the same chain, as lane C's seams do: `test_seam_qkv_to_scores_km.py` (`repack`, the row-0 guard), `kernels/attention/test_seams.py`, `kernels/attention/test_softmax_feed.py`. Unpack's output is one channel a row (pack `None`). Check whether `qk_scores_*`'s `b`/`a` take that as a resident or need `chain.write(..., pack=...)`. |

## Traps found while promoting the kernels

- **R_MASK carries over.** Pack, unpack and the linears leave a one-hot R_MASK
  (the packed LayerNorm ends by reloading all ones). The residual add relies
  on all ones and loads no mask. The old tests call
  `set_r_mask(0xFF * 128)` before it (L5 `:386`). On a Chain, pass
  `reset=("r_mask",)` on every residual-add step.
- **Row 0.** `packed_residual_add`'s `a` is read through CR0. The residual
  skip (X, then resid1) must be copied to row 0 right before each add.
  Reserve a guard on row 0 earlier only if another step would otherwise
  first-fit there.
- **Hold the skips.** X (the layer input) is read again by residual 1, and
  resid1 by residual 2, 5-8 steps later. `chain.hold` them, or a step's
  scratch covers them and they go stale (the chain then refuses them,
  loudly).
- **Linear call counts.** L5: QKV 90 calls, OutProj 30, FFN1 120 (SiLU entry),
  FFN2 30 with `k = 960`. L4: QKV 288, OutProj 96, FFN1 384, FFN2 96 with
  `k = 768`. One `chain.run` each is fine; the program is cached per `k`.
- **Tolerances.** The linear entries are FP32_LONG with `atol = 5e-5`
  (`ffn2`'s long sums). On chained data the stages may exceed it. Hold
  intermediate steps with `check=False` and compare the chain's end, and
  each stage, against the float64 reference chain:
  - `ops.layernorm` (no eps; the packed tests' `numpy_layernorm` already
    has none);
  - `ops.linear`, `ops.silu`, `ops.softmax`, `ops.residual_add`.

  The old per-boundary asserts (LN < 1e-3, others < 1e-2, pack < 1e-5) are
  the budgets to keep.
- **`count_instructions`** (`fixture_packed_l{4,5}_measure.py`) observes an
  `IpuState`. A `Chain` runs a `ReferenceModel`, so count by stepping
  (`record.slots`) or drop the instruction totals. They are printed only,
  never asserted.
- **Weights.** `run_packed_output_linear` writes `W[g]` as weight-chunk rows
  itself. The entry's `a` input (`[n_out, k]`, pack `weight_chunks`) does
  that now, so pass the plain `[per_call, k]` slice.

## Then

- Move both tests to `kernels/mobilevit/` (manual/nightly like the others).
- Delete `fixture_packed_l{4,5}_measure.py` if nothing else uses them
  (`test_count_instructions_modes.py` uses `fixture_packed_l5_measure`), and
  `fixture_kernel_support.kernel_program`.
- Add DELETIONS.md rows and "Tests from" tables.
