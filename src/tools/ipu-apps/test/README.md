# Packed activation layout — kernel index

This directory holds a standalone, **untracked, `BUILD.bazel`-unregistered**
experiment: an alternative "packed" XMEM layout for MobileViT L4/L5
activations, where multiple channels are packed side-by-side into one
128-lane row instead of one channel per row. Packing is free for elementwise
kernels (residual add: 8x fewer cycles/instructions/XMEM bytes for L5) but a
packed **linear** layer needs a cross-partition combine step the ISA has no
direct instruction for — the files here are the kernels, seams, and
primitives built to work around that, plus the baselines used to validate
them.

For the full design writeup — `rc_idx` scatter/gather formulas,
packing-factor derivations, replication-slot counts, the no-segmented-reduce
workaround — see [`kernel_docs/kernel_layer_map.md`](../../../../kernel_docs/kernel_layer_map.md#packed-activation-layout),
"Packed activation layout" section. For the engineering history behind the
corrections and bugs referenced below (broadcast formula, `ACC.ADD.FIRST`
scheduling, the `R_MASK` hazard, etc.), see
[`docs/isa_friction_log.md`](../../../../docs/isa_friction_log.md).

These kernels are exercised only via direct pytest (`test_*.py` files in
this directory, or the `test_full_layer_l{4,5}_packed.py` /
`test_l5_real_size_packed_b.py` harnesses for the three files with no
same-stem test) — not `bazel test`. See each file's `Tests:` header field
for the exact command.

For the branch's main, Bazel-registered production kernel library (the
non-experimental L3/L4/L5 kernels these packed variants are ported from),
see [`src/tools/ipu-apps/src/ipu_apps/README.md`](../src/ipu_apps/README.md).

## Index

| File | Layer | Scope | Layout | Status | Role |
|---|---|---|---|---|---|
| `asm_packed_layernorm_192x64.asm` | L4 | single-stream | packed | validated | LayerNorm |
| `asm_packed_layernorm_240x16.asm` | L5 | single-stream | packed | validated | LayerNorm |
| `asm_packed_linear_240to8_masked.asm` | L5 | single-stream | packed→unpacked | validated | Masked-pass linear (adopted) |
| `asm_packed_linear_240to8_replicated.asm` | L5 | single-stream | packed→unpacked | **superseded** | Pre-replicated-weight linear (memory blowup) |
| `asm_packed_linear_masked_generic.asm` | L5 | single-stream | packed→unpacked | template | Generic K/N_OUT masked-linear template |
| `asm_packed_output_linear_generic.asm` | L5 | single-stream | packed | validated | Packed-output linear, 4-slot baseline |
| `asm_packed_output_linear_1slot.asm` | L5 | single-stream | packed | validated | Packed-output linear, 1-slot optimization |
| `asm_packed_output_linear_silu.asm` | L5 | single-stream | packed | validated | Packed-output linear, SiLU (FFN1-style) |
| `asm_packed_output_linear_generic_p4.asm` | L4 | all-stream/P4 | packed | validated | Packed-output linear, L4 port |
| `asm_packed_output_linear_silu_p4.asm` | L4 | all-stream/P4 | packed | validated | Packed-output linear, SiLU, L4 port |
| `asm_packed_output_linear_tiny.asm` | L5 | single-stream | packed | validated | Packed-output linear, K=8 original construction |
| `asm_packed_pack_192x64.asm` | L4 | single-stream | seam (→packed) | validated | Unpacked→packed seam |
| `asm_packed_pack_240x16.asm` | L5 | single-stream | seam (→packed) | validated | Unpacked→packed seam |
| `asm_packed_unpack_192x64.asm` | L4 | single-stream | seam (→unpacked) | validated | Packed→unpacked seam |
| `asm_packed_unpack_240x16.asm` | L5 | single-stream | seam (→unpacked) | validated | Packed→unpacked seam |
| `asm_packed_residual_add_192x64.asm` | L4 | single-stream | packed | validated | Residual add |
| `asm_packed_residual_add_240x16.asm` | L5 | single-stream | packed | validated | Residual add |
| `asm_primitive_a_combine2x64.asm` | L4 | single-stream | n/a (microbenchmark) | validated | Cross-partition combine primitive |
| `asm_primitive_a_combine8x16.asm` | L5 | single-stream | n/a (microbenchmark) | validated | Cross-partition combine primitive |
| `asm_unpacked_linear_240to8.asm` | L5 | single-stream | unpacked | validated | Baseline comparison for the linear family |

## By family

### LayerNorm

`asm_packed_layernorm_192x64.asm` (L4, packing factor 2) and
`asm_packed_layernorm_240x16.asm` (L5, packing factor 8) — six-step
algorithm (sum → broadcast → center → sum-of-squares → broadcast → affine)
mirroring the unpacked `layernorm_16x240.asm` shape, but with every
cross-partition step re-expressed via masked `MULT.RC.VE` gather/scatter.
The L5 file is the original; L4 is a direct port. Both broadcast steps use
`rc_idx=(-16*p) mod 512` — **not** `mod 128** as an earlier task brief
stated; see the friction log for the derivation and the four kernel-authoring
bugs found and fixed against the corrected formula.

### Linear / output-linear

Three separate constructions, not stages of one pipeline:

- **Masked-pass** (`asm_packed_linear_240to8_masked.asm`,
  `asm_packed_linear_masked_generic.asm`): packed input, unpacked output,
  weights read with a masked pass rather than replicated. This is the
  adopted, memory-optimal approach.
- **Pre-replicated** (`asm_packed_linear_240to8_replicated.asm`):
  compute-optimal but **superseded** — a full per-token weight replica costs
  too much XMEM. Kept only as the compute-optimal comparison point.
- **Packed-output** (`asm_packed_output_linear_generic.asm` and its
  `_1slot`/`_silu`/`_tiny`/`_p4` variants): both input and output packed, via
  a scatter-on-write construction (`rc_idx` picks the read window,
  `mask_offset` picks the write window). `_1slot` is a replication-count
  optimization valid for L5 only — the L4 port (`_generic_p4`, `_silu_p4`)
  needs 2 replication slots (0 and 3), which **contradicts** the L5 finding
  rather than refining it.

`asm_unpacked_linear_240to8.asm` is the unpacked baseline all of the above
are measured against.

### Pack / unpack seams

`asm_packed_pack_192x64.asm` / `asm_packed_pack_240x16.asm` convert
unpacked→packed; `asm_packed_unpack_192x64.asm` / `asm_packed_unpack_240x16.asm`
convert packed→unpacked, at L4 and L5 respectively. Each pack/unpack kernel
fires `ACC.ADD.FIRST` at `p_out==0` for **every** chunk, not just the outer
peel — chunks don't accumulate across each other here, unlike layernorm's
single running sum.

### Residual add

`asm_packed_residual_add_192x64.asm` (L4) / `asm_packed_residual_add_240x16.asm`
(L5) — elementwise, so packing is free (exactly 8x fewer
cycles/instructions/XMEM bytes at L5, measured). These kernels assume
`R_MASK` is at its all-ones default on entry; they are only safe as the
first `R_MASK`-touching kernel in a fresh state, or immediately after an
explicit reload — chaining one directly after a packed-output-linear kernel
(which leaves `R_MASK` in a non-default state on exit) is a known hazard.

### Primitive A (cross-partition combine)

`asm_primitive_a_combine2x64.asm` (L4) / `asm_primitive_a_combine8x16.asm`
(L5) — standalone microbenchmarks of the store/reload workaround used
inline by the linear-family kernels above: no ISA instruction reduces
partition-wise (`AGG.SUM` collapses all active lanes to one scalar; `RESHAPE`
only permutes `r_acc`→`r_acc`; `CR15.partition` only feeds mask/shift math),
so combining N packed partitions costs a full store→reload→N×MULT.RC.VE
round trip.

## What's not here

`rc_idx` formulas, packing-factor derivations, replication-slot counts,
measured cycle/perf numbers, and the full corrected-vs-original narrative
all live in `kernel_docs/kernel_layer_map.md`'s "Packed activation layout"
section — this README intentionally doesn't reproduce them, to avoid
drifting into a second, staler copy of that document.
