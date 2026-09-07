# Kernel `.asm` and memory-map conventions

[Adding applications](adding-applications.md) covers the Python side of a kernel — the harness,
`SPEC`, `cases.py`, and the Bazel declaration. This page covers the two things it does not: the
**`.asm` header block** and how a harness **derives its memory map**.

Both are existing conventions, followed by 93 assembly files across the transformer kernel
families, written down here so they survive review rather than being copied from whichever
neighbour was open at the time.

---

## 1. The `.asm` header block

Every kernel assembly opens with a one-line description, then seven fields in this order:

```
# Multi-stream transformer projection matmul (Layer 5 QKV, P=4 pixel-streams).
#
# Layer:   L5
# Scope:   all-stream/P4
# Layout:  unpacked
# Shape:   240ch, P4 (4-stream), K=240->N_OUT=720
# Status:  validated
# Related: proj_outproj_240_p4 / proj_ffn1_240_p4 / proj_ffn2_240_p4 are the
#          other L5 P4 projections; shape suffix 144/192/240 = L3/L4/L5
# Tests:   test_proj_qkv_240_p4_wide (src/tools/ipu-apps/BUILD.bazel)
```

| Field | Meaning | Values |
|---|---|---|
| `Layer:` | Which network stage this kernel serves | `L3` / `L4` / `L5`, or `-` if shape-generic |
| `Scope:` | How much of the problem one invocation covers | `single-stream`, `all-stream/P4`, … |
| `Layout:` | On-device element packing | `unpacked` or `packed` |
| `Shape:` | The exact dimensions this `.asm` is generated for | free text, but state every dimension |
| `Status:` | Verification state | `validated`, or `new (<why>)` until a test covers it |
| `Related:` | Sibling kernels and how they differ | package names, real ones |
| `Tests:` | The Bazel target that exercises it | must exist in `BUILD.bazel` |

Notes:

- `Status: new` is for a kernel whose test does not exist yet. Promote it to `validated` in the
  same change that adds the test — a stale `new` is worse than no field.
- `Related:` earns its place by saying *how* the siblings differ, not merely listing them. A reader
  arriving at the wrong kernel should learn which one they wanted.
- `Tests:` is checkable and worth checking: every target named across the projection kernels
  resolves today.

Below the block, in order: the formula, the design narrative (loop nest, pipelining, why this
shape needs its own kernel), then the register map.

### Alias both LR *and* CR symbolically

Use Jinja `{% set %}` aliases and refer to them in the body:

```jinja
{% set data_ptr = "lr0"  %}  {# data row pointer, +1 row per k-step #}
{% set W_STRIDE = "cr8"  %}  {# weight rows per output channel #}
{% set DSTRUCT  = "cr15" %}  {# reserved dstructure register #}
```

A prose CR table drifts silently, because nothing breaks when it goes stale. One in the tree still
reads `CR7 = ROW_BYTES (512)` where the harness sets `1` — a leftover from the row-vs-byte
migration (issue #179). Symbolic aliases cannot drift that way: the body would stop assembling.

### Jinja cannot parameterize shape

`Template(text).render()` is called with **no context**, and the build passes no defines, so every
Jinja variable is a `{% set %}` local to that same file. Shape parameters must therefore travel in
CRs at runtime.

This is why near-identical `.asm` files exist per shape. It is a known constraint, not an
oversight — do not try to collapse them with templating alone.

---

## 2. Deriving the memory map

XMEM operands in `.asm` are **row numbers**, not byte addresses (issue #179). Element-addressed
`r_cyclic` operands are the exception and must not be rescaled.

Derive every region base from row counts, and never hardcode a byte address:

```python
ELEM_BYTES = 4                          # FP32
LANES      = 128                        # elements per XMEM row
ROW_BYTES  = LANES * ELEM_BYTES         # 512

DATA_ROWS   = K
WEIGHT_ROWS = N_OUT * W_STRIDE_ROWS

DATA_BASE_ROW    = 0
WEIGHTS_BASE_ROW = DATA_BASE_ROW + DATA_ROWS
OUTPUT_BASE_ROW  = WEIGHTS_BASE_ROW + WEIGHT_ROWS

DATA_BASE    = DATA_BASE_ROW * ROW_BYTES    # bytes, for xmem.write_address
WEIGHTS_BASE = WEIGHTS_BASE_ROW * ROW_BYTES
OUTPUT_BASE  = OUTPUT_BASE_ROW * ROW_BYTES
```

Keep the two families in parallel — `*_BASE_ROW` for the CR values, `*_BASE` for
`xmem.write_address`. A hardcoded byte map goes wrong the moment a dimension changes: regions
overlap and silently overwrite one another with no crash.

If the map depends on a **runtime** shape rather than module constants, compute it in a `_layout()`
method called from `__init__`. Do not *also* declare module-level constants for those same regions:
several apps carry base constants that `_layout()` immediately overrides, which reads as fact and
is not.

---

## 3. Padding

Padding is applied in the Python harness, never in the `.asm`. The assembly side only *masks*
already-padded regions — via `CR15.valid_elements`, a second dstructure CR for the tail, a resident
keep-mask vector, or an `R_MASK` bit image, depending on the kernel.

Two rules:

- **The pad value is kernel-semantic; never assume zero.** `softmax_columns` pads with `0.0`,
  while its sibling `softmax_columns_packed` must pad with `-1e30` (`NEG_PAD`) or its cross-group
  fold pollutes real columns with a spurious maximum.
- **Padding must not reach disk.** The output file carries the caller's layout, dense and unpadded;
  `teardown` un-packs. See the layout contract in
  [Adding applications](adding-applications.md).

Where padding leaves lanes idle, quantify it in `caveats` so the registry can report it — a kernel
running 16 tokens in a 128-lane row is not free, and must not claim `cost=0.0`.

`XMem.load_matrix_to` looks like a shared padding helper but is used by no app and pads to 128 B,
the wrong granularity for the 512 B FP32 wide row. Do not reach for it.

---

## Checklist

- [ ] Seven header fields present, in order, with a one-line description above them
- [ ] `Tests:` names a target that exists in `BUILD.bazel`
- [ ] `Status:` is `validated`, or `new` with a reason and a test to follow
- [ ] `Related:` says how the siblings differ
- [ ] Both LR and CR aliased via `{% set %}`; no prose-only CR table
- [ ] Region bases derived from row counts; no hardcoded byte addresses
- [ ] No module-level base constants that `_layout()` overrides
- [ ] Pad value chosen deliberately and justified where it is not zero
- [ ] Idle lanes disclosed in `caveats`
