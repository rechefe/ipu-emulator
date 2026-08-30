# MambaVision kernels

Five IPU applications implementing the data-movement and residual operations of
`MambaVisionMixer` and of the MLP that follows it inside `Block`, from the NVIDIA
reference model `mamba_vision.py`.

| App | Reference line it implements |
|---|---|
| `reshape_token_view` | `rearrange(xz, "b l d -> b d l")` and `rearrange(y, "b d l -> b l d")` |
| `split_xz` | `x, z = xz.chunk(2, dim=1)` |
| `concat_yz` | `y = torch.cat([y, z], dim=1)` |
| `residual_mixer` | `x = x + drop_path(gamma_1 * mixer(norm1(x)))` |
| `residual_mlp` | `x = x + drop_path(gamma_2 * mlp(norm2(x)))` |

Each follows the `fully_connected` layout: an `.asm`, an `IpuApp` subclass in
`__init__.py`, a `__main__.py` debug runner, `test_data_format/<stage>/*.bin`, and
a regression test under `test/`. Everything is INT8 with an INT32 accumulator.

---

## Contents

1. [The four machine facts everything follows from](#1-the-four-machine-facts-everything-follows-from)
2. [Where the kernels come from](#2-where-the-kernels-come-from)
3. [The shapes](#3-the-shapes)
4. [XMEM layouts](#4-xmem-layouts)
5. [Three idioms every kernel is built from](#5-three-idioms-every-kernel-is-built-from)
6. [reshape_token_view](#6-reshape_token_view)
7. [split_xz](#7-split_xz)
8. [concat_yz](#8-concat_yz)
9. [residual_mixer and residual_mlp](#9-residual_mixer-and-residual_mlp)
10. [Measured cost](#10-measured-cost)
11. [File map and Bazel targets](#11-file-map-and-bazel-targets)
12. [How this was verified](#12-how-this-was-verified)
13. [Known limitations and next steps](#13-known-limitations-and-next-steps)

---

## 1. The four machine facts everything follows from

**Everything is 128 elements wide.** An XMEM row is 128 bytes = 128 INT8 elements.
`R0`/`R1`, `R_CYCLIC` and `R_ACC` all hold 128-element vectors; `R_ACC` is
128 × INT32 = 512 bytes.

**There is no move and no shuffle.** The only path into `R_ACC` runs through the
multiplier, and there is no permute/gather network. So moving a row means
*multiplying it by 1*, and moving element *i* of one line to element *j* of
another means *multiplying by the one-hot vector e_j and accumulating*. Those two
observations shape all five kernels.

**A multiply's scalar can come straight from a CR.** `MULT.RC.VE`'s `src` operand
is an `LcrIdx`: naming an LR uses that LR's value as an index into `R0 ++ R1`,
but naming a **CR** takes the CR's low byte as the scalar directly
(`Ipu._mult_resolve_lcr_scalar`). Since `CR1` is permanently 1, a copy needs no
constants row in XMEM and no `LDR_MULT_REG` at all.

**MULT reads its register data from the start-of-word snapshot** (emulator issues
#157 / #172), while LR operands are read live and LR sub-slots resolve before
LOAD/MULT within a word. That is why every loop here is software-pipelined: the
row a multiply consumes must have been loaded at least one word earlier. The
comment block in `fully_connected.asm` says the same thing; these kernels reuse
its loop shape deliberately.

Two constraints that bite in practice:

* **`.asm` XMEM operands are row numbers, not byte addresses.** The Python
  `setup()` converts. In narrow (INT8) mode only rows 0–16383 are addressable —
  the first 2 MB of the 8 MB allocation (`Ipu._xmem_row_addr`), which is why the
  buffer bases below all sit well under 16384.
* **`R_MASK` defaults to all-ones**, so passing mask offset 0 with a shift of 0
  leaves every element active without ever loading the mask register.

---

## 2. Where the kernels come from

```python
class Block(nn.Module):
    def forward(self, x):                      # x: (B, L, D)  token view
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))   # residual_mixer
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))     # residual_mlp
        return x


class MambaVisionMixer(nn.Module):
    def forward(self, hidden_states):          # (B, L, D)
        xz = self.in_proj(hidden_states)                    # fully_connected app
        xz = rearrange(xz, "b l d -> b d l")                # reshape_token_view (t2c)
        x, z = xz.chunk(2, dim=1)                           # split_xz
        ...
        x = F.silu(F.conv1d(x, ...))                        # depthwise conv + SiLU
        z = F.silu(F.conv1d(z, ...))
        y = selective_scan_fn(x, dt, A, B, C, ...)          # the SSM scan
        y = torch.cat([y, z], dim=1)                        # concat_yz
        y = rearrange(y, "b d l -> b l d")                  # reshape_token_view (c2t)
        out = self.out_proj(y)                              # fully_connected app
        return out
```

`DropPath` is `nn.Identity` at inference and `mamba_vision_T` passes no
`layer_scale`, so `Block.__init__` leaves `gamma_1` and `gamma_2` as the Python
int `1` — both residuals are plain element-wise adds. The kernels still route the
branch through a gamma scalar, which costs nothing because it shares the multiply
that gets the row into `R_ACC`; `mamba_vision_B`/`_L` (`layer_scale=1e-5`) only
need a different value in `CR6`.

`reshape_token_view` runs twice per mixer, in opposite directions. Both are the
same transpose, so there is one program and two CR configurations
(`direction="t2c"` / `"c2t"`).

---

## 3. The shapes

`mamba_vision_T` is built with `dim=80`, `depths=[1,3,8,4]`,
`window_size=[8,8,14,7]`, `mlp_ratio=4`, `resolution=224`. `MambaVision.__init__`
gives level *i* `dim * 2**i` channels and uses `ConvBlock` for levels 0–1, so the
mixer only exists in levels 2 and 3. `MambaVisionLayer.forward` calls
`window_partition` first, so `L = window_size²`. `Block.__init__` builds the mixer
with `expand=1`, hence `d_inner = d_model`.

| | **stage3** (level 2) | **stage4** (level 3) |
|---|---|---|
| feature map (224 input) | 14 × 14 | 7 × 7 |
| `window_size` | 14 | 7 |
| `L` tokens per window | **196** | **49** |
| `d_model` | **320** | **640** |
| `d_inner` = `expand * d_model` | 320 | 640 |
| `half` = `d_inner // 2` | 160 | 320 |
| `mlp_hidden` = `4 * d_model` | 1280 | 2560 |
| mixer blocks in the stage | 4 (of 8) | 2 (of 4) |

Each kernel carries this as a small `STAGES` dict, the way `fully_connected`
carries its `#define`-style constants — nothing else hard-codes a number.

Neither stage is a clean multiple of 128 in both axes, which is where most of the
padding discussion below comes from. Stage 3: 320 = 2.5 rows, 196 = 1.53 rows.
Stage 4: 640 = 5 rows exactly, but 49 = 0.38 of a row.

---

## 4. XMEM layouts

A tensor lives in XMEM as a stack of 128-element rows. One *logical line* occupies
`ceil(len / 128)` consecutive rows, zero-padded in the tail. Two layouts, matching
the two `rearrange` calls:

```
TOKEN VIEW   (B, L, D)   "b l d"     line = one token,   length D
             token t at row  base + t * ceil(D/128)

CHANNEL VIEW (B, D, L)   "b d l"     line = one channel, length L
             channel c at row  base + c * ceil(L/128)
```

For stage 3 (`D = d_inner = 320`, `L = 196`):

```
token view  (196, 320)                    channel view (320, 196)
  row 0   token 0, channels   0..127        row 0  channel 0, tokens   0..127
  row 1   token 0, channels 128..255        row 1  channel 0, tokens 128..195 + 60 zeros
  row 2   token 0, channels 256..319 + 64 zeros
  row 3   token 1, channels   0..127        row 2  channel 1, tokens   0..127
  ...     3 rows per token, 588 rows        ...    2 rows per channel, 640 rows
```

**Why zero-pad instead of masking.** `CR15` could mark a line partially valid, but
every kernel here sets `valid_elements = 128` and pads with zeros. Padding costs a
few wasted elements; masking would cost a conditional in the inner loop. For pure
data movement the padding is strictly cheaper, and it makes the reference data an
exact byte image.

**The 4-row store trick.** `STR_POST_AAQ_REG` always writes the whole 512-byte
register — 4 XMEM rows — even though only the leading 128 bytes carry the INT8
result; the rest is cleared. `fully_connected.asm` handles this by striding its
output 4 rows apart. These kernels do the opposite: they store to **consecutive**
rows, so each store overwrites the three zero rows the previous one left, and the
output buffer ends up **packed**.

```
store row 0  ->  [ data0 ][  0   ][  0   ][  0   ]
store row 1  ->          [ data1 ][  0   ][  0   ][  0   ]
store row 2  ->                  [ data2 ][  0   ][  0   ][  0   ]
result:          [ data0 ][ data1 ][ data2 ][  0   ][  0   ][  0   ]
                                            ^^^^^^^^^^^^^^^^^^^^^^ 3 rows of slack
```

Every output buffer therefore needs **3 rows of slack** after it. In exchange one
kernel's output is directly usable as the next kernel's input with no repacking
pass — which is what makes `test_mamba_vision_block.py` possible.

---

## 5. Three idioms every kernel is built from

### 5.1 Multiply by one — the copy

```
R_CYCLIC <- the row to copy
MULT.RC.VE lr15 cr1 0 lr15 cr15   ->   1 * row   ->   ACC.ADD.FIRST  ->  R_ACC
ACTIVATE.QUANTIZE identity cr15   ->   POST_AAQ_REG
STR_POST_AAQ_REG lr7 cr3          ->   XMEM
```

`CR1` is permanently 1 and `MULT.RC.VE`'s scalar operand accepts a CR, so the
scalar is free: no constants row, no `R0` load, no prologue beyond priming
`R_CYCLIC`. `split_xz`, `concat_yz` and both residuals are built from this. Worth
saying out loud when reading the utilization numbers: **for those four kernels the
multiplier is a data path, not an ALU** — the metric that matters is bytes per
cycle, not MACs.

### 5.2 The cyclic one-hot table — the transpose

`reshape_token_view` needs the one-hot vector e_j for every output element.
Keeping 128 one-hot rows in XMEM would need a second load per inner iteration and
there is only one load slot per word — it would double the kernel's cost.

Instead: `R_CYCLIC` holds **512** elements and `MULT.RC.VE` reads it as a
*wrapping* 128-element window starting at an element index. Load it once with a
buffer that is zero everywhere except element 128:

```
R_CYCLIC = [ 0 x128 , 1 , 0 x383 ]
                      ^ element 128

window starting at element (128 - j)  =>  a 1 at position j, zeros elsewhere  =  e_j
```

| window start (`rc_idx`) | 128 | 127 | 126 | … | 2 | 1 |
|---|---|---|---|---|---|---|
| one-hot position *j* | 0 | 1 | 2 | … | 126 | 127 |

Four prologue loads replace 128 loads per output row, and the inner loop's single
load slot stays free for source data. Walking *j* upward is one
`SUB lr5 lr5 cr1` on the window start.

### 5.3 INT32 accumulate, clamp once on the way out

Everything sums in `R_ACC` at INT32; `ACTIVATE.QUANTIZE identity` brings the
result back to INT8 exactly once, at the end. In the emulator's INT8 mode that is
a direct clamp to [-128, 127] (`Ipu.execute_activate_quantize`).

For the three data-movement kernels the values are already valid INT8, so the
clamp is the identity and the output is an exact permutation of the input bytes.
Only the residuals can saturate; the reference data is generated so that ~21 % of
the sums do, which means the regression tests actually cover that path.

---

## 6. reshape_token_view

**What it computes.** Transpose an INT8 matrix between row-padded layouts:

```
input :  M lines of N elements,  line m at  SRC_BASE + m*SPL       SPL = ceil(N/128)
output:  N lines of M elements,  line n at  DST_BASE + n*DPL       DPL = ceil(M/128)
```

with `out[n][m] = in[m][n]`. Two configurations:

| direction | source | result | reference line |
|---|---|---|---|
| `t2c` | token view `(L, d_inner)` | channel view `(d_inner, L)` | `rearrange(xz, "b l d -> b d l")` |
| `c2t` | channel view `(d_inner, L)` | token view `(L, d_inner)` | `rearrange(y, "b d l -> b l d")` |

**The algorithm.** Output row *(n, b)* — block *b* of output line *n*, covering
source lines `128b … 128b+127` — is

```
R_ACC  =  sum over j in [0,128)  of   in[128*b + j][n] * e_j
```

128 masked broadcast-multiply-accumulate steps, with e_j from the cyclic table of
§5.2. The scalar `in[128b+j][n]` is element `n % 128` of `R0` after loading XMEM
row `(128b+j)*SPL + n//128`.

**The loop nest.**

```
for n in range(N):                       # LR0, output line
    nRow, nIdx = divmod(n, 128)          # LR2, LR3 — maintained incrementally
    for b in range(DPL):                 # LR1, output block
        R_ACC = 0
        for j in range(128):             # LR6
            R_ACC += R0[nIdx] * e_j      # R0 = row (128b+j)*SPL + nRow
        store to DST_BASE + n*DPL + b
```

`divmod(n, 128)` is kept without a divider: `INCR_MOD_POW2 lr3 cr1 7` advances
`nIdx` mod 128, and `BNE lr3 cr0` bumps `nRow` on the wrap.

**Software pipelining.** Because MULT reads `R0` from the start-of-word snapshot,
element 0's row is loaded in a priming word and each body loads element *j+1*'s
row while multiplying element *j*'s — the same two-word `pre`/`body` shape
`fully_connected.asm` uses:

```
element_loop_pre:
    ADD  lr4 lr4 cr5 ;        # source row -> next element
    SUB  lr5 lr5 cr1 ;        # window start -> this element's one-hot
    ADD  lr6 lr6 cr1 ;;
element_loop:
    MULT.RC.VE  lr5 lr3 0 lr15 cr15 ;
    ACC.ADD ;
    LDR_MULT_REG r0 lr4 cr2 ;
    BNE lr6 cr9 element_loop_pre ;;
```

**Padding requirement.** The inner loop is unconditionally 128 iterations, so the
caller must zero-fill source lines `M … 128*DPL-1`. Those zero lines land in
exactly the padding elements of the output. `setup()` zeroes the region before
loading, and the shipped `*_in_int8.bin` files already contain the padding.

**Register map.**

| CR | | CR | | LR | | LR | |
|---|---|---|---|---|---|---|---|
| 2 | `SRC_BASE_ROW` | 9 | 128 | 0 | `n` | 6 | `j` |
| 3 | `DST_BASE_ROW` | 10 | `128*SPL` | 1 | `b` | 7 | dest row `n*DPL+b` |
| 4 | `DPL` | 11 | `ONEHOT_BASE_ROW` | 2 | `nRow` | 8 | src block base |
| 5 | `SPL` | 13 | 256 | 3 | `nIdx` | 9 | `n*DPL` |
| 6 | 129 | 14 | 384 | 4 | src row | 10, 11 | prologue scratch |
| 7 | `N` | 15 | dstructure | 5 | window start | 15 | 0 |
| 8 | `DPL` (blocks) | | | | | | |

**Cost.** `N × (DPL × 263 + 5) + 11 + N//128` VLIW words — see §10.

---

## 7. split_xz

**What it computes.** `x, z = xz.chunk(2, dim=1)` on the channel-view tensor
`(d_inner, L)`. Because the tensor is already in channel view when the chunk
happens, the split is along the *line* axis of the XMEM layout: the first
`half = d_inner//2` channel lines become x, the rest become z. Two back-to-back
row copies over one monotonically increasing source pointer.

```
rows [0, HALF_ROWS)              ->  DSTX_BASE      HALF_ROWS = half * ceil(L/128)
rows [HALF_ROWS, 2*HALF_ROWS)    ->  DSTZ_BASE
```

The x loop's last iteration already prefetched source row `HALF_ROWS`, which is
z's first row, so the z loop starts with no re-priming — one word of hand-off.

**The four-word body.**

```
x_loop:
    MULT.RC.VE lr15 cr1 0 lr15 cr15 ;   # 1 * row, scalar straight from CR1
    ACC.ADD.FIRST ;
    ADD lr4 lr4 cr1 ; ADD lr6 lr6 cr1 ;;
    ACTIVATE.QUANTIZE identity cr15 ;
    LDR_CYCLIC_MULT_REG lr4 cr2 lr15 ;;  # prefetch the next row
    STR_POST_AAQ_REG lr7 cr3 ;;
    ADD lr7 lr7 cr1 ;
    BLT lr6 lr5 x_loop ;;
```

The store sits in its own word rather than sharing one with the `ADD lr7` that
advances it, because LR sub-slots resolve *before* the store slot and the store
would otherwise use the bumped pointer.

> **On silicon this kernel is free.** x and z are disjoint address ranges of the
> same cache table, so the RISC core can hand the two `conv1d` kernels different
> base addresses and no data moves at all. The copy version exists because the
> downstream kernels take independent buffers, and because it gives the zero-copy
> claim a measured baseline. See §13.

---

## 8. concat_yz

The exact inverse: `y = torch.cat([y, z], dim=1)` stacks the selective-scan output
and the gated branch, both channel-view `(half, L)`, into one `(d_inner, L)`
buffer for `out_proj`.

```
DST rows [0, HALF_ROWS)             <-  y   (SRCY_BASE)
DST rows [HALF_ROWS, 2*HALF_ROWS)   <-  z   (SRCZ_BASE)
```

Same four-word copy body. The only structural difference from `split_xz` is the
hand-off: here the *source* changes buffer while the destination pointer keeps
counting, so `LR4` is rewound and `R_CYCLIC` re-primed from `SRCZ_BASE` — two
words instead of one. The same zero-copy remark applies.

---

## 9. residual_mixer and residual_mlp

**What they compute.** `out = skip + gamma * branch`, element-wise over token-view
`(L, d_model)` tensors, INT32 in `R_ACC`, clamped to INT8 on the way out.

| | `residual_mixer` | `residual_mlp` |
|---|---|---|
| statement | `x = x + gamma_1 * mixer(norm1(x))` | `x = x + gamma_2 * mlp(norm2(x))` |
| `skip` | block input `x` | the mixer residual's output |
| `branch` | mixer output after `out_proj` | timm `Mlp` output (`fc1`→GELU→`fc2`) |

They are separate applications — separate `.asm`, separate Bazel targets, separate
latency budgets — because they are separate points in the block schedule with
different gammas and very different branch latencies.

**The five-word body**, one row per iteration:

```
residual_loop:
    MULT.RC.VE lr15 cr1 0 lr15 cr15 ;    # skip row * 1
    ACC.ADD.FIRST ;
    LDR_CYCLIC_MULT_REG lr4 cr4 lr15 ;;  # fetch the matching branch row

    MULT.RC.VE lr15 cr6 0 lr15 cr15 ;    # branch row * gamma (CR6's low byte)
    ACC.ADD ;
    ADD lr4 lr4 cr1 ; ADD lr6 lr6 cr1 ;;

    ACTIVATE.QUANTIZE identity cr15 ;
    LDR_CYCLIC_MULT_REG lr4 cr2 lr15 ;;  # prefetch the next skip row

    STR_POST_AAQ_REG lr7 cr3 ;;

    ADD lr7 lr7 cr1 ;
    BLT lr6 lr5 residual_loop ;;
```

Both addends flow through `R_CYCLIC` exactly one word apart, which is the snapshot
distance MULT needs. Two loads, two multiplies, one store per 128 elements.

---

## 10. Measured cost

`bazel run` each app with the debug runner and it prints
`state.stats.format_summary()`. The numbers below were measured on `ipu_emu` over
the shipped reference data.

| Kernel | Stage | Input → output | Cycles | MULT active | ACC active | XMEM rd / wr | Elements/cycle |
|---|---|---|---|---|---|---|---|
| reshape `t2c` | 3 | (196, 320) → (320, 196) | 169 933 | 48.2 % | 48.2 % | 82 564 / 640 | 0.37 |
| reshape `c2t` | 3 | (320, 196) → (196, 320) | 155 636 | 48.4 % | 48.4 % | 75 856 / 588 | 0.40 |
| reshape `t2c` | 4 | (49, 640) → (640, 49) | 171 536 | 47.8 % | 47.8 % | 82 564 / 640 | 0.18 |
| reshape `c2t` | 4 | (640, 49) → (49, 640) | 64 691 | 48.5 % | 48.5 % | 31 609 / 245 | 0.48 |
| split_xz | 3 | (320, 196) → 2 × (160, 196) | 2 565 | 25.0 % | 25.0 % | 641 / 640 | 24.5 |
| split_xz | 4 | (640, 49) → 2 × (320, 49) | 2 565 | 25.0 % | 25.0 % | 641 / 640 | 12.2 |
| concat_yz | 3 | 2 × (160, 196) → (320, 196) | 2 566 | 24.9 % | 24.9 % | 642 / 640 | 24.4 |
| concat_yz | 4 | 2 × (320, 49) → (640, 49) | 2 566 | 24.9 % | 24.9 % | 642 / 640 | 12.2 |
| residual (each) | 3 | 2 × (196, 320) → (196, 320) | 2 944 | 39.9 % | 39.9 % | 1 177 / 588 | 21.3 |
| residual (each) | 4 | 2 × (49, 640) → (49, 640) | 1 229 | 39.9 % | 39.9 % | 491 / 245 | 25.5 |

Closed forms, exact against every row above:

| Kernel | Cycles |
|---|---|
| `reshape_token_view` | `N × (DPL × 263 + 5) + 11 + N//128` |
| `split_xz` | `5 + 8 × HALF_ROWS` |
| `concat_yz` | `6 + 8 × HALF_ROWS` |
| `residual_mixer` / `residual_mlp` | `4 + 5 × NROWS` |

**Reading these numbers.**

*The transpose dominates, by ~58×.* Each 128-wide multiply contributes exactly one
element to the output, so the *arithmetic* efficiency of the transpose is capped at
**1/128 = 0.78 %** of the multiplier array — intrinsic to transposing on a machine
with no shuffle network, not an artefact of this implementation. Note the two
different senses of "utilization" the stats invite: the MULT *slot* is busy ~48 %
of words (the other half is the `pre` word), while the 128 multiplier *elements*
are 0.78 % useful.

*Padding shows up asymmetrically.* Stage 4's `t2c` costs 171 536 cycles to move
31 360 elements because the output is 640 channel lines of only 49 valid tokens
each — only 38 % of each output row is real. Its `c2t` direction, moving the same
data the other way, costs 64 691 and wastes nothing (640 = 5 × 128 exactly). Same
tensor, 2.7× the cost, purely from which axis lands in the 128-element row.

*The copies and residuals are bandwidth-bound and cheap.* At 4 and 5 words per
128-element row they run at 32 and 25.6 bytes/cycle, and the multiplier does no
useful arithmetic at all — for them the 25 % / 40 % MULT-active figures measure
data-path occupancy, not compute.

---

## 11. File map and Bazel targets

```
src/tools/ipu-apps/
├── BUILD.bazel                     + 5 assemble_asm, 5 py_binary, 6 py_pytest_test
├── README.md                       + "MambaVision kernels" section
├── MAMBAVISION_KERNELS.md          this document
├── src/ipu_apps/
│   ├── reshape_token_view/  reshape_token_view.asm, __init__.py, __main__.py,
│   │                        test_data_format/{stage3,stage4}/{t2c,c2t}_{in,out}_int8.bin
│   ├── split_xz/            split_xz.asm, …, {xz_in,x_out,z_out}_int8.bin
│   ├── concat_yz/           concat_yz.asm, …, {y_in,z_in,yz_out}_int8.bin
│   ├── residual_mixer/      residual_mixer.asm, …, {skip_in,branch_in,out}_int8.bin
│   └── residual_mlp/        residual_mlp.asm, …, {skip_in,branch_in,out}_int8.bin
└── test/
    ├── test_reshape_token_view.py  golden + round-trip
    ├── test_split_xz.py            golden + halves reconstruct the input
    ├── test_concat_yz.py           golden + result is y ++ z
    ├── test_residual_mixer.py      golden + clamp coverage
    ├── test_residual_mlp.py        golden + clamp coverage
    └── test_mamba_vision_block.py  all five chained as Block.forward sequences them
```

Each app is self-contained in the `fully_connected` style: constants, a
`parse_stage` helper and one `IpuApp` subclass per `__init__.py`, with no shared
support modules. Nothing outside these directories changed except `BUILD.bazel`
and `README.md`; `src/ipu_apps/__init__.py` is untouched. The apps declare no
`KernelSpec`, so the kernel registry does not pick them up — same as
`fully_connected`.

```bash
# run a kernel
bazel run //src/tools/ipu-apps:reshape_token_view -- --stage stage3 --direction t2c
bazel run //src/tools/ipu-apps:split_xz           -- --stage stage4
bazel run //src/tools/ipu-apps:concat_yz          -- --stage stage3
bazel run //src/tools/ipu-apps:residual_mixer     -- --stage stage3
bazel run //src/tools/ipu-apps:residual_mlp       -- --stage stage4

# regression tests
bazel test //src/tools/ipu-apps:test_reshape_token_view
bazel test //src/tools/ipu-apps:test_split_xz
bazel test //src/tools/ipu-apps:test_concat_yz
bazel test //src/tools/ipu-apps:test_residual_mixer
bazel test //src/tools/ipu-apps:test_residual_mlp
bazel test //src/tools/ipu-apps:test_mamba_vision_block
```

---

## 12. How this was verified

**Every kernel was assembled and executed on `ipu_emu`, and every output matched
its reference file byte for byte** — 14 cases: four reshape (2 stages × 2
directions), split and concat (2 stages each, both outputs compared), and both
residuals (2 stages each). The cycle counts in §10 are what the emulator reported,
and they reproduce the closed forms exactly.

**All six test modules were then executed end to end — 26 test cases, 26 passing**,
including `test_mamba_vision_block.py`'s full mixer round trip on both stages.

Beyond the golden compares:

* **The reference data is derived from the model's own semantics.** The generator
  runs the exact tensor ops from `mamba_vision.py` — `permute`, `chunk(2, dim=1)`,
  `torch.cat(..., dim=1)`, `x + gamma * branch` with an INT8 clamp — and the
  residual data is drawn so that ~21 % of the sums saturate, so the tests cover
  the clamp rather than just the add.
* **The transpose was checked dimension by dimension**, on the real emulator, with
  reduced shapes chosen to isolate each axis of the loop nest: single block,
  `DPL`=2, `DPL`=3, `N` just past 128 (the `nIdx` wrap), `SPL`=2, and
  `DPL`=2/`SPL`=3 with two wraps. All pass.
* **The tests assert structural properties too**, not just byte equality, so they
  stay meaningful if the reference data is regenerated: the transpose round-trips
  to the identity, the split's two halves concatenate back to its input, the
  concat's output is `y ++ z`, and `test_mamba_vision_block.py` chains
  reshape → split → concat → reshape with the scan replaced by a pass-through and
  checks the block gives its input back byte for byte.

One caveat on the sandbox this was developed in: `lark` could not be installed, so
the `.asm` was encoded for the emulator by a small purpose-built encoder driven by
the project's own `INSTRUCTION_SPEC`, `PSEUDO_INSTRUCTION_SPEC`,
`SLOT_UNIONS.opcode_bindings` and token classes rather than by `ipu-as` itself.
That encoder was gated on re-running the repository's own `fully_connected.asm`
and reproducing `out_int8_acc_int32.bin`, so it agrees with the assembler on every
construct these kernels use. `bazel test //src/tools/ipu-apps:test_*` with the real
`ipu-as` in the loop remains the authoritative check and should be run once.

---

## 13. Known limitations and next steps

**The transpose is the whole cost.** At `N × DPL × 263` words it is ~58× everything
else combined. Three things would help, in order of payoff:

1. **Don't transpose.** The `rearrange` calls exist because PyTorch needs a
   contiguous tensor for `conv1d` and the scan. On the IPU the layout is a property
   of how the *next* kernel walks its addresses. If `x_proj`, the depthwise
   `conv1d` and `out_proj` are written to consume whichever view they are handed,
   `reshape_token_view` leaves the schedule entirely. Biggest win available in this
   block, and worth doing before optimizing the kernel itself.
2. **`ACC.RESHAPE`.** The ACC slot has a `RESHAPE` opcode (`source`, `dest`,
   `reshape_mask`). Nothing here uses it; if it can move elements across positions
   within `R_ACC` it could replace the one-hot broadcast entirely and change the
   cost class of this kernel. **Worth checking first.**
3. **A one-word inner loop.** The body is two words (`pre` + `body`) purely to keep
   LR updates out of the same word as the load and branch that consume them — the
   conservative shape `fully_connected.asm` uses. Confirming the LR-before-LOAD and
   LR-before-COND ordering within a word would collapse it to one word and halve
   the kernel.

**Split and concat should be zero-copy.** Both are contiguous range aliases, not
data movement. With the cache-table model (`table_id` + `dram_base_address`) the
RISC core can point the consumers at sub-ranges and skip both kernels; they are
implemented here so the port runs end to end on independent buffers, and so the
saving has a measured baseline of 2 565 + 2 566 cycles per mixer invocation.

**Padding is chosen per-axis, not per-tensor.** Stage 4's `t2c` wastes 62 % of its
output elements because 49 tokens sit in a 128-element row. Packing several windows
into one line would recover most of that, at the cost of a more complicated address
generator.

**Not covered by these five kernels**, and needed for a complete mixer: the
depthwise `conv1d` + SiLU on both branches, `x_proj` / `dt_proj`, the selective scan
itself, and the two `LayerNorm`s. `in_proj` and `out_proj` are the existing
`fully_connected` app; GELU and SiLU are `ACTIVATE.QUANTIZE` activation codes.

**Wide (FP32) debug mode is not wired up.** Everything here is INT8 with an INT32
accumulator. Wide-vector mode would need different load sizes and `STR_ACC_REG`
instead of the quantized store path, and the row arithmetic in each `__init__.py`
would need a per-element-size parameter.
