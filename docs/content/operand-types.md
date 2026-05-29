# Operand types

This page is **generated** by `ipu_as.gen_docs` from `VALID_OPERAND_TYPES` in `instruction_spec.py` and the descriptions below. For encodings per instruction, see the [Instruction reference](instructions.md).

## `AaqRegIdx` {: #aaqregidx }

AAQ register selector: **`aaq0`** … **`aaq3`**.

## `ActivationFn` {: #activationfn }

AAQ-slot keyword on **`ACTIVATE`**: one of **identity**, **relu**, **relu6**, **sigmoid**, **tanh**, **gelu**, **softplus**, **elu**, **exp2** (see ``ACTIVATION_FN_NAMES`` in ``ipu_common.activations``). Emulator-only calibration (including α) and how **`POST_AAQ_REG`** (interim **512 B**) and **`STR_POST_AAQ_REG`** (store to XMEM) are described in **Building Applications** (`docs/content/building-applications.md#activations-emulator`).

## `AddSubSrcB` {: #addsubsrcb }

Second source for **`ADD`** / **`SUB`** in the LR slot: **`lr0`–`lr15`**, **`cr0`–`cr14`**, or an **unsigned 5-bit immediate** (`0`–`31`). Encoded in **6 bits**: **`0`–`31`** use the same ordering as **`LcrIdx`**; **`32`–`63`** encode immediates as **`32 + imm`**. `cr15` is reserved and is not a valid operand.

## `AggMode` {: #aggmode }

AAQ-slot immediate: aggregation mode for the `AGG` / `AGG.FIRST` instructions (sum / max family); see `acc_agg_enums`.

## `BreakImmediate` {: #breakimmediate }

16-bit value for **`BREAK`** / breakpoint slot conditions.

## `CrIdx` {: #cridx }

Constant-register index: **`cr0`** … **`cr14`**. CRs are **read-only** in assembly; the harness initializes them (e.g. base pointers, strides). `cr15` is reserved for dstructure configuration and is not a valid ISA operand. **`SET`** in the LR slot copies the full **32-bit** CR value into an LR.

## `ElementsInRow` {: #elementsinrow }

ACC-slot immediate: encoded **elements-per-row** selector (see `acc_stride_enums` in `ipu_common`).

## `FullXmemRow` {: #fullxmemrow }

1-bit control flag on **`AAQ`**: **`1`** = always process all **128 lanes** (full XMEM row, ignores ``CR15.valid_elements``); **`0`** = process only the first ``CR15.valid_elements`` lanes (clamped to 128) and zero the rest. Defaults to **`1`** for backward compatibility.

## `HorizontalStride` {: #horizontalstride }

ACC-slot immediate: **horizontal stride** bit pattern for `ACC.STRIDE` (see `acc_stride_enums`).

## `Label` {: #label }

Branch target: a symbolic **`label`** or a relative offset accepted by the cond slot (e.g. `loop`, `+3`).

## `LcrIdx` {: #lcridx }

LR **or** CR index in one field: lower indices map to **`lr0`–`lr15`**, higher indices to `**cr0`–`cr14`** in the usual combined ordering used by the assembler. `cr15` is reserved and is not a valid operand.

## `LrIdx` {: #lridx }

Loop register index: resolves to **`lr0`** … **`lr15`**. Often used for addresses, strides, and control values. When marked `read: live` in the spec, the emulator reads the **current** LR value after earlier slots in the same cycle.

## `LrModPow2KImmediate` {: #lrmodpow2kimmediate }

Four-bit immediate for **`INCR_MOD_POW2`**: encodes exponent **k** with semantic **k ∈ [1, 9]** as **(k − 1)** in the word.

## `MultMaskOffsetImmediate` {: #multmaskoffsetimmediate }

Unsigned **3-bit** immediate on multiply instructions: **`mask_offset`** selects slot **`0`**–**`7`**, each a **128-bit** region of **`R_MASK`** (eight mask slots total). **`mask_shift`** remains an **`LrIdx`**.

## `MultStageReg` {: #multstagereg }

Multiply-stage field in the VLIW encoding. Assembly accepts **`r0`** and **`r1`** only; the field is **two bits** wide (encoding `2` is reserved). Used as the destination of `LDR_MULT_REG` and as the **`ra`** operand of `MULT.EE`.

## `PostFn` {: #postfn }

AAQ-slot immediate: post-aggregation function selector (identity, inverse sqrt, etc.); see `acc_agg_enums`.

## `VerticalStride` {: #verticalstride }

ACC-slot immediate: **vertical stride** bit pattern for `ACC.STRIDE` (see `acc_stride_enums`).
