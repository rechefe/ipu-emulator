# Operand types

This page is **generated** by `ipu_gen.docs_pages` from `VALID_OPERAND_TYPES` in `instruction_spec.py` and the descriptions below. For encodings per instruction, see the [Instruction reference](instructions.md).

## `ActivationFn` {: #activationfn }

AAQ-slot keyword on **`ACTIVATE.QUANTIZE`**: one of **identity**, **relu**, **relu6**, **sigmoid**, **tanh**, **gelu**, **softplus**, **elu**, **exp2**, **reciprocal**, **rsqrt**, **silu**, **window** (see ``ACTIVATION_FN_NAMES`` in ``ipu_isa.activations``). Emulator-only calibration (including α and the **window** bounds a, b) is described in **Building Applications** (`docs/content/building-applications.md`).

## `AddbiImmediate` {: #addbiimmediate }

8-bit byte immediate for **`ADDBI`**: written as an unsigned value **`0`**–**`255`** or an equivalent signed **`-128`**–**`127`** literal (both spellings encode the same bit pattern). The encoded byte is reinterpreted as a signed two's-complement `int8` when broadcast-added to **`ADDB`**/**`ADDBI`**'s 8 destination byte elements.

## `BreakImmediate` {: #breakimmediate }

16-bit value for **`BREAK`** / breakpoint slot conditions.

## `CrIdx` {: #cridx }

Constant-register index: **`CR0`** … **`CR15`**. CRs are **read-only** in assembly; the harness initializes them (e.g. base pointers, strides). **`CR15`** conventionally holds dstructure configuration but is a valid `CrIdx` operand like any other CR. **`SET`** in the LR slot copies the full **32-bit** CR value into an LR.

**CR as an element scalar.** When a CR is a multiply's scalar element operand (**`MULT.VE`**, **`MULT.EE`**, a CR **`src`** of **`MULT.RC.VE`**), the scalar is the CR's low element-width bits decoded in the active element format: the low byte in INT8 and E1–E7, all 32 bits in INT32 and FP32 (so FP32 1.0 is the CR value `0x3F800000`). One rule for every format. **`CR0`** reads as 0 and **`CR1`** as 1 in the active format (1.0 in a float format), so a multiply by **`CR1`** is an identity in every format; as an LR or integer operand **`CR1`** is the integer 1.

## `DstructureCrIdx` {: #dstructurecridx }

Constant-register index: **`CR0`** … **`CR15`**, selecting which CR supplies the **valid element mask** / dstructure configuration (`valid_elements`, `partition`) for `AGG.SUM`, `AGG.SUM.FIRST`, `AGG.MAX`, `AGG.MAX.FIRST`, `ACTIVATE.QUANTIZE`, and the masking multiply instructions (`MULT.RC.VV`, `MULT.RC.VE`, `MULT.RC.VS`, `MULT.VE`, `MULT.EE`). Unlike **`CrIdx`**, **`CR15`** is allowed here — it's the conventional dstructure register — but the operand is always mandatory; there is no implicit fallback to `CR15` when it is omitted.

## `ElementsInRow` {: #elementsinrow }

ACC-slot immediate: elements per row for **`ACC.STRIDE`**. Valid values: **`16`**, **`32`**, **`64`** (minimum is 16; encoded 0→16, 1→32, 2→64). See `acc_stride_enums` in `ipu_isa`.

## `HorizontalStride` {: #horizontalstride }

ACC-slot immediate: horizontal stride mode for **`ACC.STRIDE`**. Valid values: **`off`**, **`on`**, **`on_inv`** (2-bit encoded enum; **`reserved3`** is reserved). Expand padding is fixed hardware behaviour and is not programmable. See `acc_stride_enums` in `ipu_isa`.

## `Label` {: #label }

Branch target: a symbolic **`label`** or a relative offset accepted by the cond slot (e.g. `loop`, `+3`).

## `LcrIdx` {: #lcridx }

LR **or** CR index in one field: lower indices map to **`LR0`–`LR15`**, higher indices to **`CR0`–`CR15`** in the usual combined ordering used by the assembler. Used as **`src_b`** on **`ADD`**/**`SUB`**, as **`step`** on **`INC.MODP2`**, and as **`src`** on **`MULT.RC.VE`** (an LR's stored value selects an element from `R0`/`R1`; a CR is the scalar directly, decoded as in [CrIdx](#cridx)).

## `LrIdx` {: #lridx }

Loop register index: resolves to **`LR0`** … **`LR15`**. Often used for addresses, strides, and control values. When marked `read: live` in the spec, the emulator reads the **current** LR value after earlier slots in the same cycle.

## `LrIncDecImmediate` {: #lrincdecimmediate }

Unsigned immediate for **`INC`** / **`DEC`** in the LR slot. The bit width **W** is not hardcoded — it is derived from the LR slot union layout so the total slot width stays constant. Valid range: **`0`** to **`2^W − 1`**.

## `LrModPow2KImmediate` {: #lrmodpow2kimmediate }

Four-bit immediate for **`INC.MODP2`**: encodes exponent **k** with semantic **k ∈ [1, 9]** as **(k − 1)** in the word.

## `LrOrReshapeMaskImmediate` {: #lrorreshapemaskimmediate }

Operand on **`ACC.RESHAPE`** (bit width auto-derived from union layout): selects the first participating element slot (elements **`reshape_mask`**…**`7`** of the source/dest LRD pair participate). Accepts either an unsigned immediate **`0`**–**`7`** or an LR register (its value is used directly as the mask; a resolved value greater than **`8`** raises an error rather than being clamped). **`0`** uses all 8 slots; **`7`** uses only slot 7; **`8`** uses none.

## `LrdIdx` {: #lrdidx }

Register-pair index: **`LRD0`**, **`LRD2`**, **`LRD4`**, … **`LRD14`** — named after the lower register in the pair — each aliasing two adjacent **`LR`** registers as one 8-byte-element value: **`LRDn`** = **`LR(n+1)`** (high elements) concatenated with **`LR(n)`** (low elements). Purely an assembler/emulator view; there is no separate physical storage. Used as **`dest`** on **`ADDB`** / **`ADDBI`**.

## `MultMaskOffsetImmediate` {: #multmaskoffsetimmediate }

Unsigned **3-bit** immediate on multiply instructions: **`mask_offset`** selects slot **`0`**–**`7`**, each a **128-bit** region of **`R_MASK`** (eight mask slots total). **`mask_shift`** remains an **`LrIdx`**.

## `MultStageReg` {: #multstagereg }

Multiply-stage field in the VLIW encoding. Assembly accepts **`R0`** and **`R1`** only; the field is **two bits** wide (encoding `2` is reserved). Used as the destination of `LD.MULT` and as the **`ra`** operand of `MULT.RC.VV`.

## `VerticalStride` {: #verticalstride }

ACC-slot immediate: **vertical stride** bit pattern for `ACC.STRIDE` (see `acc_stride_enums`).
