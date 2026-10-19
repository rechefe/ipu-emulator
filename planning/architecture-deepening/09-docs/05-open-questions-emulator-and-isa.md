# Add the "Emulator and ISA" section to open-questions.md

Part of [Epic E09: Documentation restructure](00-epic.md) · ID E09-05 · Size S · Depends on: none · Behaviour change: none

## Goal

Every known open emulator and ISA question is written down in the one
open-questions page the project already has, with an ID, the evidence, a
proposal where one exists, and an owner. Readers, reviewers and other epics
link to an OQ ID instead of restating the problem.

## Background

- Decisions Q26, Q36 and Q54: one open-questions page, the existing
  `docs/content/requirements/open-questions.md`, with a new "Emulator and ISA"
  section and OQ numbers that continue the existing ones. The page today ends
  at **OQ-17** (section "Pipeline"), followed by the "Actions on other
  documents" table (A-1 to A-3).
- The page's rule (line 3): "To close a question, move its answer into the
  requirement and delete the row." The new section follows the same rule.
- Q25, Q27, Q35: AaQ is a placeholder and out of scope for the whole program.
  Its known defects, including the crashes, are recorded here and **not
  fixed**.
- Q53: accumulator overflow and NaN follow the proposed arithmetic-events
  requirement (`requirements/faults.md`, "Arithmetic events").
- Q57: LayerNorm keeps one reference that matches the kernel (no epsilon) until
  AaQ is revisited.
- Q56, Q59: ETISS is out of scope; F-8 is "not modelled".
- OQ-6 (AAQ behaviour and data types) and OQ-7 (store row format) already cover
  part of AaQ. The new rows link to them instead of repeating them.

## Implementation

Add this section after "Pipeline" and before "Actions on other documents".
Keep the existing table format. Re-verify every file:line at implementation
time; the numbers below are from `694a8a8`.

### Emulator and ISA

| ID | Question | Evidence | Proposal | Owner |
|---|---|---|---|---|
| OQ-18 | What does the emulator do on an FP32 accumulator overflow or a NaN result? | `ACC.ADD` with a lane sum above the FP32 maximum raises `OverflowError: float too large to pack with f format`; pinned by `test_execute.py:3139` and `test_wide_vector_debug.py:645`. | Follow the [arithmetic events](faults.md#arithmetic-events): store the IEEE result (±inf, NaN), set the per-event sticky flag, never halt (Q53). Emulator change in E04 (faults and sticky arithmetic flags). Close when the arithmetic-events requirement leaves "Proposed". | Yuval Harary |
| OQ-19 | What are the special values of the FP8 formats E1 to E7? | The emulator decodes every all-ones exponent as NaN, including mantissa 0, which IEEE-style decoders (ml_dtypes `float8_e4m3`, `float8_e5m2`) read as ±inf. Encoding saturates to the maximum finite value on overflow. E4's maximum is 240, not the 448 of OCP E4M3FN. E1 has no normal values. | Keep the current convention: all-ones exponent is NaN, saturate on overflow, no infinities. Add OCP formats only if the hardware has them. State the convention in the element-formats page (E09-19). | Unassigned |
| OQ-20 | What is the AaQ stage's behaviour? The emulator's `ACTIVATE.QUANTIZE` is a placeholder. | The emulator implements 13 exact functions in Python (`activations.py`) and clamps to INT8. The [AaQ spec](../specs/stage-aaq-str.md) has 7 function types evaluated through a LUT, a `LOAD` operation, an `e8m0` block scale and a 1033-bit `POST_AAQ_REG` payload. See also OQ-6 and OQ-7. | The AaQ spec is the hardware behaviour (as OQ-6). The emulator model is replaced when the spec is complete. Until then, every AaQ row below stays open and unfixed (Q35). | Unassigned |
| OQ-21 | What does `exp2` return for large inputs? | `activations.py:127` computes `math.exp(x * math.log(2.0))`, which raises `OverflowError` for x above about 1024. The prototype path does the same (`ipu.py:1731`). In wide FP32 mode, results between about 2^128 and 2^1024 are finite doubles, and the FP32 store then raises `float too large to pack`. | Decide with OQ-18 and OQ-20: return +inf and let the quantizer saturate. | Unassigned |
| OQ-22 | How does quantization treat ±inf and NaN? | Wide quantize calls `int(round(val))` (`ipu.py:1806`), which raises on ±inf (`OverflowError`) and NaN (`ValueError`). The INT32 wide path (`ipu.py:1790`) and the narrow path (`ipu.py:1827`) do the same. | Saturate ±inf to the format limits and map NaN to 0, before the integer conversion. Decide with OQ-19 and OQ-20. | Unassigned |
| OQ-23 | Which `POST_AAQ_REG` lanes does an activation write when fewer than 128 elements are valid? | The wide non-quantize path writes only the `active` lanes (`ipu.py:1780-1797`); lanes `active` to 127 keep the previous value. The narrow path zeroes the remaining bytes. `STR_POST_AAQ_REG` stores all 512 bytes, so stale data reaches XMEM. | Zero the inactive lanes, as the narrow path does. | Unassigned |
| OQ-24 | What do `reciprocal(0)` and `rsqrt(0)` return? | The AaQ spec pseudocode returns inf (`stage-aaq-str.md:188`, `:190`). The spec's function table returns 0 (`stage-aaq-str.md:205-206`). The emulator returns 0, and `rsqrt` returns 0 for every x ≤ 0 (`activations.py:128-131`). | Decide with OQ-20. The answer fixes OQ-27. | Unassigned |
| OQ-25 | How are the AaQ LUT, the `LOAD` operation and the block scale specified, and when does the emulator model them? | The spec describes a 256 × 17-bit LUT plus a 75-bit metadata block filled by `LOAD` through `lut_addr` (`stage-aaq-str.md:19`, `:84-87`, `:414-471`), and an `e8m0` block scale (`:104`, `:235-240`). The encodings of `asymetric` and `range_mode` are TBD (`:470`). §5.1 takes the maximum over `i in 0..254` for a 128-element batch (`:240`). The emulator models none of these. | Complete the spec first (owner of the AaQ spec). Then model it behind one AaQ interface in the emulator. | Unassigned |
| OQ-26 | Does the AaQ slot have a `LOAD` opcode? | The AaQ spec declares a 2-bit `op` with `AAQ_INST_OPCODE_LOAD = 2` (`stage-aaq-str.md:84`). The ISA and the generated SystemVerilog package define a 1-bit `aaq_inst_opcode_t` with `NOP` and `ACTIVATE_QUANTIZE` only. Adding `LOAD` changes `AAQ_SLOT_T_WIDTH` (9 to 10) and `IPU_COMPOUND_INST_WIDTH`, which is a breaking wire-format change. | Correct the spec table to the 1-bit opcode now. Add `LOAD` as its own ISA change when OQ-25 is answered. | Unassigned |
| OQ-27 | Does LayerNorm add an epsilon before `rsqrt`? | The kernel golden has no epsilon, because `rsqrt` maps a non-positive variance to 0 (`kernels/normalize/cases.py:15-17`). Full-layer and seam tests use eps = 1e-5. | One reference that matches the kernel, with no epsilon, until AaQ is defined (Q57). Revisit with OQ-24. | Unassigned |
| OQ-28 | How is F-8 (word stall limit) checked when the emulator has no stall model? | The emulator executes one word per cycle and never stalls, so F-8 can never fire. E04 models the other faults with their IDs and enable bits. | List F-8 as "not modelled" in the emulator and in the reference-model contract, until a cache-agent timing model exists (Q59). | Yuval Harary |
| OQ-29 | When, and how, does the reference model move to ETISS? | Out of scope for this program (Q56). The reference contract is language-neutral: a versioned JSON commit-record and trace schema plus a conformance suite stored as data (Q60), so another implementation can be checked against the same suite. | Keep the contract language-neutral. Decide timing and scope (whole model or fast path only) later. | Unassigned |

### Other edits on the page

- Add one sentence under the page's intro: questions OQ-18 onwards concern
  the emulator and the ISA definition, not only the hardware requirements.
- Do not change any existing row.

## Open questions for this issue

- A-1 ("Replace `BKPT` with `END`") conflicts with decision Q40 (`BKPT` becomes
  hardware `HALT`). Should this PR change A-1 to `HALT`, or is A-1 updated by the
  E03 issue that adds `HALT`? The requirements content is otherwise unchanged
  by E09.
- Owners marked "Unassigned" for AaQ rows: the user names the AaQ spec owner.

## Tests

- [ ] `bazel build //docs:build_docs` shows no new warnings, and the
      `faults.md#arithmetic-events` and `../specs/stage-aaq-str.md` links
      resolve.
- [ ] Every file:line in the table is re-checked against the PR's base commit
      (reviewer check).

## Acceptance Criteria

- [ ] The "Emulator and ISA" section exists in
      `docs/content/requirements/open-questions.md` with OQ-18 to OQ-29.
- [ ] Each row has evidence and an owner (a name or "Unassigned").
- [ ] Every other epic's issue that touches one of these topics (E04 faults and
      arithmetic flags, E04 numerics, E03 element formats, E07 normalize
      kernels) links the OQ ID instead of restating it.
- [ ] No existing row changes.
