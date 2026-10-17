# Program CRs from `[abi]`: region bases, integers, element scalars

Part of [Epic: Kernel framework](00-epic.md) · ID E06-06 · Size S · Depends on: E06-04, E06-05, E04 (Q49: CR scalars decoded in the active format) · Behaviour change: none

## Goal

Write every CR a kernel needs from its `[abi]` table. A kernel has no
`set_cr(<n>, ...)` calls and no CR map in comments. The same table gives the
template its register names (`{{ cr.x }}` renders `CR2`), so the program and
the harness cannot drift apart.

## Background

- There are 512 `set_cr(` calls in kernel `app.py` files. The cleanest block is
  15 lines (`kernels/projections/app.py:213-233`).
- The register interface is written twice, as asm comments and as harness
  writes, and it has already drifted. `softmax/softmax_rows/softmax_rows.asm:32`
  documents `CR7 = ROW_BYTES (512)`, while `kernels/softmax/app.py:166` sets
  `set_cr(7, 1)`, in rows since issue #179.
- `MemoryApp.setup` writes the default dstructure
  (`kernel_registry/memory.py:74`, `set_cr_dstructure(valid_elements=LANES)`).
  CLAUDE.md: `CR15` conventionally holds the dstructure, but any `CR0`-`CR15`
  is a valid operand.
- Q49: a CR scalar used as an element operand (for example `MULT.VE ... CR5`)
  is decoded in the active format, from the low element-width bits. So `s = 2.0`
  must be written as FP32 bits in FP32, as the INT8 value 2 in INT8, and as the
  E4 code for 2.0 in E4.
- L16: kernels whose CRs are still in byte units are marked `unit = "bytes"`
  and stay frozen, rather than being converted in a refactor.

## Implementation

Module `ipu_harness/abi.py`:

```python
@dataclass(frozen=True)
class CrWrite:
    cr: int
    value: int            # raw register bits
    source: str           # "x (region base)", "rows (int param)", "s (element scalar, e4)"

def cr_writes(plan: CasePlan, placement: Placement) -> tuple[CrWrite, ...]: ...
def template_names(manifest: KernelManifest) -> Mapping[str, str]:   # {"x": "CR2", ...}
```

Value rules, by what the `[abi]` key names:

| Key names | Value written |
|-----------|---------------|
| a tensor | its base row (from `Placement`); for `unit = "bytes"` kernels, `base_row x row_bytes(format)` |
| an `int` / `bool` param | the integer |
| a `float` param | the E03 encoding of the value in `plan.format` (validated exactly representable in E06-03) |
| a `derived` value | see the open question below |

Also:
- If no `[abi]` key binds `CR15`, write the default dstructure
  (`valid_elements = 128`), as `MemoryApp` does today.
- After setup, check that every CR the rendered program reads is `CR0`, `CR1`,
  or a bound CR. A read of an unbound CR is a configuration error that names
  the instruction line. This replaces the "is the comment map still right"
  review step.
- CR writes go through the E04 reference-model API (registers by spec name),
  not `state.regfile` internals.

## Tests

- [ ] `scale` in FP32/INT8/E4: CR2, CR3 and CR4 are the same in every format; CR5 holds `0x40000000`, `0x02` and the E4 code of 2.0 respectively.
- [ ] Binding a tensor in a `unit = "bytes"` kernel writes `base_row x row_bytes`.
- [ ] A program that reads `CR9` with no `[abi]` binding fails with the instruction line.
- [ ] The projections `[abi]` table plus placement reproduces today's 13 CR values (`kernels/projections/app.py:221-233`) for all 12 kernels (guards E07-02). This needs the derived values below.

## Acceptance Criteria

- [ ] CRs are written only by `ipu_harness.abi`. Kernels contain no CR numbers outside `[abi]` and no CR comment maps.
- [ ] `bazel test //...` is green.

## Open questions

- **Derived CR values.** Projections need CR values computed from parameters,
  for example `TAIL_BOUND = k - 128 x (ceil(k / 128) - 1) - 2`, `CHUNK_COUNT`,
  `LAST_CHUNK_IDX`, `-1` and `126` (`kernels/projections/app.py:225-232`).
  The ISA has no instruction that loads an arbitrary immediate into an LR or a
  CR, so the template cannot compute them alone. Options:
  (a) a `[derived]` table of expressions, using the same whitelist grammar as
  the shape expressions in E06-03, for example `tail_bound = "k - 128 * (ceil(k / 128) - 1) - 2"`;
  (b) a `"module:fn"` hook returning a dict;
  (c) both, with (a) preferred.
  The drafts assume (c).
- **LR presets.** Today 27 harnesses preset 234 LRs before the run. Q80 defines
  `[abi]` as name -> CR. Should `[abi]` also accept `LR0`-`LR15` targets, or
  must those kernels initialise LRs in the program from CRs?
