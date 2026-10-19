# Stages hook for multi-stage kernels

Part of [Epic: Kernel framework](00-epic.md) · ID E06-14 · Size S · Depends on: E06-11 · Behaviour change: none

## Goal

Some kernels run more than one program on the same machine state, where stage 2
reads what stage 1 left in XMEM. The stages hook lets such a kernel describe its
stages in a few lines of Python. Placement, packing, CR writes, debugging,
profiling and comparison stay with the framework. This is the third escape
hatch named in Q80.

## Background

The 3 `depthwise_conv_stride2_{narrow,128,16}` kernels each override `run`.
For example, `convolutions/depthwise_conv_stride2_narrow/app.py:202-252`:
- It assembles a sibling kernel's program (`depthwise_conv_universal`) as
  stage 1, renders its own `.asm` with `render_asm(elements_in_row=...)` as
  stage 2, and assembles both into a temp dir.
- It runs stage 1 through a nested harness, then calls `run_test` directly for
  stage 2. In between it resets `program_counter = 0`, rewrites CR15
  (`set_cr_dstructure(valid_elements=128, partition=P0)`) and sets 4 CRs by
  number.
- Stage 2 gets no debug callback, and the `IPU_DEBUG_TUI` launch covers only
  stage 1.

The family data glob exists so that this can happen: `ipu_app.bzl:103-107`
makes every family `.asm` data of every kernel.

## Implementation

`kernel.toml`:

```toml
[kernel]
stages = "stages.py:run"      # optional; absent = one stage, the kernel's own .asm
```

Module `ipu_harness/stages.py`:

```python
class StageContext(Protocol):
    plan: CasePlan
    placement: Placement
    def run_stage(self, program: str, *, params: Mapping[str, object] = {},
                  abi: Mapping[str, str] = {}, dstructure: Dstructure | None = None,
                  max_cycles: int | None = None) -> int:
        """Assemble `program` (this kernel's asm, or "<family-relative kernel>/<name>.asm")
        with `params` as template values; reset PC; write `abi` CRs from plan/placement
        names; set CR15 if given; run to HALT on the SAME machine state; return cycles."""

def run(ctx: StageContext) -> None: ...   # what a kernel's stages.py:run implements
```

A stage-2 hook for `depthwise_conv_stride2_narrow` becomes about 10 lines:
`ctx.run_stage("../depthwise_conv_universal/depthwise_conv_universal.asm", abi={...})`,
then `ctx.run_stage("depthwise_conv_stride2_narrow.asm", params={"elements_in_row": ctx.plan.params["cols"]}, abi={...}, dstructure=Dstructure(valid_elements=128))`.

Rules:
- Intermediate regions are declared as `[scratch.*]` in `kernel.toml`, so
  placement (E06-04) owns their rows. The hook never computes an address.
- Every stage receives the debug callback and the alias profile (E06-12).
  Cycles are summed into `RunResult.cycles`, and per-stage cycles are reported.
- A stage may reference another kernel's `.asm` only within its family, or a
  family listed in `FAMILY_DEPS` (E06-18). That reference is how Bazel data is
  declared, so the whole-family glob is not needed.

## Tests

- [ ] A two-stage toy kernel (stage 1 copies `x` to scratch, stage 2 scales scratch into `y`) passes with the reference `y = x * s`.
- [ ] Both stages stop at entry under a fake debug callback.
- [ ] `RunResult.cycles` equals the sum of the stage cycles, and the report lists both stages.
- [ ] Referencing an `.asm` from an undeclared family fails at load.

## Acceptance Criteria

- [ ] Multi-stage kernels need no `run` override, no temp-dir assembly and no CR numbers.
- [ ] `bazel test //...` is green.
