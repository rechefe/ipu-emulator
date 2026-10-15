# Convolutions (4/4): `conv_universal_wide384` and `depthwise_conv_stride2_*` on params and stages

Part of [Epic: Kernel migration](00-epic.md) · ID E07-19 · Size M · Depends on: E07-18, E06-14 (stages hook), E06-07 (case params as template values) · Behaviour change: none

## Goal

Migrate the 4 kernels that override `run` today:
- `conv_universal_wide384` renders its program per run;
- `depthwise_conv_stride2_{narrow,128,16}` run two programs.

After this issue no kernel assembles in a temp dir, swaps `inst_path` or calls
`run_test` directly.

## Background

Paths are under `src/tools/ipu-apps/src/ipu_apps/kernels/convolutions/`.

- **`conv_universal_wide384`** (387 lines of `app.py`):
  - `run` renders with `render_asm(..., cpr=self.cpr, rows=self.rows)`
    (`app.py:266-267`), assembles into a temp dir and swaps
    `self.inst_path` (`app.py:329-338`);
  - the template hides missing parameters with
    `cpr | default(3)` and `rows | default(8)` (`conv_universal_wide384.asm:128-129`);
  - its teardown transposes the output (`app.py:316-327`).
- **`depthwise_conv_stride2_{narrow,128,16}`** (266-313 lines of `app.py`):
  - stage 1 is the `depthwise_conv_universal` program;
  - stage 2 is their own program, rendered with `elements_in_row`
    (`depthwise_conv_stride2_narrow.asm:3`, `| default(32)`);
  - in between: a PC reset, a CR15 dstructure rewrite and 4 CR writes by
    number (`depthwise_conv_stride2_narrow/app.py:202-252`);
  - stage 2 gets no debug callback.
- All 4 have `benchmark.py` files.

## Implementation

- **`conv_universal_wide384`** is a folder kernel. `cpr` and `rows` are
  `[params]`, and the template uses `{{ cpr }}` and `{{ rows }}`. Remove
  `| default`: E05's strict render rejects it, and the default case supplies
  today's defaults (3, 8). Its output declaration uses an output `pack`
  transform that reproduces today's channel-major file layout. The digests are
  checked per distinct `(cpr, rows)` the cases use.
- **`depthwise_conv_stride2_*`** are folder kernels with `stages = "stages.py:run"`
  (E06-14), each about 10 lines:
  - stage 1 runs `../depthwise_conv_universal/...` (or the E07-18 template
    entry);
  - stage 2 runs the kernel's own program with `elements_in_row` from
    `[params]` and `dstructure = Dstructure(valid_elements=128, partition=P0)`;
  - the intermediate region is `[scratch.stage1_out]`.

  If the 3 stage-2 programs differ only in `elements_in_row`, make one template
  with 3 `registry.toml` entries (check the digests).
- References: `ops.conv2d` (stride 2, depthwise).
- The benchmarks move. Apply the per-family checklist.

## Tests

- [ ] The digests for every distinct template-value set used by the cases equal E02. The E02 recording covers render-per-run kernels by their case values.
- [ ] `compare_legacy` is identical, including the cycles summed over both stages.
- [ ] `--debug` stops in stage 2 (with a fake callback).

## Acceptance Criteria

- [ ] No `run` override, `render_asm`, temp-dir assembly or `| default` remains in the convolution family. `kernels/convolutions/app.py` is deleted.
- [ ] `bazel test //...` is green.
