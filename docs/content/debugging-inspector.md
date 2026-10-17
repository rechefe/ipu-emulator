# Cycle Inspector

The [Cycle Inspector](inspector/index.html) replays a recorded kernel run cycle
by cycle, in the browser, with nothing to install. Pick a kernel, then step
forward and back, run to a breakpoint, or drag the timeline. For each cycle it
shows:

- the program listing, with the word that just ran highlighted and how often
  each word ran;
- the VLIW word: each slot's instruction and the word's bit fields, MSB first;
- the register file after the cycle (LR, CR and the vector registers), with the
  cells the cycle changed outlined;
- the hardware pipeline: which blocks the word used, or each word's stage
  timing;
- the XMEM traffic and the multiplier lanes in use.

The **Assembly source** tab shows the kernel's `.asm` file as written, before
Jinja rendering. The **ISA reference** tab lists every instruction of each
slot; select one for its full description.

The page is static HTML and JavaScript. It makes no request outside the docs
site and uses system fonts.

## Where the traces come from

Every trace is a build output; none is checked in. The docs build records each
curated case with the reference model's trace writer and converts it to the
page's format, so the page always matches the current ISA and kernels:

1. `//src/tools/ipu-apps:trace_case` runs a registered kernel case exactly as
   its test runs it. It writes the case's trace (one commit record per retired
   word; the schemas are in `ipu_isa/schemas/`) and a context file: the
   kernel's source and labels, the starting registers, and per word the
   multiplier lanes used and the XMEM ranges read.
2. `ipu-gen inspector-trace` converts the two files into the page's JSON
   (`ipu_gen/schemas/inspector-trace.schema.json`). It keeps the first 5000
   cycles of a long run and marks the trace as truncated.
3. `//docs/inspector:inspector_data` gathers every trace, the kernel list
   (`kernels.json`, from `docs/inspector/curated.toml`) and `isa.json`, the
   instruction reference the editor extension also reads. The docs build copies
   them to `inspector/data/`.

The `kernel_trace` macro in `docs/inspector.bzl` does steps 1 and 2 for one
case. To record a trace by hand:

```bash
bazel run //src/tools/ipu-apps:trace_case -- identity --trace "$PWD/trace.jsonl" --context "$PWD/context.json"
bazel run //src/tools/ipu-gen:ipu-gen -- inspector-trace trace.jsonl context.json identity.json
```

## Add a kernel

1. Add an entry to `CURATED` in `docs/inspector/BUILD.bazel`, where the page's
   order is kept. Each entry becomes one `kernel_trace` target, and its name is
   the kernel's id on the page:

    ```python
    {"name": "softmax_rows", "kernel": "softmax_rows", "options": {"rows": 2}},
    ```

    `case` picks one of the kernel's cases (`default` unless given), and
    `options` overrides its options. A kernel's element formats are its cases,
    such as `fully_connected`'s `int8`, `fp8_e4m3` and `fp8_e5m2`.

2. Add a `[[kernel]]` entry with the same `id`, at the same place, to
   `docs/inspector/curated.toml`, with its `group` (`native` or `wide`),
   `title`, `label` and `description`.

`bazel test //docs/inspector:test_kernels` fails when the two lists disagree.
`bazel build --config=docs //docs:build_docs` then rebuilds the site, and
`bazel test //docs:test_inspector_site` checks that the page and every trace are
in it.
