# Reshape (1/2): `identity`, `depth_to_space`, `concat` x3

Part of [Epic: Kernel migration](00-epic.md) · ID E07-04 · Size M · Depends on: E07-02 · Behaviour change: none

## Goal

Migrate the first 5 reshape kernels. Two of them, `identity` and
`depth_to_space`, are whole-image (MemoryApp) kernels and become ordinary
declared-tensor kernels. The three `concat_*` copies become one template plus 3
`registry.toml` entries. Binaries and FP32 outputs stay identical.

## Background

Paths are under `src/tools/ipu-apps/src/ipu_apps/kernels/reshape/`.

- **`identity`** (`identity/app.py`, 40 lines) is the smallest MemoryApp
  kernel. Its CR map `{2: 0, 3: rows, 4: rows}` is in `memory_layout`
  (`identity/app.py:27`). Its check is byte-exact (`identity/cases.py`,
  `check_output_bytes`). Its program uses lower-case `lr`/`cr` template
  variables (`identity/identity.asm:17-31`).
- **`depth_to_space`** (`depth_to_space/app.py`, 27 lines) is MemoryApp. Its
  `cases.py` builds the complete XMEM image. Its header documents some CRs as
  byte addresses. Confirm against the case builder, and if confirmed, apply
  `unit = "bytes"` (L16).
- **`concat_{8x8x160x160,16x16x128x128,32x32x96x96}`** have identical bodies
  once comments are stripped. Each has a 114-line `app.py` with 9 `set_cr`.
- The family `reshape/app.py` (199 lines), `cases.py` (160) and `test.py` (90)
  hold the shared query and layouts.

## Implementation

- `identity`:
  - `kernel.toml` with `x` and `y` of shape `["rows", 128]`, `[abi]`
    `x = "CR2"`, `y = "CR3"`, `rows = "CR4"`.
  - `op = "identity"` reference (`y = x`), tier `EXACT`.
  - `formats = ["fp32"]` for now. `identity` is a natural candidate for the
    first "add formats" follow-up after E07-03.
  - It is the second worked example in the how-to (E06-21), next to `scale`.
- `depth_to_space`: declare the input as the logical `[c, h, w]` tensor. The
  layout that `cases.py` built into the image becomes a family `pack.py`
  transform. Scratch and padding rows become `[scratch.*]`.
- `concat`: one template `kernels/reshape/concat/` with the shape parameters,
  plus 3 `registry.toml` entries named as today.
- Apply the per-family checklist (epic): upper-case mnemonics, design notes into
  the `.asm` header, exact input sizes, and family tests moved into `test_*.py`.

## Tests

- [ ] `compare_setup` and `compare_legacy reshape` (these 5 kernels): binaries, FP32 raw outputs and cycles are identical.
- [ ] `identity` passes byte-exact against the reference in its cases.
- [ ] `//kernels:query` routing answers for `identity`, `depth_to_space` and `concat` match the old `:query`.

## Acceptance Criteria

- [ ] No MemoryApp users remain in reshape. The 5 kernels have no `app.py`, `cases.py`, `test.py` or `__init__.py`.
- [ ] `bazel test //...` is green.
