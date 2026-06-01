# IPU host-control register block (SystemRDL)

`src/hw/ipu_ctrl.rdl` is the single source of truth for the IPU host MMIO map
(control registers + fully mapped instruction memory). Bazel targets in this
package generate:

| Artifact | Consumer | Tooling |
|----------|----------|---------|
| `ipu_ctrl_regs.py` | Python emulator MMIO model (issue #2) | Custom IR exporter (`export_codegen.py`) |
| `ipu-pac` Rust crate | `no_std` firmware (issue #5) | Same IR exporter (Jinja → Rust) |
| HTML register map | Documentation site | `peakrdl-html` |

## Codegen path (issue #1)

We evaluated **peakrdl-python** (full simulation-oriented register model) and
**RDL → SVD → svd2rust** (standard Rust PAC path). Both add heavyweight or
fragile intermediate layers for what we need: **offsets, bit masks, and typed
enum constants** shared exactly between Python and Rust.

The chosen approach is a **small custom exporter** over the
[`systemrdl-compiler`](https://systemrdl-compiler.readthedocs.io/) IR, with
Jinja2 templates emitting Python and `no_std` Rust from the same walk. HTML
documentation still uses **peakrdl-html** (the supported path for register-map
docs). This keeps Python and Rust artifacts in lock-step while remaining easy
to extend when issue #2 adds bridge semantics.

`INSTRUCTION_ALIGNED_BYTES` is passed into elaboration from the assembler
(`instruction_aligned_bytes_len()`) so `IMEM_MAP_SIZE` matches the binary format.

## Build

```bash
bazel build //src/tools/ipu-ctrl-rdl:all
bazel test //src/tools/ipu-ctrl-rdl:test_codegen
```

When adding or changing Python dependencies in `pyproject.toml`, regenerate the
repo lockfile (used by Bazel, no `uv` required at build time):

```bash
./tools/update_requirements.sh
```
