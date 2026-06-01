# IPU Host Register Map

The authoritative register definitions live in
[`src/hw/ipu_ctrl.rdl`](https://github.com/rechefe/ipu-emulator/blob/master/src/hw/ipu_ctrl.rdl).
Python offsets/masks, the Rust `ipu-pac` crate, and this HTML map are all generated from
that SystemRDL source (see
[`src/tools/ipu-ctrl-rdl/README.md`](https://github.com/rechefe/ipu-emulator/blob/master/src/tools/ipu-ctrl-rdl/README.md)).

!!! note "Generated documentation"
    The interactive register map is produced by **peakrdl-html** at build time.
    After `bazel build //docs:build_docs`, open
    `bazel-bin/docs/site_output/specs/ipu-ctrl-regmap/index.html`.

For architecture and execution semantics, see
[RISC-V Host Integration](riscv-host-integration.md) and
[Control Stage Specification](stage-control.md).
