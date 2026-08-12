# ipu-ctrl-rdl

Generates Python register constants, a Rust `no_std` PAC crate, and HTML
register-map documentation from `src/hw/ipu_ctrl.rdl`.

Codegen uses a custom exporter over the `systemrdl-compiler` IR (see
`docs/content/specs/riscv-host-integration.md` §6) so Python and Rust stay
lock-step without a fragile RDL→SVD→`svd2rust` pipeline.

Invoke via Bazel (`//src/tools/ipu-ctrl-rdl:codegen`) — do not run standalone
scripts outside the build graph.
