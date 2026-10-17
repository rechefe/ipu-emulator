# Family `registry.toml`: shape variants over a template

Part of [Epic: Kernel framework](00-epic.md) · ID E06-17 · Size M · Depends on: E06-13, E06-16 · Behaviour change: none

## Goal

A family declares its shape variants as rows of one `registry.toml`, each a name
plus parameter values over a **template kernel** (one `.asm` + `kernel.toml`).
This replaces per-shape folders (Q76, which replaces L12). Kernel names and
Bazel target names stay the same, so routing answers, benchmark history and
documentation links keep working.

## Background

- 61 of 89 kernels are named after a shape. Their folders are copies.
  - The 12 `projections/proj_*_{144,192,240}_p4` folders each hold an `app.py`
    of about 60 lines (docstring plus four numbers,
    `projections/proj_ffn1_192_p4/app.py:48-64`), a 3-line `cases.py`, a 3-line
    `test.py` and an `.asm` file.
  - With comments stripped, the 12 `.asm` bodies hash to 4 distinct programs:
    single token group identity x6, single token group SiLU x2, two token groups
    identity x3, and two token groups SiLU x1.
- Other copy groups: `residual_add` x4, `concat` x3, `fold`/`unfold` x3 each,
  `matmul_*_x128` x12, `layernorm_*` x4, and the attention shape triples.

## Implementation

`kernels/<family>/registry.toml`:

```toml
# kernels/projections/registry.toml
[[kernel]]
name = "proj_ffn1_192_p4"           # kernel name == Bazel target name (unchanged)
template = "proj_p4"                # kernels/projections/proj_p4/{proj_p4.asm, kernel.toml}
params = { k = 192, n_out = 384, n_tok = 64, n_tg = 1, activation = "silu" }
doc = "Layer 4 FFN1"                # one line, shown on the kernel page and in query output

[[kernel]]
name = "proj_qkv_144_p4"
template = "proj_p4_2tg"
params = { k = 144, n_out = 432, n_tok = 128, n_tg = 2, activation = "none" }
doc = "Layer 3 QKV"
```

- **Template.** A template is an ordinary kernel folder. Its `kernel.toml`
  declares the parameter domain and the cases, and it can be run and tested
  itself.
- **Entries.** An entry loads the template manifest with `overrides` (E06-02):
  the entry's `params` become fixed values. Fixing a parameter removes it from
  the query domain for that entry, which is today's "exact-match" `supports`.
  The template's cases run with those values. The entry inherits the template's
  formats unless it narrows them with `formats = [...]`.
- **Bazel.** `ipu_kernel_family()` in `kernels/<family>/BUILD.bazel` reads
  `registry.toml` at load time (how is an open question below). It declares
  `:<name>`, `:assemble_<name>` and `:<name>_nightly` for each entry, with the
  same behaviour as `ipu_kernel()`.
- **Registry.** `Registry.load` (E06-16) adds each entry as a kernel named
  `name`, with the template's `op`.
- **Names.** Entry names must be unique across the repo, and must not collide
  with a folder kernel. Both are checked at load.

## Tests

- [ ] A family with a template and 3 entries yields 3 kernels with the given names. Each one's domain is fixed to its `params`, and `resolve` routes each exact query to its entry.
- [ ] `bazel test //kernels/<family>:<entry>` runs the template's cases with the entry's values.
- [ ] A duplicate name, an unknown template, or a `params` key outside the template's `[params]` is a located error (`registry.toml:12`).
- [ ] The `assemble_<entry>` output equals the binary of the old per-shape folder, via the E02 digest (exercised by E07-02).

## Acceptance Criteria

- [ ] A new shape variant is a 4-line `registry.toml` entry, with no new folder.
- [ ] `bazel test //...` is green.

## Open questions

- **Bazel label for entries.** Folder kernels get `//kernels/<family>/<name>`
  (Q80). Entries live in the family package, so they get
  `//kernels/<family>:<name>`. Is that acceptable, or should each entry also get
  a generated one-line package so the label shape is uniform?
- **Reading TOML in Starlark.** Options are (a) a minimal Starlark reader for
  the `[[kernel]]` subset; (b) a checked-in `registry.bzl` generated from
  `registry.toml`, with a test that fails if it is stale; (c) a repository rule.
  The drafts assume (b), the simplest and most hermetic.
