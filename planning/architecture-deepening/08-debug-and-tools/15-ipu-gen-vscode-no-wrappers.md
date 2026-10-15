# Move `gen_vscode` into `ipu_gen`; delete every `*_wrapper.py` shim

Part of [Epic: Debugger, generators and editor](00-epic.md) · ID E08-15 · Size S · Depends on: E08-14 · Behaviour change: none (`isa.json`, TextMate grammar and language configuration byte-identical)

## Goal

The VS Code language-data generator joins the other generators in `ipu_gen`. Every 3-line `*_wrapper.py` shim is replaced by a real entry point in its package (Q69).

## Background

- `ipu_as/gen_vscode.py` (532 lines) writes the TextMate grammar, `language-configuration.json` and `data/isa.json`. It runs through `vscode-ipu-asm/gen_vscode_wrapper.py` (`vscode-ipu-asm/BUILD.bazel:37-44, 68-77`).
- `gen_vscode_wrapper.py:16, 23` imports `ipu_emu.ipu.Ipu` to call the private `Ipu._lrd_lr_indices`. A generator depends on the emulator through a private name.
- Wrapper shims:
  - `docs/gen_docs_wrapper.py`
  - `docs/gen_codegen_wrapper.py`
  - `docs/gen_instruction_format_md_wrapper.py`
  - `docs/mkdocs_wrapper.py`
  - `docs/serve_docs_wrapper.py`
  - `vscode-ipu-asm/gen_vscode_wrapper.py`
  - `vscode-ipu-asm/dump_parser_tokens_wrapper.py`

  E08-13 and E08-14 already retire the first three.
- `.github/workflows/vscode-extension.yml:82, 88` runs `bazel run //vscode-ipu-asm:gen_vscode` and `//vscode-ipu-asm:dump_parser_tokens`.

## Implementation

1. `git mv` `ipu_as/gen_vscode.py` → `ipu_gen/vscode_data.py`, and add `ipu-gen vscode-data OUT_DIR`.
2. Read the LRD pair indices from `ipu_isa`, where the LRD register definition belongs. If E03 does not provide them, add a public accessor in `ipu_emu` and use that; no private access (Q66).
3. Keep the Bazel labels the workflow uses: `//vscode-ipu-asm:gen_vscode` becomes an `alias` (or a `py_binary` with `main_module`) to `//src/tools/ipu-gen:ipu-gen` with the `vscode-data` argument, and `generate_grammar` is unchanged.
4. `dump_parser_tokens` stays in `ipu_as`; it is parser introspection, and E08-22 replaces its Jinja part. Point its Bazel target at the module with `main_module` (rules_python 0.40 supports it) and delete the wrapper.
5. `mkdocs_wrapper.py` only re-exports `mkdocs.__main__.cli`. Replace it with `py_binary(main_module = "mkdocs")` (rules_python 0.40.0 is pinned in `MODULE.bazel:7`). `serve_docs_wrapper.py` is a real script: it extracts `site.tar` and runs `http.server`. Rename it to `docs/serve_docs.py` without changing its behaviour.

## Tests

- [ ] `bazel run //vscode-ipu-asm:gen_vscode` output is byte-identical to the base (the E02 `isa.json` snapshot plus a diff of the grammar files).
- [ ] `ipu-as-py:test_hover_data` and `:test_isa_data` pass, moved to `ipu-gen` if they test the generator.
- [ ] `find . -name "*_wrapper.py" -not -path "./bazel-*"` is empty.
- [ ] `bazel test //...` green. The vscode-extension workflow passes on the PR (its path triggers include `src/tools/ipu-gen/**` per E01).

## Acceptance Criteria

- [ ] `ipu_gen` imports no private name from any package.
- [ ] No wrapper shim remains.
