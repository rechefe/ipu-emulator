# Epic: Assembler as a deep module — slot codec, assembly session, located errors, strict templates

ID E05 · Folder `05-assembler` · Runs after E03 (ISA package, binary codec), alongside E04.

## Goal

Make `ipu_as` a small set of deep modules with explicit seams:

- **Encoding** uses the binary codec from `ipu_isa` (E03). The nine copy-paste `Inst` subclasses go, and no spec table is rebuilt per instruction.
- **One assembly pipeline** (`AssemblySession`): render → parse → resolve symbols → encode. It owns its symbol table (no globals), renders once, and raises typed, located errors. The library never prints and never exits; only the CLI does.
- **One template render path** (`ipu_as.template.render`): declared parameters, `StrictUndefined`, a sandboxed macro loader, `cr`/`lr` ABI namespaces. The CLI, Bazel `assemble_asm(defines=)` and the kernel harness (E06) all call it.
- **Language additions**: GNU numeric local labels, `--target hardware`, the new mnemonic standard.

The assembler encode stage gets at least 3x faster through cached tables. Binaries stay byte-identical throughout, except where an issue is labelled as a behaviour change.

## Background

Verified at HEAD `694a8a8`.

- `inst.py:365-686` defines nine `Inst` subclasses (`LoadInst` … `BreakInst`). They differ only in slot name and doc text. Each `struct_by_opcode_table()` calls `_build_struct_table(slot)` (`inst.py:54`) on every call, and `find_inst_type_by_opcode` (`inst.py:182-187`) scans every subclass per instruction. `EnumToken._reverse_map` (`ipu_token.py:126`) is rebuilt on every `encode`.
- Decoded-word field names are derived from Python class names (`compound_inst.py:145-178`, `camel_case_to_snake_case(inst_type.__name__)`). The emulator rebuilds the same strings independently (`ipu.py` `_TYPE_FIELD_SUFFIX` / `_SLOT_FIELD_PREFIX`). Renaming a class silently changes the wire format.
- The label table is a module global (`label.py:28-34`). It is reset in `lark_tree.parse` (`lark_tree.py:169`), in `diagnostics.check` (`diagnostics.py:215, 508`) and directly by 7 test files.
- `lark_tree.parse` and `lark_tree.assemble` print and call `exit(1)` (`lark_tree.py:183, 193`). In-process callers (`kernel_registry/cases.py:65`, four convolution `app.py` files) get a `SystemExit`.
- Jinja is rendered twice when markers remain (`lark_tree.py:174-175`, then `parse_tree` at `:36-37`). `template.render(text)` (`template.py:44-49`) takes no context. Missing names render as empty strings. There is no `{% include %}`. The convolution family has a second sandbox (`kernels/convolutions/app.py:128-137`), and 6 mobilevit tests use unsandboxed `jinja2.Template`.
- `diagnostics.check` (`diagnostics.py:468-532`) re-implements the pipeline. It recovers positions by regex-matching message text (`_LOCATION_RE`, `diagnostics.py:36`).
- `cli.py:16-20, 36-40` uses `prompt=` for `--format`, so non-interactive use without the flag fails. There is no `--define`, although SKILL.md documents one.
- The VS Code extension depends on `check --json`. `checker.js:130` spawns it, `extension.js:291` shows `stage`, and `editing.js:179` keys on `stage` plus message text.

## Decisions applied

- **Q69**: The binary codec lives in `ipu_isa` (E03 provides it). The assembler uses it, and the nine `Inst` subclasses are dropped.
- **Q28 / Q59**: `AssemblerError` carries a location and sits in the `IpuError` hierarchy. It is a setup-time error, never a hardware fault. The library never prints or exits; only the CLI does.
- **L7**: GNU numeric local labels (`1:`, `1f`, `1b`).
- **L8**: Keep `--input` / `--output`, drop the `--format` prompt, add `--define`.
- **L9**: Field names come from an explicit table, not class names.
- **L10**: Strict Jinja: declared params, `StrictUndefined`, a sandboxed macro loader, and one render path used by Bazel `assemble_asm(defines=)`, the CLI and the harness.
- **Q42**: `--target hardware` rejects EMULATOR-tagged instructions and emits hardware-only words (the top bits of the emulator word). The default target is the emulator.
- **Q45**: New mnemonic standard. The rename itself is E03; the assembler only has to accept the new names.
- **Q80**: `{{ cr.<name> }}` ABI names. E06 owns `kernel.toml [abi]`; the assembler accepts an ABI mapping.
- **Q71**: Assembler speed target of at least 3x, through cached tables. Numbers are tracked, not gated (Q64).
- **D3**: Refactor PRs are output-identical, and behaviour changes get their own labelled PRs.
- **L40**: The `check --json` contract stays stable and gets an integer `schemaVersion`.

## Scope

- `src/tools/ipu-as-py/src/ipu_as/**` (moving to `src/tools/ipu-as/` under the E03 package split)
- `asm_rules.bzl` (`assemble_asm(defines=, includes=)`)
- Migrating existing render callers: convolution `render_asm`, the mobilevit tests' `jinja2.Template`, `kernel_registry/cases.py`
- The JSON contract snapshot for `check --json`

## Out of scope

- The binary codec, `DecodedWord` and slot tags: E03.
- The mnemonic rename script and the SV rename table: E03.
- `kernel.toml`, region placement and the runner's error handling: E06.
- Emulator error classes and faults: E04.
- Generators (`gen_docs`, `gen_codegen`, `gen_vscode`) moving to `ipu_gen`: E08.
- Language services for VS Code (`check --json` v2 tokens, words and Jinja scopes) and `ipu-as format`: E08. E05-08 provides only the versioned contract they extend.
- Template collapse of kernel families: E07.

## Issue index

| ID | File | Title | Size | Depends on | Behaviour change? |
|----|------|-------|------|-----------|-------------------|
| E05-01 | [01-cache-encode-tables.md](01-cache-encode-tables.md) | Cache per-slot encode tables and enum reverse maps | S | E02 (asm digests), E01 (benchmarks) | none |
| E05-02 | [02-use-isa-slot-codec.md](02-use-isa-slot-codec.md) | Encode through the `ipu_isa` slot codec; delete the nine `Inst` subclasses | M | E05-01, E03 (binary codec) | none |
| E05-03 | [03-explicit-field-names.md](03-explicit-field-names.md) | Decoded-word field names from an explicit table | S | E05-02, E03 (binary codec) | none |
| E05-04 | [04-located-assembler-errors.md](04-located-assembler-errors.md) | `AssemblerError` hierarchy with `SourceLocation` | S | E03 (`IpuError` root) | none |
| E05-05 | [05-assembly-session.md](05-assembly-session.md) | `AssemblySession` and `SymbolTable`: no global label table, one render | M | E05-04 | none |
| E05-06 | [06-library-never-exits.md](06-library-never-exits.md) | Library raises `AssemblerError`; only the CLI prints and exits | S | E05-05 | yes (in-process callers get an exception instead of `SystemExit`) |
| E05-07 | [07-diagnostics-on-session.md](07-diagnostics-on-session.md) | `diagnostics.check` runs on the session and reads error locations | M | E05-05, E05-08 | none |
| E05-08 | [08-versioned-check-json.md](08-versioned-check-json.md) | Pin and version the `check --json` contract | S | E02 | none (v1 output byte-identical) |
| E05-09 | [09-template-render-api.md](09-template-render-api.md) | Template render API: declared params, defines, sandboxed macro loader | M | E05-05 | none |
| E05-10 | [10-cli-define-no-prompt.md](10-cli-define-no-prompt.md) | CLI: drop the `--format` prompt, add `--define` and `-I` | S | E05-06, E05-09 | yes (`--format` inferred instead of prompted) |
| E05-11 | [11-strict-undefined.md](11-strict-undefined.md) | Switch templates to `StrictUndefined` | S | E05-09, E05-12 | yes (an undefined template name is an error) |
| E05-12 | [12-one-render-path.md](12-one-render-path.md) | One render path: Bazel `assemble_asm(defines=)`, registry and tests | M | E05-09 | none (digest-proven) |
| E05-13 | [13-abi-namespaces.md](13-abi-namespaces.md) | `cr` / `lr` ABI namespaces in the render context | S | E05-09 | none (additive) |
| E05-14 | [14-local-labels.md](14-local-labels.md) | GNU numeric local labels (`1:`, `1f`, `1b`) | S | E05-05 | yes (new syntax, additive) |
| E05-15 | [15-target-hardware.md](15-target-hardware.md) | `--target hardware`: reject EMULATOR instructions, emit hardware words | M | E05-02, E03 (tags, slot order) | yes (new option; default unchanged) |
| E05-16 | [16-mnemonic-standard.md](16-mnemonic-standard.md) | Assembler readiness for the new mnemonic standard | S | E05-02, E03 (rename) | none (binaries unchanged) |

## Suggested order

```
E02 ──▶ 01 ──▶ 02 ──▶ 03
          │      ├──▶ 15
          │      └──▶ 16
E03 ──▶ 04 ──▶ 05 ──▶ 06 ──▶ 10
                │
                ├──▶ 09 ──▶ 12 ──▶ 11
                │     ├──▶ 13
                │     └──▶ 07
                └──▶ 14
E02 ──▶ 08 (any time; before 07 and E08's language-service work)
```

01-03 (encoding) and 04-07 (pipeline) are independent tracks. 09-13 (templates) need the session. 14-16 are small language additions.

## Epic acceptance

- [ ] No `Inst` subclass per slot. No spec table is rebuilt per instruction. The encode stage over all `.asm` sources is at least 3x faster than at the base commit (tracked number in the PR).
- [ ] No global label state, one render per assembly, and no `print` or `exit` in library code.
- [ ] Every user-facing assembler failure is an `AssemblerError` subclass with a location. `_LOCATION_RE` is gone.
- [ ] Exactly one Jinja render path: `ipu_as.template.render`.
- [ ] `check --json` v1 output is byte-identical to the base commit for the extension's corpus, and `schemaVersion` exists.
- [ ] All `.asm` digests (E02) are unchanged by every refactor issue, and `bazel test //...` is green after each.
