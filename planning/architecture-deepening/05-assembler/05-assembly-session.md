# `AssemblySession` and `SymbolTable`: no global label table, one render

Part of [Epic: Assembler as a deep module](00-epic.md) · ID E05-05 · Size M · Depends on: E05-04 · Behaviour change: none

## Goal

One object runs one assembly from source text to words: render (once) → parse → resolve symbols → encode. It owns its own symbol table, so two assemblies in one process never interfere and nobody calls `reset_labels()`. `lark_tree` and (in E05-07) `diagnostics` become thin layers over this pipeline instead of two copies of it.

This is the module's main seam. Its interface is small (`AssemblySession(source, defines=…).assemble() -> AssemblyResult`), and it hides render, parse, symbol resolution and encoding.

## Background

- The global table is `label.ipu_labels`, with `reset_labels()` at `label.py:28-34`.
  - Reset in `lark_tree.parse` (`lark_tree.py:169`) and `diagnostics.py:215, 508`.
  - Written by `ASTBuilder.compound_instruction` (`lark_tree.py:131`).
  - Read by `LabelToken.__init__` and `encode` (`ipu_token.py:175, 195`).
- Seven test files call `reset_labels` directly:
  - `ipu-emu-py/test/test_emulator_e2e.py`;
  - `ipu-apps/test/mobilevit/test_full_layer_l{4,5}_packed.py`;
  - `ipu-apps/test/mobilevit/test_packed_layernorm_{192x64,240x16}.py`;
  - `ipu-apps/test/mobilevit/test_packed_residual_add_{192x64,240x16}.py`.
- Double render: `lark_tree.parse` renders when markers are present (`lark_tree.py:174-175`), and `parse_tree` renders again if markers remain (`lark_tree.py:36-37`). A rendered program that legitimately contains `{{` is rendered twice.
- In-process assemblers: `kernel_registry/cases.py:65` (`assemble_to_bin_file`), four convolution `app.py` files (render-per-run kernels), and many tests.

## Implementation

```python
# ipu_as/session.py
class SymbolTable:
    def define(self, name: AnnotatedToken, address: int) -> None   # LabelError on a duplicate
    def address(self, ref: AnnotatedToken, *, at: int) -> int       # LabelError if undefined; "+N" relative stays in LabelToken
    def __contains__(self, name: str) -> bool
    def items(self) -> Iterable[tuple[str, int]]

@dataclass(frozen=True)
class AssemblyResult:
    words: tuple[int, ...]
    symbols: Mapping[str, int]
    rendered: str
    warnings: tuple[str, ...]        # simulation-only slot warnings, today's warnings.warn text
    def to_mem(self) -> str          # "0x0{word:08x}\n" per word, as assemble_to_mem_file writes today
    def to_bin(self) -> bytes        # little-endian, aligned word length, as assemble_to_bin_file writes today

class AssemblySession:
    def __init__(self, source: str, *, defines: Mapping[str, object] | None = None, name: str = "<input>"): ...
    def render(self) -> str                      # exactly once; TemplateError
    def parse(self) -> list[ParsedCompound]      # ParseError; defines symbols
    def assemble(self) -> AssemblyResult         # EncodeError / LabelError

def assemble(source: str, *, defines=None, name="<input>") -> AssemblyResult   # convenience
```

1. Pass a `SymbolTable` into `ASTBuilder` and into the operand-conversion step from E05-02. `LabelToken` no longer reads a global.
2. `parse_tree(text)` stops rendering, which removes the double render. Rendering happens only in `AssemblySession.render`.
3. `lark_tree.parse`, `lark_tree.assemble`, `assemble_to_mem_file` and `assemble_to_bin_file` become wrappers over the session, keeping today's signatures. The `print` + `exit(1)` stays in the wrappers for now; E05-06 removes it.
4. Keep `label.reset_labels()` as a no-op with a `DeprecationWarning` for one PR. Migrate the seven test files to `AssemblySession` in this PR, then delete `label.ipu_labels` and `reset_labels()`.
5. `defines` is accepted and forwarded to `template.render`. Until E05-09 lands, a non-empty `defines` raises `TemplateError("defines not supported yet")`, so nobody depends on half a feature.
6. Add an in-process cache keyed by `(sha256(source), frozen defines, target)` for repeated assemblies of the same kernel in one process. Kernel test suites assemble the same source many times.

## Tests

- [ ] Two sessions with the same label names, created and interleaved in one process, both assemble correctly with no reset.
- [ ] A rendered program containing a literal `{{ '{{' }}` is rendered once. The parser sees `{{`, which is a parse error, rather than an empty string.
- [ ] `AssemblyResult.to_bin()` and `to_mem()` equal the files `assemble_to_bin_file` and `assemble_to_mem_file` write, for every E02 source.
- [ ] The cache returns the same result object for a repeated `(source, defines)` and a new one when `defines` changes.
- [ ] E02 digests unchanged; `bazel test //...` green.

## Acceptance Criteria

- [ ] `grep -rn "ipu_labels\|reset_labels" src` is empty.
- [ ] Rendering happens in exactly one function on the assembly path.
- [ ] `lark_tree` holds no pipeline logic of its own; it only adapts file I/O to the session.

## Risks

- Hidden reliance on label state leaking between calls (for example a test that assembles two fragments and expects the second to see the first's labels). The seven migrated files are the known users. Run the full suite before deleting the shim.
