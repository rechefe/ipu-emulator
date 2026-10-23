# `DebugSession` core: public parse, resolve and read API

Part of [Epic: Debugger, generators and editor](00-epic.md) · ID E08-04 · Size M · Depends on: E08-03 · Behaviour change: none

## Goal

Create the deep module both frontends sit on. `ipu_debug.session.DebugSession` has no curses and no I/O: it parses user input, resolves register names, reads registers and XMEM windows, and disassembles, returning values. The TUI stops using the CLI module as a library through private names.

## Background

- `ipu_debug/tui.py` (formerly `debug_tui.py:27-40`) imports 11 private names from the CLI module: `_CELL16_BACKGROUND_SHIFT`, `_CELL16_CHARACTER_MASK`, `_F32_DISPLAY_WIDTH`, `_HEX_BYTES_PER_GROUP` and others. It also imports `_resolve_xmem_request`, `_tokenize_xmem`, `_parse_xmem_request` and `_parse_int` inside functions (old lines 568, 1334, 1344, 3099, 3159, 3201, 3322).
- Values are formatted for display inside the CLI (`print_all_registers`, `disassemble_at` at old `debug_cli.py:361`).
- The TUI reads raw register bytes with `RegFile.raw()` (old `debug_tui.py:621, 3124, 3173`).

## Implementation

```python
# ipu_debug/session.py  (no curses, no stream writes)
class DebugError(IpuError, ValueError): ...        # every user-facing failure; message is the text shown

@dataclass(frozen=True)
class RegisterView: name: str; index: int | None; dtype: RegDtype; value: int | bytes; width: int
@dataclass(frozen=True)
class XmemRequest: mode: Literal["row", "byte"]; base: int; offset: int; count: int; fmt: XmemFormat
@dataclass(frozen=True)
class XmemWindow: request: XmemRequest; cells: tuple[Cell, ...]   # decoded value + raw bytes
@dataclass(frozen=True)
class DisasmLine: pc: int; text: str; is_current: bool; has_breakpoint: bool

class DebugSession:
    def __init__(self, state: IpuState, control: RunControl, *, kernel_name: str | None = None): ...
    @staticmethod
    def parse_int(text: str) -> int                                  # DebugError
    def parse_xmem(self, text: str) -> XmemRequest                   # DebugError
    def resolve_register(self, name: str) -> tuple[RegDescriptor, int | None]
    def registers(self, names: Iterable[str] | None = None) -> list[RegisterView]
    def read_xmem(self, request: XmemRequest) -> XmemWindow
    def disassemble(self, start: int, count: int) -> list[DisasmLine]
```

1. Move the parse and resolve helpers out of `cli.py` into `session.py` as public functions or methods, unchanged in behaviour.
2. Move the display constants and `decode_16bit_cell` into `ipu_debug/format.py` (public names). Both frontends import from there.
3. `cli.py` and `tui.py` call the session for parsing, reading and disassembly. Output formatting stays in each adapter.
4. Replace the TUI's `RegFile.raw()` calls with `registers()` / `read_xmem()`. If a view needs raw bytes, `RegisterView.value` is `bytes` for vector registers.
5. Add a lint test that fails on `from ipu_debug.cli import _` and `from ipu_debug.tui import _` anywhere: no private cross-module access (Q66).

## Tests

- [ ] New `test_session.py` exercises parse, resolve, `registers`, `read_xmem` and `disassemble` on a real `IpuState` with an assembled program. It uses no curses and no monkeypatch.
- [ ] `test_debug_cli` and `test_debug_tui` pass. CLI output text is unchanged; snapshot the CLI transcript of `test_debug_cli`'s command sequences before and after.
- [ ] The lint test fails on a planted private import.
- [ ] `bazel test //...` green.

## Acceptance Criteria

- [ ] The TUI imports no `_`-prefixed name from another module.
- [ ] `DebugSession` has no dependency on `curses`, `cmd` or `sys.stdout`.
