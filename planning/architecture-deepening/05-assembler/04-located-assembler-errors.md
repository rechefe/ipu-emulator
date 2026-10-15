# `AssemblerError` hierarchy with `SourceLocation`

Part of [Epic: Assembler as a deep module](00-epic.md) · ID E05-04 · Size S · Depends on: E03 (`IpuError` root in `ipu_isa`) · Behaviour change: none

## Goal

Every user-facing assembler failure raises a typed exception that carries a structured location, as part of the shared `IpuError` hierarchy (Q28, Q59). Message text stays byte-identical, so the VS Code extension (which keys on message text) and `test_diagnostics` see no change. E05-07 then reads the location from the exception instead of regex-parsing the message.

## Background

- Assembler failures are bare `ValueError`s whose message embeds `"Line N, Column M"`. That text is formatted in several places:
  - `ipu_token.py:14` (`get_location_string`) and `ipu_token.py:38` (`_raise_error`);
  - `inst.py:94-97` (operand count);
  - `compound_inst.py:66, 71` (slot overflow);
  - `label.py:16-19, 25-27` (duplicate or undefined label).
- `diagnostics.py:36` (`_LOCATION_RE`) and `:41` (`_QUOTED_NAME_RE`) recover the position and the offending token from that text (`diagnostics.py:233, 403-404`).
- Parse errors are `lark.exceptions.LarkError`. Template errors are `jinja2.TemplateError`.
- Q59 sets two error kinds for the program: hardware faults (F-1..F-9) and configuration errors (setup time). Assembler errors are setup-time errors and are never hardware faults.

## Implementation

1. `ipu_as/errors.py`:

   ```python
   @dataclass(frozen=True)
   class SourceLocation:
       line: int                 # 1-based, in the rendered text
       column: int               # 1-based
       end_column: int | None = None
       source: str = "<input>"   # template name once includes exist (E05-09)
       def __str__(self) -> str: return f"Line {self.line}, Column {self.column}"   # today's text

   class AssemblerError(IpuError, ValueError):     # ValueError keeps existing except / pytest.raises working
       stage: ClassVar[str]                        # "template" | "parse" | "assemble" (the JSON 'stage' values)
       def __init__(self, message: str, *, location: SourceLocation | None = None, token: str | None = None): ...
   class TemplateError(AssemblerError): stage = "template"   # wraps jinja2 errors, __cause__ kept
   class ParseError(AssemblerError):    stage = "parse"      # wraps lark errors, __cause__ kept
   class EncodeError(AssemblerError):   stage = "assemble"
   class OperandError(EncodeError): ...   # bad value, range, operand count
   class SlotError(EncodeError): ...      # unknown mnemonic, too many per slot
   class LabelError(EncodeError): ...     # duplicate or undefined label
   ```

   `IpuError` is the shared root. Assumption: E03 puts it in `ipu_isa.errors`, the one package every other package depends on (Q62). If it is not there yet, add it in this PR as a three-line module and tell E03.
2. Replace each `raise ValueError(...)` listed above with the matching subclass, passing `location=` and `token=`. Build the message with `str(location)`, so the text is unchanged.
3. Do not change where errors are caught yet. Callers keep catching `ValueError`; E05-06 and E05-07 change the catch sites.

## Tests

- [ ] Each error source raises its specific subclass, with the expected `location` and `token` (one parametrized test per site).
- [ ] `str(error)` equals the base-commit message for every case in `test_diagnostics`. The suite passes unchanged.
- [ ] `isinstance(error, ValueError)` and `isinstance(error, IpuError)` both hold.
- [ ] `bazel test //...` green; E02 digests unchanged.

## Acceptance Criteria

- [ ] No bare `raise ValueError` remains in `ipu_as` for user input errors.
- [ ] Message strings are byte-identical, checked by the `test_diagnostics` and `check --json` snapshot (E05-08).

## Open question

- The `IpuError` root location (`ipu_isa.errors`) is assumed, not decided. Confirm with E03.
