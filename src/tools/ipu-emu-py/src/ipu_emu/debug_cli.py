"""Interactive GDB-like debug CLI for the IPU emulator.

All register display/get/set commands are **auto-generated** from
the single source of truth in ``ipu_common.registers`` — adding a new
register descriptor automatically creates the corresponding debug commands.

Built on Python's ``cmd.Cmd`` for readline support, history, and ``?`` help.

Debugger **verbs** (``set``, ``get``, ``continue``, …) are matched **case-insensitively**
on the first word, same spirit as assembly mnemonics — they still do **not** go through
the Lark assembler (that path is only for ``*.asm`` / ``assemble()``).

Usage from Python::

    from ipu_emu.debug_cli import debug_prompt, DebugAction
    action = debug_prompt(state)          # interactive REPL
    action = debug_prompt(state, level=2) # auto-save JSON on entry

Or as a callback for :func:`emulator.run_with_debug`::

    from ipu_emu.debug_cli import debug_prompt
    run_with_debug(state, lambda s, c: debug_prompt(s))
"""

from __future__ import annotations

import cmd
import json
import shlex
import struct
import sys
import types
from io import StringIO
from pathlib import Path
from typing import Any, TextIO

try:
    import readline  # noqa: F401 -- importing enables input() history/editing
except ImportError:  # pragma: no cover - unavailable on some platforms
    readline = None

# Single source of truth — import register types/schema from ipu_common
from ipu_common.types import RegDescriptor, RegDtype, RegKind
from ipu_common.registers import create_regfile_schema

from ipu_emu.ipu_state import IpuState, INST_MEM_SIZE
from ipu_emu.emulator import DebugAction
from ipu_emu.ipu import XMEM_ADDRESSABLE_ROWS, xmem_row_size_bytes
from ipu_emu.xmem import XMEM_SIZE_BYTES, XMEM_WIDTH_BYTES

# Re-export so callers only need this module
__all__ = [
    "debug_prompt",
    "DebugAction",
    "format_register",
    "DebugCLI",
    "VISIBLE_ALPHABET",
    "encode_16bit_cell",
    "decode_16bit_cell",
]

# Build the schema once at import time
REGFILE_SCHEMA: list[RegDescriptor] = create_regfile_schema()

_XMEM_FORMAT_ITEM_SIZE_BYTES = {
    "hex": 1,
    "int8": 1,
    "u8": 1,
    "cell16": 2,
    "u32": 4,
    "f32": 4,
}
_XMEM_FORMATS = tuple(_XMEM_FORMAT_ITEM_SIZE_BYTES)
_XMEM_FORMATS_TEXT = "|".join(_XMEM_FORMATS)
_XMEM_VALUES_PER_LINE = {
    "hex": 16,
    "int8": 16,
    "u8": 16,
    "cell16": 32,
    "u32": 4,
    "f32": 4,
}

_HEX_BYTES_PER_GROUP = 4
_CELL16_CHARACTERS_PER_GROUP = 2
_INT8_DECIMAL_WIDTH = 4
_U8_DECIMAL_WIDTH = 3
_U32_DECIMAL_WIDTH = 10
_F32_DISPLAY_WIDTH = 15
_F32_SIGNIFICANT_DIGITS = 9

_CELL16_CHARACTER_MASK = 0xFF
_CELL16_COLOR_MASK = 0xF
_CELL16_FOREGROUND_SHIFT = 8
_CELL16_BACKGROUND_SHIFT = 12
_ANSI_STANDARD_COLOR_COUNT = 8
_ANSI_STANDARD_FG_BASE = 30
_ANSI_BRIGHT_FG_BASE = 90
_ANSI_STANDARD_BG_BASE = 40
_ANSI_BRIGHT_BG_BASE = 100

# A control-free character for every possible 8-bit cell value. The ASCII and
# Latin-1 ranges provide 188 entries; Latin Extended-A supplies the remaining
# 68 so all indices 0..255 are valid.
_VISIBLE_ASCII = [chr(i) for i in range(33, 127)]
_VISIBLE_LATIN1 = [chr(i) for i in range(161, 256) if i != 173]
_VISIBLE_EXTENDED = [chr(i) for i in range(0x100, 0x144)]
VISIBLE_ALPHABET = _VISIBLE_ASCII + _VISIBLE_LATIN1 + _VISIBLE_EXTENDED
assert len(VISIBLE_ALPHABET) == 256


def encode_16bit_cell(char_byte: int, fg_4bit: int, bg_4bit: int) -> str:
    """Render one character/color cell as an ANSI terminal string."""
    char = VISIBLE_ALPHABET[char_byte & _CELL16_CHARACTER_MASK]
    fg = fg_4bit & _CELL16_COLOR_MASK
    bg = bg_4bit & _CELL16_COLOR_MASK
    ansi_fg = (
        _ANSI_STANDARD_FG_BASE + fg
        if fg < _ANSI_STANDARD_COLOR_COUNT
        else _ANSI_BRIGHT_FG_BASE + (fg - _ANSI_STANDARD_COLOR_COUNT)
    )
    ansi_bg = (
        _ANSI_STANDARD_BG_BASE + bg
        if bg < _ANSI_STANDARD_COLOR_COUNT
        else _ANSI_BRIGHT_BG_BASE + (bg - _ANSI_STANDARD_COLOR_COUNT)
    )
    return f"\033[{ansi_fg};{ansi_bg}m{char}\033[0m"


def decode_16bit_cell(
    char_code: int, fg_4bit: int, bg_4bit: int
) -> tuple[str, int, int]:
    """Return the visible character and normalized 4-bit colors."""
    return (
        VISIBLE_ALPHABET[char_code & _CELL16_CHARACTER_MASK],
        fg_4bit & _CELL16_COLOR_MASK,
        bg_4bit & _CELL16_COLOR_MASK,
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _hex32(val: int) -> str:
    return f"0x{val:08x}"


def _format_bytes(data: bytes | bytearray, offset: int = 0, count: int = 16) -> str:
    """Format *count* bytes starting at *offset* as hex."""
    end = min(offset + count, len(data))
    if offset < 0 or offset >= len(data):
        return f"Error: offset {offset} out of range [0, {len(data) - 1}]"
    hexes = " ".join(f"{b:02x}" for b in data[offset:end])
    return f"bytes[{offset}..{end - 1}]: {hexes}"


def _format_words(data: bytes | bytearray, word_offset: int = 0, count: int = 4) -> str:
    """Format *count* uint32 words starting at *word_offset* as hex."""
    total_words = len(data) // 4
    if word_offset < 0 or word_offset >= total_words:
        return f"Error: word offset {word_offset} out of range [0, {total_words - 1}]"
    end = min(word_offset + count, total_words)
    vals = []
    for i in range(word_offset, end):
        val = struct.unpack_from("<I", data, i * 4)[0]
        vals.append(f"{val:08x}")
    return f"words[{word_offset}..{end - 1}]: {' '.join(vals)}"


def format_register(
    state: IpuState,
    desc: RegDescriptor,
    index: int = 0,
    offset: int = 0,
    count: int | None = None,
    as_words: bool = False,
) -> str:
    """Format a register value for display.

    Parameters
    ----------
    state : IpuState
    desc : RegDescriptor
    index : int
        Element index for array registers (e.g. r0 vs r1).
    offset : int
        Byte or word offset within the element.
    count : int or None
        Number of bytes/words to display (defaults: 16 bytes, 4 words).
    as_words : bool
        If True, display as uint32 words.
    """
    if desc.dtype in (RegDtype.UINT32, RegDtype.INT32):
        # Scalar register (LR / CR)
        val = state.regfile.get_scalar(desc.name, index)
        return f"{desc.name}{index} = {val} ({_hex32(val)})"

    # Byte-array register
    data = state.regfile.get_register_bytes(desc.name, index)
    if as_words:
        return _format_words(data, offset, count or 4)
    return _format_bytes(data, offset, count or 16)


# ---------------------------------------------------------------------------
# Register summary printers (mirror the C print_* functions)
# ---------------------------------------------------------------------------


def _print_scalar_group(state: IpuState, name: str, out: TextIO) -> None:
    desc = state.regfile._desc(name)
    header = f"=== {name.upper()} Registers ==="
    out.write(header + "\n")
    for i in range(desc.count):
        val = state.regfile.get_scalar(name, i)
        out.write(f"  {name}{i:>2d} = {val:>10d} ({_hex32(val)})\n")


def _print_byte_register(
    state: IpuState, name: str, out: TextIO, preview_bytes: int = 16
) -> None:
    desc = state.regfile._desc(name)
    data = state.regfile.raw(name)
    if desc.count == 1:
        header = f"=== {name} ({desc.size_bytes} bytes) ==="
        out.write(header + "\n")
        hexes = " ".join(f"{b:02x}" for b in data[:preview_bytes])
        out.write(f"  {name}: {hexes} ...\n")
    else:
        header = f"=== {name} ({desc.size_bytes} bytes × {desc.count}) ==="
        out.write(header + "\n")
        for idx in range(desc.count):
            start = idx * desc.size_bytes
            hexes = " ".join(f"{b:02x}" for b in data[start : start + preview_bytes])
            out.write(f"  {name}{idx}: {hexes} ...\n")


def print_all_registers(state: IpuState, out: TextIO | None = None) -> str:
    """Print all registers, matching C ``cmd_regs`` output format.

    Returns the formatted string (also writes to *out* if given).
    """
    buf = StringIO()
    buf.write(f"=== Program Counter ===\n  PC = {state.program_counter}\n")

    for desc in REGFILE_SCHEMA:
        if desc.dtype in (RegDtype.UINT32, RegDtype.INT32):
            _print_scalar_group(state, desc.name, buf)
        else:
            _print_byte_register(state, desc.name, buf)

    text = buf.getvalue()
    if out is not None:
        out.write(text)
    return text


# ---------------------------------------------------------------------------
# JSON export (matches C save_registers_to_json output)
# ---------------------------------------------------------------------------


def state_to_json_dict(state: IpuState) -> dict[str, Any]:
    """Serialise all IPU state to a JSON-compatible dict.

    The output structure matches the C ``save_registers_to_json`` format:
    ``pc``, ``lr``, ``cr``, ``r_regs``, ``r_cyclic``, ``r_mask``, ``acc``.
    """
    d: dict[str, Any] = {"pc": state.program_counter}

    for desc in REGFILE_SCHEMA:
        raw = state.regfile.raw(desc.name)
        if desc.dtype in (RegDtype.UINT32, RegDtype.INT32):
            arr = [state.regfile.get_scalar(desc.name, i) for i in range(desc.count)]
            d[desc.name] = arr
        elif desc.count > 1:
            d[desc.name] = [
                list(raw[i * desc.size_bytes : (i + 1) * desc.size_bytes])
                for i in range(desc.count)
            ]
        else:
            d[desc.name] = list(raw)

    return d


def save_state_json(state: IpuState, path: str | Path) -> None:
    """Write the full IPU state to a JSON file."""
    data = state_to_json_dict(state)
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Disassembly
# ---------------------------------------------------------------------------


def disassemble_current(state: IpuState) -> str:
    """Disassemble the instruction at the current PC."""
    from ipu_as.compound_inst import CompoundInst

    pc = state.program_counter
    if pc >= INST_MEM_SIZE:
        return "PC out of bounds"

    inst = state.inst_mem[pc]
    if inst is None:
        return f"PC {pc}: <nop>"

    # Re-encode the field dict back to an int and use CompoundInst.decode()
    fields = CompoundInst.get_fields()
    word = 0
    shift = 0
    for name, width in fields:
        word |= (inst.get(name, 0) & ((1 << width) - 1)) << shift
        shift += width

    return f"PC {pc}: {CompoundInst.decode(word)}"


# ---------------------------------------------------------------------------
# Register resolution helper
# ---------------------------------------------------------------------------


def _resolve_register(name: str) -> tuple[RegDescriptor | None, int]:
    """Resolve a register name (e.g. "lr5", "r0", "acc") to (descriptor, index).

    Returns (None, 0) if not found.
    """
    # Try exact match on canonical name
    for desc in REGFILE_SCHEMA:
        if name == desc.name:
            return desc, 0
        # Check debug aliases
        for alias in desc.debug_aliases:
            if name == alias:
                if desc.count > 1:
                    for i in range(desc.count):
                        if alias == f"{desc.name}{i}":
                            return desc, i
                return desc, 0

    # Try pattern: "lr5", "cr12" → scalar registers
    for desc in REGFILE_SCHEMA:
        if desc.dtype in (RegDtype.UINT32, RegDtype.INT32):
            if name.startswith(desc.name):
                suffix = name[len(desc.name) :]
                try:
                    idx = int(suffix)
                    if 0 <= idx < desc.count:
                        return desc, idx
                except ValueError:
                    pass

    # Try pattern: "r0", "r1" (for array-type registers)
    for desc in REGFILE_SCHEMA:
        if desc.count > 1 and desc.dtype not in (RegDtype.UINT32, RegDtype.INT32):
            if name.startswith(desc.name):
                suffix = name[len(desc.name) :]
                try:
                    idx = int(suffix)
                    if 0 <= idx < desc.count:
                        return desc, idx
                except ValueError:
                    pass

    return None, 0


def _parse_int(s: str) -> int | None:
    """Parse an integer, supporting 0x hex prefix."""
    try:
        return int(s, 0)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# XMEM helpers
# ---------------------------------------------------------------------------


def _resolve_address_term(state: IpuState, token: str) -> tuple[int | None, str | None]:
    """Resolve an XMEM address term from an immediate or indexed LR/CR."""
    immediate = _parse_int(token)
    if immediate is not None:
        return immediate, None

    lower = token.lower()
    if not (lower.startswith("lr") or lower.startswith("cr")):
        return None, f"Invalid address term '{token}': expected an immediate, lrN, or crN"

    prefix = lower[:2]
    suffix = lower[2:]
    if not suffix.isdigit():
        return None, f"Invalid address register '{token}': expected {prefix}N"

    desc, index = _resolve_register(lower)
    if desc is None or desc.name != prefix:
        return None, f"Unknown address register: {token}"
    return state.regfile.get_scalar(desc.name, index), None


def _resolve_xmem_address(
    state: IpuState, mode: str, base_token: str, offset_token: str
) -> tuple[int | None, dict[str, int | str] | None, str | None]:
    """Resolve row/byte debugger operands to a physical XMEM byte address."""
    mode = mode.lower()
    if mode not in ("row", "byte"):
        return None, None, "XMEM address mode must be 'row' or 'byte'"

    base, error = _resolve_address_term(state, base_token)
    if error is not None:
        return None, None, error
    offset, error = _resolve_address_term(state, offset_token)
    if error is not None:
        return None, None, error
    assert base is not None and offset is not None

    effective = base + offset
    if effective < 0:
        return None, None, f"XMEM {mode} address must be non-negative; got {effective}"

    metadata: dict[str, int | str] = {"address_mode": mode}
    if mode == "row":
        row_size = xmem_row_size_bytes(state)
        if effective >= XMEM_ADDRESSABLE_ROWS:
            return (
                None,
                None,
                f"XMEM row {effective} out of range "
                f"[0, {XMEM_ADDRESSABLE_ROWS - 1}]",
            )
        byte_address = effective * row_size
        metadata.update(
            {
                "row": effective,
                "row_size_bytes": row_size,
                "byte_address": byte_address,
            }
        )
    else:
        if effective >= XMEM_SIZE_BYTES:
            return (
                None,
                None,
                f"XMEM byte address {effective} out of range [0, {XMEM_SIZE_BYTES - 1}]",
            )
        byte_address = effective
        metadata["byte_address"] = byte_address

    return byte_address, metadata, None


def _xmem_range_error(
    byte_address: int, byte_count: int, limit: int = XMEM_SIZE_BYTES
) -> str | None:
    if byte_count < 1:
        return "XMEM read size must be positive"
    end = byte_address + byte_count
    if end > limit:
        return (
            f"XMEM range out of bounds: byte address {byte_address}, "
            f"size {byte_count}, end {end}, max {limit}"
        )
    return None


def _xmem_address_limit(state: IpuState, mode: str) -> int:
    """Return the accessible byte count for an XMEM addressing mode."""
    if mode == "row":
        return XMEM_ADDRESSABLE_ROWS * xmem_row_size_bytes(state)
    return XMEM_SIZE_BYTES


def _format_xmem(
    data: bytes | bytearray, byte_address: int, fmt: str, row_size: int
) -> str:
    """Format an already-validated XMEM range for interactive display."""
    values_per_line = _XMEM_VALUES_PER_LINE[fmt]
    item_size = _XMEM_FORMAT_ITEM_SIZE_BYTES[fmt]
    lines: list[str] = []
    total_items = len(data) // item_size
    start = 0

    while start < total_items:
        address = byte_address + start * item_size
        row = address // row_size
        row_address = row * row_size
        if start == 0 or address == row_address:
            marker = f"--- XMEM row {row} starts at byte address 0x{row_address:08x}"
            if address != row_address:
                marker += f"; display starts at 0x{address:08x}"
            lines.append(marker + " ---")

        items_left_in_row = (row_address + row_size - address) // item_size
        count = min(values_per_line, total_items - start, items_left_in_row)
        if fmt == "hex":
            byte_values = [f"{value:02x}" for value in data[start : start + count]]
            if row_size > XMEM_WIDTH_BYTES:
                values = "  ".join(
                    " ".join(byte_values[i : i + _HEX_BYTES_PER_GROUP])
                    for i in range(0, len(byte_values), _HEX_BYTES_PER_GROUP)
                )
            else:
                values = " ".join(byte_values)
        elif fmt == "int8":
            unpacked = (
                struct.unpack_from("<b", data, start + i)[0]
                for i in range(count)
            )
            values = " ".join(
                f"{value:>{_INT8_DECIMAL_WIDTH}d}" for value in unpacked
            )
        elif fmt == "u8":
            values = " ".join(
                f"{data[start + i]:0{_U8_DECIMAL_WIDTH}d}" for i in range(count)
            )
        elif fmt == "cell16":
            rendered: list[str] = []
            for i in range(count):
                cell = struct.unpack_from("<H", data, (start + i) * item_size)[0]
                char_code = cell & _CELL16_CHARACTER_MASK
                fg_4bit = (
                    cell >> _CELL16_FOREGROUND_SHIFT
                ) & _CELL16_COLOR_MASK
                bg_4bit = (
                    cell >> _CELL16_BACKGROUND_SHIFT
                ) & _CELL16_COLOR_MASK
                rendered.append(encode_16bit_cell(char_code, fg_4bit, bg_4bit))
            values = " ".join(
                "".join(rendered[i : i + _CELL16_CHARACTERS_PER_GROUP])
                for i in range(0, len(rendered), _CELL16_CHARACTERS_PER_GROUP)
            )
        elif fmt == "u32":
            unpacked = (
                struct.unpack_from("<I", data, (start + i) * item_size)[0]
                for i in range(count)
            )
            values = " ".join(
                f"{value:0{_U32_DECIMAL_WIDTH}d}" for value in unpacked
            )
        else:
            unpacked = (
                struct.unpack_from("<f", data, (start + i) * item_size)[0]
                for i in range(count)
            )
            values = " ".join(
                f"{value:>{_F32_DISPLAY_WIDTH}.{_F32_SIGNIFICANT_DIGITS}g}"
                for value in unpacked
            )
        lines.append(f"  0x{address:08x}: {values}")
        start += count

    return "\n".join(lines)


def _xmem_sidecar_path(json_path: Path) -> Path:
    return json_path.with_suffix(".xmem.bin")


def _save_state_with_xmem(
    state: IpuState,
    json_path: Path,
    byte_address: int,
    byte_count: int,
    metadata: dict[str, int | str],
) -> Path:
    """Save registers to JSON and one raw XMEM range to a binary sidecar."""
    sidecar_path = _xmem_sidecar_path(json_path)
    xmem_data = bytes(state.xmem.read_address(byte_address, byte_count))
    data = state_to_json_dict(state)
    data["xmem"] = {
        **metadata,
        "file": sidecar_path.name,
        "byte_address": byte_address,
        "byte_count": byte_count,
    }

    sidecar_path.write_bytes(xmem_data)
    json_path.write_text(json.dumps(data, indent=2) + "\n")
    return sidecar_path


# ---------------------------------------------------------------------------
# The Debug CLI class — built on cmd.Cmd
# ---------------------------------------------------------------------------


class DebugCLI(cmd.Cmd):
    """Interactive debug CLI — instantiated per breakpoint entry.

    Uses Python's ``cmd.Cmd`` for readline, history, and ``help_*`` support.
    Register group commands (``do_lr``, ``do_acc``, etc.) are auto-generated
    from ``REGFILE_SCHEMA`` (sourced from ``ipu_common``).
    """

    prompt = "debug >>> "
    intro = ""  # We print our own banner in run()

    def __init__(
        self, state: IpuState, out: TextIO = sys.stdout, inp: TextIO = sys.stdin
    ):
        super().__init__(stdin=inp, stdout=out)
        self.state = state
        self.out = out
        self.inp = inp
        input_is_tty = bool(getattr(inp, "isatty", lambda: False)())
        output_is_tty = bool(getattr(out, "isatty", lambda: False)())
        self.use_rawinput = bool(input_is_tty and output_is_tty and readline is not None)
        self._result: DebugAction = DebugAction.QUIT
        self._generate_register_commands()

    def precmd(self, line: str) -> str:
        """If the first token names a ``do_*`` handler, normalize it to lowercase.

        So ``SET lr0 1`` dispatches to ``do_set`` like ``set lr0 1``, without using
        the assembly parser.
        """
        stripped = line.strip()
        if not stripped:
            return line
        parts = stripped.split(maxsplit=1)
        verb = parts[0]
        tail = parts[1] if len(parts) > 1 else ""
        canon = verb.lower()
        if hasattr(self, f"do_{canon}"):
            return f"{canon} {tail}".rstrip() if tail else canon
        return line

    def _generate_register_commands(self) -> None:
        """Auto-generate ``do_<name>`` methods for each register group."""
        for desc in REGFILE_SCHEMA:
            all_names = [desc.name] + list(desc.debug_aliases)
            for cmd_name in all_names:
                if not hasattr(self, f"do_{cmd_name}"):

                    def _make_handler(d: RegDescriptor):
                        def handler(self_inner, args: str) -> None:
                            if d.dtype in (RegDtype.UINT32, RegDtype.INT32):
                                _print_scalar_group(
                                    self_inner.state, d.name, self_inner.out
                                )
                            else:
                                _print_byte_register(
                                    self_inner.state, d.name, self_inner.out
                                )

                        handler.__doc__ = f"Print {d.name} register(s)"
                        return handler

                    setattr(
                        self,
                        f"do_{cmd_name}",
                        types.MethodType(_make_handler(desc), self),
                    )

    # -- Exit commands ------------------------------------------------------

    def do_continue(self, args: str) -> bool:
        """Continue execution."""
        self.out.write("Continuing execution...\n")
        self._result = DebugAction.CONTINUE
        return True

    def do_c(self, args: str) -> bool:
        """Continue execution (shortcut)."""
        return self.do_continue(args)

    def do_quit(self, args: str) -> bool:
        """Quit debugger and halt execution."""
        self.out.write("Halting execution.\n")
        self.state.program_counter = INST_MEM_SIZE
        self._result = DebugAction.QUIT
        return True

    def do_q(self, args: str) -> bool:
        """Quit debugger (shortcut)."""
        return self.do_quit(args)

    def do_step(self, args: str) -> bool:
        """Execute one instruction and break again."""
        self.out.write("Stepping one instruction...\n")
        self._result = DebugAction.STEP
        return True

    # -- Register commands --------------------------------------------------

    def do_regs(self, args: str) -> None:
        """Print all registers."""
        print_all_registers(self.state, self.out)

    def do_pc(self, args: str) -> None:
        """Print program counter."""
        self.out.write(
            f"=== Program Counter ===\n  PC = {self.state.program_counter}\n"
        )

    def do_get(self, args: str) -> None:
        """Get register value.  Usage: get <register> [offset] [count]"""
        parts = args.split()
        if not parts:
            self.out.write("Usage: get <register> [offset] [count]\n")
            return
        reg_name = parts[0]
        offset = _parse_int(parts[1]) if len(parts) >= 2 else 0
        count = _parse_int(parts[2]) if len(parts) >= 3 else None

        if reg_name == "pc":
            self.out.write(f"pc = {self.state.program_counter}\n")
            return

        desc, idx = _resolve_register(reg_name)
        if desc is None:
            self.out.write(f"Unknown register: {reg_name}\n")
            return

        if offset is None:
            self.out.write("Invalid offset\n")
            return

        text = format_register(
            self.state, desc, idx, offset or 0, count, as_words=False
        )
        self.out.write(f"{reg_name} {text}\n")

    def do_getw(self, args: str) -> None:
        """Get register words.  Usage: getw <register> [word_offset] [count]"""
        parts = args.split()
        if not parts:
            self.out.write("Usage: getw <register> [word_offset] [count]\n")
            return
        reg_name = parts[0]
        offset = _parse_int(parts[1]) if len(parts) >= 2 else 0
        count = _parse_int(parts[2]) if len(parts) >= 3 else None

        desc, idx = _resolve_register(reg_name)
        if desc is None:
            self.out.write(f"Unknown register: {reg_name}\n")
            return
        if desc.dtype in (RegDtype.UINT32, RegDtype.INT32):
            self.out.write(f"Use 'get {reg_name}' for scalar registers\n")
            return

        if offset is None:
            self.out.write("Invalid offset\n")
            return

        text = format_register(self.state, desc, idx, offset or 0, count, as_words=True)
        self.out.write(f"{reg_name} {text}\n")

    def do_set(self, args: str) -> None:
        """Set register value.  Usage: set <register> <value>"""
        parts = args.split()
        if len(parts) < 2:
            self.out.write("Usage: set <register> <value>\n")
            return
        reg_name = parts[0]
        value = _parse_int(parts[1])
        if value is None:
            self.out.write(f"Invalid value: {parts[1]}\n")
            return

        if reg_name == "pc":
            self.state.program_counter = value
            self.out.write(f"Set pc = {value}\n")
            return

        desc, idx = _resolve_register(reg_name)
        if desc is None:
            self.out.write(f"Unknown register: {reg_name}\n")
            return

        if desc.dtype in (RegDtype.UINT32, RegDtype.INT32):
            self.state.regfile.set_scalar(desc.name, idx, value)
            self.out.write(f"Set {reg_name} = {value}\n")
        else:
            self.out.write(
                f"Cannot set byte-array register '{reg_name}' with a scalar value\n"
            )

    def do_xmem(self, args: str) -> None:
        """Read XMEM. Usage: xmem row|byte BASE OFFSET COUNT FORMAT"""
        parts = args.split()
        if len(parts) != 5:
            self.out.write(
                "Usage: xmem row|byte BASE OFFSET COUNT "
                f"{_XMEM_FORMATS_TEXT}\n"
            )
            return

        mode, base_token, offset_token, count_token, fmt = parts
        fmt = fmt.lower()
        if fmt not in _XMEM_FORMAT_ITEM_SIZE_BYTES:
            self.out.write(
                f"XMEM format must be one of: {_XMEM_FORMATS_TEXT}\n"
            )
            return

        count = _parse_int(count_token)
        if count is None or count < 1:
            self.out.write(f"XMEM count must be a positive integer; got '{count_token}'\n")
            return

        byte_address, metadata, error = _resolve_xmem_address(
            self.state, mode, base_token, offset_token
        )
        if error is not None:
            self.out.write(error + "\n")
            return
        assert byte_address is not None and metadata is not None

        item_size = _XMEM_FORMAT_ITEM_SIZE_BYTES[fmt]
        if item_size > 1 and byte_address % item_size:
            self.out.write(
                f"XMEM {fmt} address must be {item_size}-byte aligned; "
                f"got {byte_address}\n"
            )
            return
        byte_count = count * item_size
        limit = _xmem_address_limit(self.state, str(metadata["address_mode"]))
        error = _xmem_range_error(byte_address, byte_count, limit)
        if error is not None:
            self.out.write(error + "\n")
            return

        data = self.state.xmem.read_address(byte_address, byte_count)
        if metadata["address_mode"] == "row":
            heading = (
                f"=== XMEM row {metadata['row']} "
                f"(byte address 0x{byte_address:08x}, {fmt}) ==="
            )
        else:
            heading = f"=== XMEM byte address 0x{byte_address:08x} ({fmt}) ==="
        self.out.write(heading + "\n")
        self.out.write(
            _format_xmem(
                data,
                byte_address,
                fmt,
                xmem_row_size_bytes(self.state),
            )
            + "\n"
        )

    def do_disasm(self, args: str) -> None:
        """Disassemble current instruction."""
        self.out.write(disassemble_current(self.state) + "\n")

    def do_save(self, args: str) -> None:
        """Save state. Usage: save [FILE [xmem all|row|byte ...]]"""
        stripped = args.strip()
        if not stripped:
            filename = "ipu_debug_dump.json"
            save_state_json(self.state, filename)
            self.out.write(f"Registers saved to {filename}\n")
            return

        try:
            parts = shlex.split(stripped)
        except ValueError as error:
            self.out.write(f"Invalid save arguments: {error}\n")
            return

        if len(parts) == 1:
            filename = parts[0]
            save_state_json(self.state, filename)
            self.out.write(f"Registers saved to {filename}\n")
            return

        usage = (
            "Usage: save [FILE]\n"
            "       save FILE xmem all\n"
            "       save FILE xmem row BASE OFFSET ROW_COUNT\n"
            "       save FILE xmem byte BASE OFFSET BYTE_COUNT\n"
        )
        if parts[1].lower() != "xmem":
            # Keep the original register-only command compatible with unquoted
            # filenames containing spaces. XMEM syntax reserves the second
            # token "xmem"; quote such filenames when using an XMEM suffix.
            filename = stripped
            save_state_json(self.state, filename)
            self.out.write(f"Registers saved to {filename}\n")
            return

        json_path = Path(parts[0])
        if len(parts) == 3 and parts[2].lower() == "all":
            byte_address = 0
            byte_count = XMEM_SIZE_BYTES
            metadata: dict[str, int | str] = {"address_mode": "all"}
        elif len(parts) == 6 and parts[2].lower() in ("row", "byte"):
            mode, base_token, offset_token, count_token = parts[2:]
            count = _parse_int(count_token)
            if count is None or count < 1:
                self.out.write(
                    f"XMEM count must be a positive integer; got '{count_token}'\n"
                )
                return

            byte_address, metadata, error = _resolve_xmem_address(
                self.state, mode, base_token, offset_token
            )
            if error is not None:
                self.out.write(error + "\n")
                return
            assert byte_address is not None and metadata is not None
            byte_count = (
                count * int(metadata["row_size_bytes"])
                if mode.lower() == "row"
                else count
            )
        else:
            self.out.write(usage)
            return

        limit = _xmem_address_limit(self.state, str(metadata["address_mode"]))
        error = _xmem_range_error(byte_address, byte_count, limit)
        if error is not None:
            self.out.write(error + "\n")
            return

        sidecar = _save_state_with_xmem(
            self.state, json_path, byte_address, byte_count, metadata
        )
        self.out.write(f"Registers saved to {json_path}\n")
        self.out.write(f"XMEM saved to {sidecar}\n")

    def default(self, line: str) -> None:
        """Handle unknown commands."""
        self.out.write(
            f"Unknown command: {line}. Type 'help' for available commands.\n"
        )

    # -- REPL entry point ---------------------------------------------------

    def run(self, level: int = 0) -> DebugAction:
        """Enter the interactive debug prompt and return the chosen action."""
        self.out.write("\n========================================\n")
        self.out.write(f"IPU Debug - Break at PC={self.state.program_counter}\n")
        self.out.write("========================================\n")

        # Level 0: print registers
        if level >= 0:
            self.do_pc("")
            _print_scalar_group(self.state, "lr", self.out)

        # Level 1: disassemble
        if level >= 1:
            self.out.write("\n=== Current Instruction ===\n")
            self.out.write(f"  {disassemble_current(self.state)}\n")

        # Level 2: auto-save JSON
        if level >= 2:
            filename = f"ipu_debug_pc{self.state.program_counter}.json"
            save_state_json(self.state, filename)
            self.out.write(f"Registers saved to {filename}\n")

        self.out.write(
            "\nType 'help' for available commands, "
            "'continue' or 'c' to resume execution.\n\n"
        )

        self.cmdloop(intro="")
        return self._result


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------


def debug_prompt(
    state: IpuState,
    cycle: int = 0,
    level: int = 0,
    out: TextIO = sys.stdout,
    inp: TextIO = sys.stdin,
) -> DebugAction:
    """Enter the interactive debug prompt.

    Suitable as a callback for :func:`emulator.run_with_debug`::

        run_with_debug(state, lambda s, c: debug_prompt(s, c))

    Parameters
    ----------
    state : IpuState
        The current IPU state.
    cycle : int
        Current cycle count (passed by ``run_with_debug``, informational).
    level : int
        Debug verbosity level (0–2), matching C ``ipu_debug__level_t``.
    """
    cli = DebugCLI(state, out=out, inp=inp)
    return cli.run(level=level)
