"""Interactive GDB-like debug CLI for the IPU emulator.

All register display/get/set commands are **auto-generated** from
the single source of truth in ``ipu_common.registers`` — adding a new
register descriptor automatically creates the corresponding debug commands.

Built on Python's ``cmd.Cmd`` for readline support, history, and ``?`` help.

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
import struct
import sys
import types
from io import StringIO
from pathlib import Path
from typing import Any, TextIO

# Single source of truth — import register types/schema from ipu_common
from ipu_common.types import RegDescriptor, RegDtype, RegKind
from ipu_common.registers import create_regfile_schema

from ipu_emu.ipu_state import IpuState, INST_MEM_SIZE
from ipu_emu.emulator import DebugAction

# Re-export so callers only need this module
__all__ = ["debug_prompt", "DebugAction", "format_register", "DebugCLI"]

# Build the schema once at import time
REGFILE_SCHEMA: list[RegDescriptor] = create_regfile_schema()


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
        self.use_rawinput = False
        self._result: DebugAction = DebugAction.QUIT
        self._generate_register_commands()

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

    def do_disasm(self, args: str) -> None:
        """Disassemble current instruction."""
        self.out.write(disassemble_current(self.state) + "\n")

    def do_save(self, args: str) -> None:
        """Save registers to JSON.  Usage: save [filename]"""
        filename = args.strip() if args.strip() else "ipu_debug_dump.json"
        save_state_json(self.state, filename)
        self.out.write(f"Registers saved to {filename}\n")

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
