"""Deprecated GDB-like debug CLI for the IPU emulator.

Retained for existing Python callers. Use ``bazel run --config=debug TARGET`` for the TUI.

All register display/get/set commands are **auto-generated** from
the single source of truth in ``ipu_isa.registers`` — adding a new
register descriptor automatically creates the corresponding debug commands.

Built on Python's ``cmd.Cmd`` for readline support, history, and ``?`` help.

Debugger **verbs** (``set``, ``get``, ``continue``, …) are matched **case-insensitively**
on the first word, same spirit as assembly mnemonics — they still do **not** go through
the Lark assembler (that path is only for ``*.asm`` / ``assemble()``).

Usage from Python::

    from ipu_debug.cli import debug_prompt, DebugAction
    action = debug_prompt(state)          # interactive REPL
    action = debug_prompt(state, level=2) # auto-save JSON on entry

Or as a callback for :func:`emulator.run_with_debug`::

    from ipu_debug.cli import debug_prompt
    from ipu_debug.control import DebugControl
    control = DebugControl()
    run_with_debug(state, lambda s, c: debug_prompt(s, c, control=control), control=control)
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

# Single source of truth — import register types/schema from ipu_isa
from ipu_debug.control import DebugControl
from ipu_debug.format import (
    VISIBLE_ALPHABET,
    decode_16bit_cell,
    encode_16bit_cell,
    format_xmem,
)
from ipu_debug.session import (
    REGFILE_SCHEMA,
    DebugError,
    DebugSession,
    XmemRequest,
    XmemWindow,
    disassemble_at,
    parse_xmem_count,
    resolve_xmem_address,
    xmem_address_limit,
    xmem_range_error,
)
from ipu_emu.emulator import DebugAction
from ipu_emu.ipu_state import IpuState
from ipu_emu.xmem import XMEM_SIZE_BYTES
from ipu_isa.types import RegDescriptor, RegDtype

# Re-export so callers only need this module
__all__ = [
    "debug_prompt",
    "DebugAction",
    "format_register",
    "DebugCLI",
    "VISIBLE_ALPHABET",
    "encode_16bit_cell",
    "decode_16bit_cell",
    "disassemble_at",
    "XmemRequest",
    "ResolvedXmemRequest",
]

# The resolved XMEM window's former name, kept for existing callers.
ResolvedXmemRequest = XmemWindow


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
# Register summary printers
# ---------------------------------------------------------------------------


def _print_scalar_group(state: IpuState, name: str, out: TextIO) -> None:
    desc = state.regfile.descriptor(name)
    header = f"=== {name.upper()} Registers ==="
    out.write(header + "\n")
    for i in range(desc.count):
        val = state.regfile.get_scalar(name, i)
        out.write(f"  {name}{i:>2d} = {val:>10d} ({_hex32(val)})\n")


def _print_byte_register(state: IpuState, name: str, out: TextIO, preview_bytes: int = 16) -> None:
    desc = state.regfile.descriptor(name)
    data = state.regfile.raw_readonly(name)
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
    """Print the PC and every register-file register, one group per register.

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
# JSON export
# ---------------------------------------------------------------------------


def state_to_json_dict(state: IpuState) -> dict[str, Any]:
    """Serialise all IPU state to a JSON-compatible dict.

    The keys are ``pc`` and one per register-file register (``REGFILE_SCHEMA``):
    a list of values for a scalar group, a list of byte lists for a
    multi-entry vector register, a byte list otherwise.
    """
    d: dict[str, Any] = {"pc": state.program_counter}

    for desc in REGFILE_SCHEMA:
        raw = state.regfile.raw_readonly(desc.name)
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
    return disassemble_at(state, state.program_counter)


# ---------------------------------------------------------------------------
# XMEM save helpers
# ---------------------------------------------------------------------------


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
    xmem_data = state.xmem.snapshot_bytes(byte_address, byte_count)
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
    """Deprecated debug CLI — retained for existing breakpoint callbacks.

    Uses Python's ``cmd.Cmd`` for readline, history, and ``help_*`` support.
    Register group commands (``do_lr``, ``do_acc``, etc.) are auto-generated
    from ``REGFILE_SCHEMA`` (sourced from ``ipu_isa``).
    """

    prompt = "debug >>> "
    intro = ""  # We print our own banner in run()

    def __init__(
        self,
        state: IpuState,
        out: TextIO = sys.stdout,
        inp: TextIO = sys.stdin,
        cycle: int = 0,
        control: DebugControl | None = None,
    ):
        super().__init__(stdin=inp, stdout=out)
        self.state = state
        # The run's control, from whoever started the run; else this prompt's own.
        self.control = control if control is not None else DebugControl()
        self.session = DebugSession(state, self.control)
        self.cycle = cycle
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
                                _print_scalar_group(self_inner.state, d.name, self_inner.out)
                            else:
                                _print_byte_register(self_inner.state, d.name, self_inner.out)

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
        self.session.halt()
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

    def _report(self, error: DebugError) -> None:
        self.out.write(f"{error}\n")

    def do_break(self, args: str) -> None:
        """Add a runtime breakpoint. Usage: break [PC]"""
        try:
            pc = self.session.parse_pc(args.strip() or str(self.state.program_counter))
            self.session.add_breakpoint(pc)
        except DebugError as error:
            return self._report(error)
        self.out.write(f"Breakpoint at PC {pc}\n")

    def do_breaks(self, args: str) -> None:
        """List runtime breakpoints. Usage: breaks"""
        if args.strip():
            self.out.write("Usage: breaks\n")
            return
        if not self.control.breakpoints:
            self.out.write("No runtime breakpoints\n")
            return
        pcs = ", ".join(map(str, sorted(self.control.breakpoints)))
        self.out.write(f"Breakpoints: {pcs}\n")

    def do_delete(self, args: str) -> None:
        """Remove runtime breakpoints. Usage: delete PC|all"""
        if args.strip().lower() == "all":
            self.session.clear_breakpoints()
            self.out.write("All runtime breakpoints removed\n")
            return
        try:
            pc = self.session.parse_pc(args.strip())
            self.session.remove_breakpoint(pc)
        except DebugError as error:
            return self._report(error)
        self.out.write(f"Breakpoint removed at PC {pc}\n")

    def do_until(self, args: str) -> bool | None:
        """Run to a PC or the next stop, whichever comes first. Usage: until PC"""
        try:
            pc = self.session.parse_pc(args.strip())
            resume = self.session.run_until(pc)
        except DebugError as error:
            return self._report(error)
        if not resume:
            self.out.write(f"Already at PC {pc}\n")
            return None
        return self.do_continue("")

    def _optional_int(self, parts: list[str], position: int, default: int | None) -> int | None:
        """The integer at *position*, *default* when absent, None when malformed."""
        if len(parts) <= position:
            return default
        try:
            return self.session.parse_int(parts[position])
        except DebugError:
            return None

    def do_regs(self, args: str) -> None:
        """Print all registers."""
        print_all_registers(self.state, self.out)

    def do_pc(self, args: str) -> None:
        """Print program counter."""
        self.out.write(f"=== Program Counter ===\n  PC = {self.state.program_counter}\n")

    def do_get(self, args: str) -> None:
        """Get register value.  Usage: get <register> [offset] [count]"""
        parts = args.split()
        if not parts:
            self.out.write("Usage: get <register> [offset] [count]\n")
            return
        reg_name = parts[0]
        offset = self._optional_int(parts, 1, 0)
        count = self._optional_int(parts, 2, None)

        if reg_name == "pc":
            self.out.write(f"pc = {self.state.program_counter}\n")
            return

        try:
            desc, idx = self.session.resolve_register(reg_name)
        except DebugError as error:
            self.out.write(f"{error}\n")
            return

        if offset is None:
            self.out.write("Invalid offset\n")
            return

        text = format_register(self.state, desc, idx, offset or 0, count, as_words=False)
        self.out.write(f"{reg_name} {text}\n")

    def do_getw(self, args: str) -> None:
        """Get register words.  Usage: getw <register> [word_offset] [count]"""
        parts = args.split()
        if not parts:
            self.out.write("Usage: getw <register> [word_offset] [count]\n")
            return
        reg_name = parts[0]
        offset = self._optional_int(parts, 1, 0)
        count = self._optional_int(parts, 2, None)

        try:
            desc, idx = self.session.resolve_register(reg_name)
        except DebugError as error:
            self.out.write(f"{error}\n")
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
        try:
            value = self.session.parse_int(parts[1])
        except DebugError:
            self.out.write(f"Invalid value: {parts[1]}\n")
            return
        try:
            self.session.set_register(parts[0], value)
        except DebugError as error:
            return self._report(error)
        self.out.write(f"Set {parts[0]} = {value}\n")

    def do_xmem(self, args: str) -> None:
        """Read XMEM. Usage: xmem row|byte BASE OFFSET COUNT FORMAT"""
        try:
            request = self.session.parse_xmem(args)
            resolved = self.session.read_xmem(request)
        except DebugError as error:
            self.out.write(f"{error}\n")
            return

        if resolved.metadata["address_mode"] == "row":
            heading = (
                f"=== XMEM row {resolved.metadata['row']} "
                f"(byte address 0x{resolved.byte_address:08x}, {request.fmt}) ==="
            )
        else:
            heading = f"=== XMEM byte address 0x{resolved.byte_address:08x} ({request.fmt}) ==="
        self.out.write(heading + "\n")
        self.out.write(
            format_xmem(
                resolved.data,
                resolved.byte_address,
                request.fmt,
                resolved.row_size,
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
            try:
                count = parse_xmem_count(count_token)
                byte_address, metadata = resolve_xmem_address(
                    self.state, mode, base_token, offset_token
                )
            except DebugError as error:
                self.out.write(f"{error}\n")
                return
            byte_count = count * int(metadata["row_size_bytes"]) if mode.lower() == "row" else count
        else:
            self.out.write(usage)
            return

        limit = xmem_address_limit(self.state, str(metadata["address_mode"]))
        try:
            xmem_range_error(byte_address, byte_count, limit)
        except DebugError as error:
            self.out.write(f"{error}\n")
            return

        sidecar = _save_state_with_xmem(self.state, json_path, byte_address, byte_count, metadata)
        self.out.write(f"Registers saved to {json_path}\n")
        self.out.write(f"XMEM saved to {sidecar}\n")

    def default(self, line: str) -> None:
        """Handle unknown commands."""
        self.out.write(f"Unknown command: {line}. Type 'help' for available commands.\n")

    # -- REPL entry point ---------------------------------------------------

    def run(self, level: int = 0) -> DebugAction:
        """Enter the line-oriented debug prompt and return the chosen action."""
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
            "\nType 'help' for available commands, 'continue' or 'c' to resume execution.\n\n"
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
    control: DebugControl | None = None,
) -> DebugAction:
    """Enter the deprecated line debugger retained for existing callers.

    Suitable as a callback for :func:`emulator.run_with_debug`::

        run_with_debug(state, lambda s, c: debug_prompt(s, c, control=control), control=control)

    Parameters
    ----------
    state : IpuState
        The current IPU state.
    cycle : int
        Current cycle count (passed by ``run_with_debug``, informational).
    level : int
        Debug verbosity level (0–2).
    control : DebugControl, optional
        The control passed to ``run_with_debug``, so ``break`` and ``until``
        reach the run; without it they last for this prompt only.
    """
    cli = DebugCLI(state, out=out, inp=inp, cycle=cycle, control=control)
    return cli.run(level=level)
