"""The debugger core the TUI sits on: parse, resolve, read, mutate, disassemble.

``DebugSession`` has no curses and no stream writes. It turns user
text into requests, reads registers and XMEM windows from an ``IpuState`` and
disassembles instruction memory, returning values. It is also the only place a
debugger changes anything: register and memory writes, the PC, breakpoints and
run-until all go through its methods, each of which bumps ``generation`` so
views can tell their cached reads are stale. Every user-facing failure
is a ``DebugError`` whose message is the text a frontend shows. How values are
laid out on screen stays in the TUI (``ipu_debug.tui``) and in the display
helpers of ``ipu_debug.format``.
"""

from __future__ import annotations

import struct
from collections.abc import Iterable
from dataclasses import dataclass

from ipu_debug.control import DebugControl, validate_pc
from ipu_debug.errors import DebugError
from ipu_emu.errors import EmulatorError
from ipu_emu.ipu import addressable_rows, xmem_row_size_bytes
from ipu_emu.ipu_state import INST_MEM_SIZE, IpuState
from ipu_emu.xmem import XMEM_SIZE_BYTES
from ipu_isa.codec.disasm import disassemble_fields
from ipu_isa.registers import create_regfile_schema
from ipu_isa.types import RegDescriptor, RegDtype

__all__ = [
    "REGFILE_SCHEMA",
    "XMEM_FORMAT_ITEM_SIZE_BYTES",
    "XMEM_FORMATS",
    "Cell",
    "DebugError",
    "DebugSession",
    "DisasmLine",
    "RegisterView",
    "XmemRequest",
    "XmemWindow",
    "disassemble_at",
    "is_scalar",
    "parse_int",
    "parse_xmem_count",
    "parse_xmem_request",
    "resolve_register",
    "resolve_xmem_address",
    "xmem_address_limit",
    "xmem_range_error",
]

REGFILE_SCHEMA: list[RegDescriptor] = create_regfile_schema()

XMEM_FORMAT_ITEM_SIZE_BYTES = {
    "hex": 1,
    "int8": 1,
    "u8": 1,
    "cell16": 2,
    "u32": 4,
    "f32": 4,
}
XMEM_FORMATS = tuple(XMEM_FORMAT_ITEM_SIZE_BYTES)
_XMEM_FORMATS_TEXT = "|".join(XMEM_FORMATS)
_XMEM_STRUCT_FORMATS = {
    "hex": "<B",
    "int8": "<b",
    "u8": "<B",
    "cell16": "<H",
    "u32": "<I",
    "f32": "<f",
}


def is_scalar(desc: RegDescriptor) -> bool:
    """Whether *desc* holds 32-bit scalars (LR, CR) rather than bytes."""
    return desc.dtype in (RegDtype.UINT32, RegDtype.INT32)


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegisterView:
    """One register element: an integer for a scalar, bytes for a vector."""

    name: str
    index: int
    dtype: RegDtype
    value: int | bytes
    width: int  # bytes per element


@dataclass(frozen=True)
class XmemRequest:
    """Unresolved, repeatable XMEM request.

    The base and offset stay tokens (an immediate, ``lrN`` or ``crN``) so a
    request re-resolves against the current registers each time it is read.
    """

    mode: str
    base_token: str
    offset_token: str
    count: int
    fmt: str

    def command(self) -> str:
        return f"xmem {self.mode} {self.base_token} {self.offset_token} {self.count} {self.fmt}"


@dataclass(frozen=True)
class Cell:
    """One decoded XMEM item: its byte address, raw bytes and value."""

    address: int
    raw: bytes
    value: int | float


@dataclass(frozen=True)
class XmemWindow:
    """An XMEM request resolved against the current state, with its bytes."""

    request: XmemRequest
    byte_address: int
    byte_count: int
    row_size: int
    metadata: dict[str, int | str]
    data: bytes

    @property
    def cells(self) -> tuple[Cell, ...]:
        """The window decoded item by item in the request's format."""
        size = XMEM_FORMAT_ITEM_SIZE_BYTES[self.request.fmt]
        layout = _XMEM_STRUCT_FORMATS[self.request.fmt]
        return tuple(
            Cell(
                self.byte_address + start,
                bytes(self.data[start : start + size]),
                struct.unpack_from(layout, self.data, start)[0],
            )
            for start in range(0, len(self.data) - size + 1, size)
        )


@dataclass(frozen=True)
class DisasmLine:
    """One disassembled instruction word."""

    pc: int
    text: str
    is_current: bool
    has_breakpoint: bool


# ---------------------------------------------------------------------------
# Parsing and resolution (state-free)
# ---------------------------------------------------------------------------


def parse_int(text: str) -> int:
    """Parse an integer with an optional ``0x``/``0b``/``0o`` prefix."""
    try:
        return int(text, 0)
    except ValueError:
        raise DebugError(f"Invalid integer: {text}") from None


def _resolve_alias(name: str) -> tuple[RegDescriptor, int] | None:
    """Match a canonical register name or one of its debug aliases."""
    for desc in REGFILE_SCHEMA:
        if name == desc.name:
            return desc, 0
        if name in desc.debug_aliases:
            indexed = [i for i in range(desc.count) if name == f"{desc.name}{i}"]
            return desc, indexed[0] if desc.count > 1 and indexed else 0
    return None


def _resolve_indexed(name: str, scalars: bool) -> tuple[RegDescriptor, int] | None:
    """Match ``<name><index>`` against scalar groups or multi-element arrays."""
    for desc in REGFILE_SCHEMA:
        if is_scalar(desc) != scalars or (not scalars and desc.count <= 1):
            continue
        if not name.startswith(desc.name):
            continue
        try:
            idx = int(name[len(desc.name) :])
        except ValueError:
            continue
        if 0 <= idx < desc.count:
            return desc, idx
    return None


def resolve_register(name: str) -> tuple[RegDescriptor, int]:
    """Resolve a register name (``lr5``, ``r0``, ``acc``) to (descriptor, index).

    Exact names and debug aliases win, then scalar groups (``lr5``, ``cr12``),
    then multi-element arrays (``r0``).
    """
    found = _resolve_alias(name) or _resolve_indexed(name, True) or _resolve_indexed(name, False)
    if found is not None:
        return found
    raise DebugError(f"Unknown register: {name}")


def parse_xmem_count(token: str) -> int:
    """Parse a positive XMEM item count."""
    try:
        count = parse_int(token)
    except DebugError:
        count = 0
    if count < 1:
        raise DebugError(f"XMEM count must be a positive integer; got '{token}'")
    return count


def parse_xmem_request(text: str) -> XmemRequest:
    """Parse an ``xmem`` command tail without resolving register operands."""
    parts = text.split()
    if len(parts) != 5:
        raise DebugError("Usage: xmem row|byte BASE OFFSET COUNT " + _XMEM_FORMATS_TEXT)

    mode, base_token, offset_token, count_token, fmt = parts
    mode = mode.lower()
    fmt = fmt.lower()
    if mode not in ("row", "byte"):
        raise DebugError("XMEM address mode must be 'row' or 'byte'")
    if fmt not in XMEM_FORMAT_ITEM_SIZE_BYTES:
        raise DebugError(f"XMEM format must be one of: {_XMEM_FORMATS_TEXT}")
    return XmemRequest(mode, base_token, offset_token, parse_xmem_count(count_token), fmt)


def xmem_range_error(byte_address: int, byte_count: int, limit: int = XMEM_SIZE_BYTES) -> None:
    """Raise when ``[byte_address, byte_address + byte_count)`` leaves ``[0, limit)``."""
    if byte_count < 1:
        raise DebugError("XMEM read size must be positive")
    end = byte_address + byte_count
    if end > limit:
        raise DebugError(
            f"XMEM range out of bounds: byte address {byte_address}, "
            f"size {byte_count}, end {end}, max {limit}"
        )


# ---------------------------------------------------------------------------
# State-dependent resolution
# ---------------------------------------------------------------------------


def _resolve_address_term(state: IpuState, token: str) -> int:
    """Resolve an XMEM address term from an immediate or indexed LR/CR."""
    try:
        return parse_int(token)
    except DebugError:
        pass

    lower = token.lower()
    if not (lower.startswith("lr") or lower.startswith("cr")):
        raise DebugError(f"Invalid address term '{token}': expected an immediate, lrN, or crN")

    prefix = lower[:2]
    if not lower[2:].isdigit():
        raise DebugError(f"Invalid address register '{token}': expected {prefix}N")

    try:
        desc, index = resolve_register(lower)
    except DebugError:
        desc = None
    if desc is None or desc.name != prefix:
        raise DebugError(f"Unknown address register: {token}")
    return state.regfile.get_scalar(desc.name, index)


def resolve_xmem_address(
    state: IpuState, mode: str, base_token: str, offset_token: str
) -> tuple[int, dict[str, int | str]]:
    """Resolve row/byte operands to a physical XMEM byte address and metadata."""
    mode = mode.lower()
    if mode not in ("row", "byte"):
        raise DebugError("XMEM address mode must be 'row' or 'byte'")

    effective = _resolve_address_term(state, base_token) + _resolve_address_term(
        state, offset_token
    )
    if effective < 0:
        raise DebugError(f"XMEM {mode} address must be non-negative; got {effective}")

    metadata: dict[str, int | str] = {"address_mode": mode}
    if mode == "row":
        row_size = xmem_row_size_bytes(state)
        rows = addressable_rows(state)
        if effective >= rows:
            raise DebugError(f"XMEM row {effective} out of range [0, {rows - 1}]")
        byte_address = effective * row_size
        metadata.update(
            {"row": effective, "row_size_bytes": row_size, "byte_address": byte_address}
        )
    else:
        if effective >= XMEM_SIZE_BYTES:
            raise DebugError(
                f"XMEM byte address {effective} out of range [0, {XMEM_SIZE_BYTES - 1}]"
            )
        byte_address = effective
        metadata["byte_address"] = byte_address
    return byte_address, metadata


def xmem_address_limit(state: IpuState, mode: str) -> int:
    """Return the accessible byte count for an XMEM addressing mode."""
    if mode == "row":
        return addressable_rows(state) * xmem_row_size_bytes(state)
    return XMEM_SIZE_BYTES


def disassemble_at(state: IpuState, pc: int) -> str:
    """Disassemble the instruction at *pc* as ``PC <pc>: <text>``."""
    if pc < 0 or pc >= INST_MEM_SIZE:
        return "PC out of bounds"

    inst = state.inst_mem[pc]
    if inst is None:
        return f"PC {pc}: <nop>"

    return f"PC {pc}: {disassemble_fields(inst)}"


# ---------------------------------------------------------------------------
# The session
# ---------------------------------------------------------------------------


class DebugSession:
    """Parse, resolve, read, mutate and disassemble for one stopped ``IpuState``.

    ``generation`` starts at 0 and goes up by one on every successful
    mutation. A view that caches what it read keys the cache by it.
    """

    def __init__(
        self,
        state: IpuState,
        control: DebugControl | None = None,
        *,
        kernel_name: str | None = None,
    ) -> None:
        self.state = state
        self.control = control if control is not None else DebugControl()
        self.kernel_name = kernel_name
        self.generation = 0

    # -- parsing ------------------------------------------------------------

    @staticmethod
    def parse_int(text: str) -> int:
        """Parse an integer; raise ``DebugError`` when *text* is not one."""
        return parse_int(text)

    def parse_xmem(self, text: str) -> XmemRequest:
        """Parse ``[xmem] row|byte BASE OFFSET COUNT FORMAT`` into a request."""
        stripped = text.strip()
        if stripped.lower().startswith("xmem "):
            stripped = stripped[5:]
        return parse_xmem_request(stripped)

    @staticmethod
    def resolve_register(name: str) -> tuple[RegDescriptor, int]:
        """Resolve a register name to its descriptor and element index."""
        return resolve_register(name)

    def parse_pc(self, text: str) -> int:
        """Parse a user-chosen PC; it must be an integer inside instruction memory."""
        try:
            pc = parse_int(text)
        except DebugError:
            raise DebugError("PC must be an integer") from None
        return validate_pc(pc)

    # -- mutations ----------------------------------------------------------

    def _changed(self) -> None:
        self.generation += 1

    def set_register(self, name: str, value: int) -> RegisterView:
        """Write *value* to the scalar register *name* (``lr3``, ``cr7``, ``pc``).

        Returns the register as written (LR and CR values are masked). Raises
        ``DebugError`` for an unknown name, a byte-array register, a PC outside
        instruction memory or a read-only CR.
        """
        if name == "pc":
            self.state.program_counter = validate_pc(value)
            self._changed()
            return RegisterView("pc", 0, RegDtype.UINT32, value, 4)
        desc, index = resolve_register(name)
        if not is_scalar(desc):
            raise DebugError(f"Cannot set byte-array register '{name}' with a scalar value")
        try:
            self.state.regfile.set_scalar(desc.name, index, value)
        except EmulatorError as error:
            raise DebugError(str(error)) from None
        self._changed()
        return self.register(desc.name, index)

    def write_register_bytes(self, name: str, offset: int, data: bytes) -> None:
        """Overwrite bytes ``[offset, offset + len(data))`` of register *name*."""
        if offset < 0 or offset + len(data) > len(self.state.regfile.view(name)):
            raise DebugError("Selected value is out of range")
        self.state.regfile.mutable(name, offset, len(data))[:] = data
        self._changed()

    def write_xmem(self, byte_address: int, data: bytes) -> None:
        """Overwrite XMEM bytes starting at *byte_address*."""
        try:
            self.state.xmem.write_address(byte_address, data)
        except (EmulatorError, ValueError) as error:
            raise DebugError(str(error)) from None
        self._changed()

    def add_breakpoint(self, pc: int) -> None:
        self.control.add_breakpoint(validate_pc(pc))
        self._changed()

    def remove_breakpoint(self, pc: int) -> None:
        """Remove the breakpoint at *pc*; nothing happens if there is none."""
        self.control.remove_breakpoint(validate_pc(pc))
        self._changed()

    def clear_breakpoints(self) -> None:
        self.control.clear_breakpoints()
        self._changed()

    def toggle_breakpoint(self, pc: int) -> bool:
        """Flip the breakpoint at *pc*; return whether it is now set."""
        enabled = self.control.toggle_breakpoint(validate_pc(pc))
        self._changed()
        return enabled

    def run_until(self, pc: int) -> bool:
        """Arm a one-stop target at *pc*; True when execution should resume.

        False means the state is already at *pc*, so nothing is armed.
        """
        resume = self.control.run_until(validate_pc(pc), self.state.program_counter)
        self._changed()
        return resume

    def halt(self) -> None:
        """Move the PC past instruction memory so the run ends (QUIT)."""
        self.state.program_counter = INST_MEM_SIZE
        self._changed()

    # -- reads --------------------------------------------------------------

    def register(self, name: str, index: int = 0) -> RegisterView:
        """Read element *index* of the register file entry *name*."""
        desc = self.state.regfile.descriptor(name)
        if not 0 <= index < desc.count:
            raise DebugError(f"{name} index {index} out of range [0, {desc.count - 1}]")
        if is_scalar(desc):
            value: int | bytes = self.state.regfile.get_scalar(desc.name, index)
        else:
            start = index * desc.size_bytes
            value = bytes(self.state.regfile.view(desc.name)[start : start + desc.size_bytes])
        return RegisterView(desc.name, index, desc.dtype, value, desc.size_bytes)

    def registers(self, names: Iterable[str] | None = None) -> list[RegisterView]:
        """Read the named registers, or every element of every register."""
        if names is None:
            return [
                self.register(desc.name, index)
                for desc in REGFILE_SCHEMA
                for index in range(desc.count)
            ]
        views = []
        for name in names:
            desc, index = resolve_register(name)
            views.append(self.register(desc.name, index))
        return views

    def read_xmem(self, request: XmemRequest) -> XmemWindow:
        """Resolve *request* against the current registers and read its bytes."""
        byte_address, metadata = resolve_xmem_address(
            self.state, request.mode, request.base_token, request.offset_token
        )
        item_size = XMEM_FORMAT_ITEM_SIZE_BYTES[request.fmt]
        if item_size > 1 and byte_address % item_size:
            raise DebugError(
                f"XMEM {request.fmt} address must be {item_size}-byte aligned; got {byte_address}"
            )
        byte_count = request.count * item_size
        xmem_range_error(
            byte_address, byte_count, xmem_address_limit(self.state, str(metadata["address_mode"]))
        )
        return XmemWindow(
            request=request,
            byte_address=byte_address,
            byte_count=byte_count,
            row_size=xmem_row_size_bytes(self.state),
            metadata=metadata,
            data=self.state.xmem.snapshot_bytes(byte_address, byte_count),
        )

    def disassemble(self, start: int, count: int) -> list[DisasmLine]:
        """Disassemble up to *count* words from *start*, clipped to instruction memory.

        ``text`` is the decoded word without the ``PC <n>:`` prefix (``<nop>``
        for an empty word).
        """
        first = max(0, start)
        last = min(INST_MEM_SIZE, start + max(0, count))
        breakpoints = self.control.breakpoints
        lines = []
        for pc in range(first, last):
            text = disassemble_at(self.state, pc)
            prefix = f"PC {pc}: "
            lines.append(
                DisasmLine(
                    pc=pc,
                    text=text[len(prefix) :] if text.startswith(prefix) else text,
                    is_current=pc == self.state.program_counter,
                    has_breakpoint=pc in breakpoints,
                )
            )
        return lines
