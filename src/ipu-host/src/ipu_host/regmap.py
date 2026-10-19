"""The register maps, compiled from their SystemRDL sources.

``load("ipu_host")`` and ``load("proto_platform")`` return a ``RegisterMap``:
every register and memory by its path (``REGS.STATUS``,
``REGS.CR_BANK[1].CR[3]``, ``IMEM0``), with its byte address and fields. The
``.rdl`` file is the only place an offset or a field is written down; the
driver, the model and the generators all read it through here.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from peakrdl_regblock.udps import ALL_UDPS
from systemrdl import RDLCompiler
from systemrdl.node import AddrmapNode, FieldNode, MemNode, RegNode

#: The register maps this package defines, by top-level addrmap name.
MAP_NAMES = ("ipu_host", "proto_platform")

RDL_DIR = Path(__file__).parent / "rdl"


class RegisterMapError(KeyError):
    """A path or field that the register map does not define."""


@dataclass(frozen=True)
class Field:
    """One field of a register."""

    name: str
    lsb: int
    width: int
    sw_readable: bool
    sw_writable: bool
    reset: int | None

    @property
    def mask(self) -> int:
        """The field's bits within its register."""
        return ((1 << self.width) - 1) << self.lsb


@dataclass(frozen=True)
class Register:
    """One 32-bit register."""

    path: str
    address: int
    fields: Mapping[str, Field]
    external: bool

    def field(self, name: str) -> Field:
        """The field called ``name``.

        Raises:
            RegisterMapError: The register has no such field.
        """
        try:
            return self.fields[name]
        except KeyError:
            raise RegisterMapError(f"{self.path} has no field {name!r}") from None

    @property
    def reset(self) -> int:
        """The register's reset value; a field with no reset counts as 0."""
        return sum((f.reset or 0) << f.lsb for f in self.fields.values())

    def encode(self, **values: int) -> int:
        """The register value with each named field set and every other field 0.

        Raises:
            RegisterMapError: A name is not a field, or a value does not fit.
        """
        word = 0
        for name, value in values.items():
            f = self.field(name)
            if not 0 <= value < (1 << f.width):
                raise RegisterMapError(f"{self.path}.{name}: {value} does not fit {f.width} bits")
            word |= value << f.lsb
        return word

    def decode(self, word: int) -> dict[str, int]:
        """Every field's value in ``word``."""
        return {name: (word & f.mask) >> f.lsb for name, f in self.fields.items()}


@dataclass(frozen=True)
class Memory:
    """A memory window: ``size`` bytes from ``address``."""

    path: str
    address: int
    size: int

    def contains(self, address: int) -> bool:
        """Whether the byte ``address`` is in the window."""
        return self.address <= address < self.address + self.size


@dataclass(frozen=True)
class RegisterMap:
    """One addrmap: its registers and memories by path."""

    name: str
    size: int
    registers: Mapping[str, Register]
    memories: Mapping[str, Memory]

    def reg(self, path: str) -> Register:
        """The register at ``path``.

        Raises:
            RegisterMapError: No register has that path.
        """
        try:
            return self.registers[path]
        except KeyError:
            raise RegisterMapError(f"{self.name} has no register {path!r}") from None

    def mem(self, path: str) -> Memory:
        """The memory at ``path``.

        Raises:
            RegisterMapError: No memory has that path.
        """
        try:
            return self.memories[path]
        except KeyError:
            raise RegisterMapError(f"{self.name} has no memory {path!r}") from None

    def at(self, address: int) -> Register | None:
        """The register at byte ``address``, or None."""
        return self._by_address.get(address)

    @functools.cached_property
    def _by_address(self) -> Mapping[int, Register]:
        return {r.address: r for r in self.registers.values()}


def rdl_path(name: str) -> Path:
    """The SystemRDL source of the map ``name``."""
    if name not in MAP_NAMES:
        raise RegisterMapError(f"no register map {name!r}; known: {', '.join(MAP_NAMES)}")
    return RDL_DIR / f"{name}.rdl"


def elaborate(name: str) -> AddrmapNode:
    """The elaborated SystemRDL tree of the map ``name``, for the generators."""
    compiler = RDLCompiler()
    # The regblock exporter's properties (buffer_writes, ...) must be known
    # to the compiler, even when no source uses them.
    for udp in ALL_UDPS:
        compiler.register_udp(udp)
    compiler.compile_file(str(rdl_path(name)))
    return compiler.elaborate(top_def_name=name).top


@functools.cache
def load(name: str) -> RegisterMap:
    """The register map ``name`` (``"ipu_host"`` or ``"proto_platform"``)."""
    top = elaborate(name)
    registers: dict[str, Register] = {}
    memories: dict[str, Memory] = {}
    for node in top.descendants(unroll=True):
        if isinstance(node, RegNode):
            path = _relative_path(node, top)
            registers[path] = Register(
                path=path,
                address=node.absolute_address,
                fields=MappingProxyType({f.inst_name: _field(f) for f in node.fields()}),
                external=node.external,
            )
        elif isinstance(node, MemNode):
            path = _relative_path(node, top)
            memories[path] = Memory(path=path, address=node.absolute_address, size=node.size)
    return RegisterMap(
        name=name,
        size=top.size,
        registers=MappingProxyType(registers),
        memories=MappingProxyType(memories),
    )


def _relative_path(node: RegNode | MemNode, top: AddrmapNode) -> str:
    return node.get_path()[len(top.inst_name) + 1 :]


def _field(node: FieldNode) -> Field:
    reset = node.get_property("reset")
    return Field(
        name=node.inst_name,
        lsb=node.lsb,
        width=node.width,
        sw_readable=node.is_sw_readable,
        sw_writable=node.is_sw_writable,
        reset=reset if isinstance(reset, int) else None,
    )
