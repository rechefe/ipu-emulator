"""Walk the SystemRDL IR and build a codegen model for templates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from systemrdl import RDLCompiler, RDLListener, RDLWalker
from systemrdl.node import AddrmapNode, FieldNode, MemNode, RegNode


@dataclass(frozen=True)
class EnumEntry:
    name: str
    value: int


@dataclass(frozen=True)
class EnumDef:
    name: str
    py_class_name: str
    rust_enum_name: str
    entries: list[EnumEntry]


@dataclass(frozen=True)
class FieldDef:
    name: str
    low: int
    high: int
    mask: int
    reset: int
    sw_access: str
    singlepulse: bool
    enum_name: str | None


@dataclass(frozen=True)
class RegDef:
    name: str
    offset: int
    absolute_offset: int
    fields: list[FieldDef]


@dataclass(frozen=True)
class MemDef:
    name: str
    base: int
    size: int


@dataclass(frozen=True)
class CodegenModel:
    registers: list[RegDef]
    enums: list[EnumDef]
    memories: list[MemDef]
    ctrl_base: int
    imem_depth: int
    inst_aligned_bytes: int
    ram_base: int = 0x8000_0000
    ram_size: int = 0x0010_0000


class _ModelBuilder(RDLListener):
    def __init__(self, ctrl_base: int) -> None:
        self.ctrl_base = ctrl_base
        self.registers: list[RegDef] = []
        self.memories: list[MemDef] = []
        self.enums: dict[str, list[EnumEntry]] = {}

    def enter_Reg(self, node: RegNode) -> None:
        if not _is_under_ctrl(node):
            return
        rel = node.address_offset
        fields: list[FieldDef] = []
        for f in node.fields():
            encode = f.get_property("encode")
            enum_name = None
            if encode is not None and hasattr(encode, "members"):
                raw = str(encode).split()[1] if " " in str(encode) else encode.__name__
                enum_name = raw.rstrip(">")
                if enum_name not in self.enums:
                    self.enums[enum_name] = [
                        EnumEntry(name=name, value=int(member.value))
                        for name, member in encode.members.items()
                    ]
            low, high = f.low, f.high
            width = high - low + 1
            mask = ((1 << width) - 1) << low
            reset = f.get_property("reset")
            reset_val = int(reset) if reset is not None else 0
            sw = f.get_property("sw")
            fields.append(
                FieldDef(
                    name=f.inst_name,
                    low=low,
                    high=high,
                    mask=mask,
                    reset=reset_val,
                    sw_access=str(sw),
                    singlepulse=bool(f.get_property("singlepulse")),
                    enum_name=enum_name,
                )
            )
        self.registers.append(
            RegDef(
                name=node.inst_name,
                offset=rel,
                absolute_offset=self.ctrl_base + rel,
                fields=fields,
            )
        )

    def enter_Mem(self, node: MemNode) -> None:
        self.memories.append(
            MemDef(
                name=node.inst_name,
                base=node.absolute_address,
                size=node.size,
            )
        )


def _is_under_ctrl(node: RegNode) -> bool:
    parent = node.parent
    while parent is not None:
        if isinstance(parent, AddrmapNode) and parent.inst_name == "ctrl":
            return True
        parent = parent.parent
    return False


def _enum_class_name(rdl_name: str) -> str:
    return "".join(part.capitalize() for part in rdl_name.split("_"))


def _collect_enums(enum_map: dict[str, list[EnumEntry]]) -> list[EnumDef]:
    return [
        EnumDef(
            name=name,
            py_class_name=_enum_class_name(name),
            rust_enum_name=_enum_class_name(name),
            entries=entries,
        )
        for name, entries in sorted(enum_map.items())
    ]


def build_model(
    rdl_path: str,
    *,
    imem_depth: int,
    inst_aligned_bytes: int,
) -> CodegenModel:
    rdlc = RDLCompiler()
    rdlc.compile_file(rdl_path)
    root = rdlc.elaborate()

    ctrl_base = 0x1000_0000
    for child in root.children():
        if isinstance(child, AddrmapNode) and child.inst_name == "ctrl":
            ctrl_base = child.absolute_address
            break

    builder = _ModelBuilder(ctrl_base)
    walker = RDLWalker(unroll=True)
    walker.walk(root, builder)

    return CodegenModel(
        registers=sorted(builder.registers, key=lambda r: r.offset),
        enums=_collect_enums(builder.enums),
        memories=builder.memories,
        ctrl_base=ctrl_base,
        imem_depth=imem_depth,
        inst_aligned_bytes=inst_aligned_bytes,
    )
