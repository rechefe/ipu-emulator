#!/usr/bin/env python3
"""Generate Python and Rust register metadata from ipu_ctrl.rdl."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape
from systemrdl import RDLCompiler
from systemrdl.node import AddrmapNode, FieldNode, MemNode, RegNode


@dataclass
class FieldInfo:
    name: str
    lsb: int
    msb: int
    mask: int
    width: int
    enum_name: str | None = None
    enum_members: list[tuple[str, int]] = field(default_factory=list)


@dataclass
class RegInfo:
    name: str
    offset: int
    width: int
    fields: list[FieldInfo]


@dataclass
class MemInfo:
    name: str
    offset: int
    memwidth: int
    mementries: int


@dataclass
class EnumInfo:
    name: str
    members: list[tuple[str, int]]


@dataclass
class ModelInfo:
    top: str
    parameters: dict[str, int]
    registers: list[RegInfo]
    memories: list[MemInfo]
    enums: list[EnumInfo]


def _field_mask(lsb: int, msb: int) -> int:
    width = msb - lsb + 1
    return ((1 << width) - 1) << lsb


def _collect_enum(field: FieldNode) -> tuple[str | None, list[tuple[str, int]]]:
    enum_type = field.get_property("encode", default=None)
    if enum_type is None:
        return None, []
    members: list[tuple[str, int]] = []
    enum_name = enum_type.type_name
    for member_name, member_val in enum_type.members.items():
        members.append((member_name, int(member_val.value)))
    return enum_name, members


def _collect_model(
    rdl_path: Path,
    *,
    top: str,
    parameters: dict[str, int],
) -> ModelInfo:
    rdlc = RDLCompiler()
    rdlc.compile_file(str(rdl_path))
    root = rdlc.elaborate(top_def_name=top, parameters=parameters)
    top_node = root.top

    registers: list[RegInfo] = []
    memories: list[MemInfo] = []

    for node in top_node.descendants():
        if isinstance(node, RegNode):
            reg_inst = node.inst
            fields: list[FieldInfo] = []
            for field_node in node.fields():
                finst = field_node.inst
                enum_name, enum_members = _collect_enum(field_node)
                fields.append(
                    FieldInfo(
                        name=finst.inst_name,
                        lsb=finst.lsb,
                        msb=finst.msb,
                        mask=_field_mask(finst.lsb, finst.msb),
                        width=finst.msb - finst.lsb + 1,
                        enum_name=enum_name,
                        enum_members=enum_members,
                    )
                )
            if node.is_array:
                for idx, elem in enumerate(node.unrolled()):
                    registers.append(
                        RegInfo(
                            name=f"{reg_inst.inst_name}{idx}",
                            offset=int(elem.absolute_address),
                            width=int(elem.get_property("regwidth")),
                            fields=fields,
                        )
                    )
            else:
                registers.append(
                    RegInfo(
                        name=reg_inst.inst_name,
                        offset=int(node.absolute_address),
                        width=int(node.get_property("regwidth")),
                        fields=fields,
                    )
                )
        elif isinstance(node, MemNode):
            minst = node.inst
            memories.append(
                MemInfo(
                    name=minst.inst_name,
                    offset=int(node.absolute_address),
                    memwidth=int(node.get_property("memwidth")),
                    mementries=int(node.get_property("mementries")),
                )
            )

    registers.sort(key=lambda r: r.offset)
    memories.sort(key=lambda m: m.offset)

    enum_map: dict[str, list[tuple[str, int]]] = {}
    for reg in registers:
        for fld in reg.fields:
            if fld.enum_name and fld.enum_name not in enum_map:
                enum_map[fld.enum_name] = fld.enum_members
    enums = [EnumInfo(name=n, members=m) for n, m in enum_map.items()]

    resolved: dict[str, int] = {}
    if isinstance(top_node, AddrmapNode):
        for param in top_node.inst.parameters:
            resolved[param.name] = int(param.get_value())
    resolved.update({k: int(v) for k, v in parameters.items()})

    return ModelInfo(
        top=top,
        parameters=resolved,
        registers=registers,
        memories=memories,
        enums=enums,
    )


def _default_instruction_aligned_bytes() -> int:
    try:
        from ipu_as.lark_tree import instruction_aligned_bytes_len

        return int(instruction_aligned_bytes_len())
    except Exception:
        return 8


def _render(template_dir: Path, template_name: str, **ctx: Any) -> str:
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(enabled_extensions=()),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env.get_template(template_name).render(**ctx)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rdl", type=Path, required=True)
    parser.add_argument("--top", default="ipu_host")
    parser.add_argument(
        "--instruction-aligned-bytes",
        type=int,
        default=None,
        help="Override INSTRUCTION_ALIGNED_BYTES elaboration parameter",
    )
    parser.add_argument("--python-out", type=Path, required=True)
    parser.add_argument("--rust-out", type=Path, required=True)
    args = parser.parse_args(argv)

    inst_bytes = (
        args.instruction_aligned_bytes
        if args.instruction_aligned_bytes is not None
        else _default_instruction_aligned_bytes()
    )
    parameters = {"INSTRUCTION_ALIGNED_BYTES": inst_bytes}

    model = _collect_model(args.rdl, top=args.top, parameters=parameters)
    mmio_base = model.parameters["MMIO_BASE"]
    imem_base = model.parameters["IMEM_BASE"]
    imem_depth = model.parameters["IMEM_DEPTH"]
    imem_map_size = imem_depth * inst_bytes

    template_dir = Path(__file__).resolve().parent / "templates"
    ctx = {
        "model": model,
        "mmio_base": mmio_base,
        "imem_base": imem_base,
        "imem_depth": imem_depth,
        "instruction_aligned_bytes": inst_bytes,
        "imem_map_size": imem_map_size,
        "ctrl_block_size": 0x1000,
    }

    args.python_out.parent.mkdir(parents=True, exist_ok=True)
    args.rust_out.parent.mkdir(parents=True, exist_ok=True)
    args.python_out.write_text(
        _render(template_dir, "ipu_ctrl_regs.py.j2", **ctx), encoding="utf-8"
    )
    args.rust_out.write_text(_render(template_dir, "lib.rs.j2", **ctx), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
