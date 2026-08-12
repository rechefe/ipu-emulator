"""Generate Python, Rust, and HTML artifacts from ipu_ctrl.rdl."""

from __future__ import annotations

import argparse
from pathlib import Path

from jinja2 import Environment, PackageLoader

from ipu_ctrl_rdl.model import CodegenModel, build_model


def _render(template_name: str, model: CodegenModel) -> str:
    env = Environment(
        loader=PackageLoader("ipu_ctrl_rdl", "templates"),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(template_name)
    imem_map_size = model.imem_depth * model.inst_aligned_bytes
    imem = next((m for m in model.memories if m.name == "imem"), None)
    imem_base = imem.base if imem else model.ctrl_base + 0x10000
    return template.render(
        model=model,
        imem_base=imem_base,
        imem_map_size=imem_map_size,
        mmio_size=0x1000,
    )


def generate_python(model: CodegenModel) -> str:
    return _render("python_regs.py.j2", model)


def generate_rust(model: CodegenModel) -> str:
    return _render("rust_pac.rs.j2", model)


def generate_html(rdl_path: str, out_dir: Path) -> None:
    from peakrdl.html import HTMLExporter
    from systemrdl import RDLCompiler

    out_dir.mkdir(parents=True, exist_ok=True)
    rdlc = RDLCompiler()
    rdlc.compile_file(rdl_path)
    root = rdlc.elaborate()
    exporter = HTMLExporter()
    exporter.export(root, str(out_dir))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate IPU host-control artifacts")
    parser.add_argument("--rdl", required=True, help="Path to ipu_ctrl.rdl")
    parser.add_argument("--imem-depth", type=int, default=1024)
    parser.add_argument("--inst-aligned-bytes", type=int)
    parser.add_argument(
        "--inst-aligned-bytes-file",
        type=Path,
        help="File containing the instruction aligned byte count (from Bazel)",
    )
    parser.add_argument("--python-out", type=Path)
    parser.add_argument("--rust-out", type=Path)
    parser.add_argument("--html-out", type=Path)
    args = parser.parse_args(argv)

    if args.inst_aligned_bytes_file:
        inst_aligned_bytes = int(args.inst_aligned_bytes_file.read_text().strip())
    elif args.inst_aligned_bytes is not None:
        inst_aligned_bytes = args.inst_aligned_bytes
    else:
        parser.error("one of --inst-aligned-bytes or --inst-aligned-bytes-file is required")

    model = build_model(
        args.rdl,
        imem_depth=args.imem_depth,
        inst_aligned_bytes=inst_aligned_bytes,
    )

    if args.python_out:
        args.python_out.parent.mkdir(parents=True, exist_ok=True)
        args.python_out.write_text(generate_python(model))

    if args.rust_out:
        args.rust_out.parent.mkdir(parents=True, exist_ok=True)
        args.rust_out.write_text(generate_rust(model))

    if args.html_out:
        generate_html(args.rdl, args.html_out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
