"""CLI for the IPU debug window.

Runs an assembled (or .asm source) program under the tracing emulator and
renders the captured cycle trace as text (paste into chat), a standalone
interactive HTML stepper, or raw JSON.

Examples
--------
    python -m ipu_emu.debug_window_cli prog.asm
    python -m ipu_emu.debug_window_cli prog.bin --format html --out trace.html --open
    python -m ipu_emu.debug_window_cli prog.asm --xmem input.bin@4096 \
        --start 100 --limit 50 --notes
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path

from ipu_emu.debug_window import capture_trace, snapshot_to_dict
from ipu_emu.debug_window_html import write_html
from ipu_emu.debug_window_text import render_trace
from ipu_emu.emulator import load_program, load_program_from_binary
from ipu_emu.execute import decode_instruction_word
from ipu_emu.ipu_state import IpuState, WideVectorArithmetic


def _load_program_any(state: IpuState, path: Path) -> None:
    if path.suffix.lower() == ".asm":
        from ipu_as.lark_tree import assemble  # assembler is a package dep

        words = assemble(path.read_text(encoding="utf-8"))
        load_program(state, [decode_instruction_word(w) for w in words])
    else:
        load_program_from_binary(state, path)


def _parse_xmem_arg(spec: str) -> tuple[Path, int]:
    try:
        file_part, addr_part = spec.rsplit("@", 1)
        return Path(file_part), int(addr_part, 0)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--xmem expects FILE@ADDR (e.g. data.bin@0x1000), got {spec!r}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m ipu_emu.debug_window_cli",
        description="Cycle-by-cycle VLIW debug window for the IPU emulator.")
    ap.add_argument("program", type=Path,
                    help="program to run: .asm source or assembled .bin")
    ap.add_argument("--format", choices=("text", "html", "json"), default="text",
                    help="output renderer (default: text)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output file (default: stdout for text/json, "
                         "<program>.debug.html for html)")
    ap.add_argument("--xmem", type=_parse_xmem_arg, action="append", default=[],
                    metavar="FILE@ADDR",
                    help="load a binary file into XMEM at ADDR (repeatable)")
    ap.add_argument("--start", type=int, default=0,
                    help="first cycle to capture (earlier cycles run uncaptured)")
    ap.add_argument("--limit", type=int, default=1000,
                    help="max cycles to capture (default 1000)")
    ap.add_argument("--max-cycles", type=int, default=1_000_000,
                    help="hard execution cap (infinite-loop guard)")
    ap.add_argument("--notes", action="store_true",
                    help="text format: include per-operand semantic notes")
    ap.add_argument("--wide-vector", action="store_true",
                    help="run in wide-vector debug mode")
    ap.add_argument("--arith", choices=("fp32", "int32"), default="fp32",
                    help="wide-vector lane arithmetic (default fp32)")
    ap.add_argument("--open", action="store_true",
                    help="html format: open the result in a browser")
    args = ap.parse_args(argv)

    state = IpuState(
        wide_vector_debug=args.wide_vector,
        wide_vector_arithmetic=WideVectorArithmetic(args.arith),
    )
    _load_program_any(state, args.program)
    for file_path, addr in args.xmem:
        state.xmem.write_address(addr, file_path.read_bytes())

    snapshots = capture_trace(
        state, start=args.start, limit=args.limit, max_cycles=args.max_cycles)

    if args.format == "text":
        text = render_trace(snapshots, notes=args.notes)
        if args.out:
            args.out.write_text(text + "\n", encoding="utf-8")
        else:
            print(text)
    elif args.format == "json":
        payload = json.dumps([snapshot_to_dict(s) for s in snapshots], indent=1)
        if args.out:
            args.out.write_text(payload + "\n", encoding="utf-8")
        else:
            print(payload)
    else:  # html
        out = args.out or args.program.with_suffix(".debug.html")
        write_html(snapshots, out, title=f"IPU debug window — {args.program.name}")
        print(f"wrote {out} ({len(snapshots)} cycles captured)")
        if args.open:
            webbrowser.open(out.resolve().as_uri())

    return 0


if __name__ == "__main__":
    sys.exit(main())
