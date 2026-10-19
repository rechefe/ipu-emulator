"""Run one program in flat mode on the card or on the model, and compare the
result with the reference model.

    bazel run //fpga/alveo-u55c:run_flat -- PROGRAM.bin --format int8 --cr 2=3 --cr 3=40 \\
        [--xmem IMAGE.bin] [--xclbin build/hw/ipu_proto.xclbin]

Without --xclbin it runs on ipu_host.model.ModelDevice. PROGRAM.bin is an
``ipu-as`` program file (emulator words, the default output). It prints the
first differing XMEM row and LR, and exits 1 on a difference.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ipu_emu.reference import ReferenceModel
from ipu_host.driver import FLAT_BYTES, ROW_BYTES, run_flat
from ipu_host.model import ModelDevice
from ipu_isa.codec.word import unpack_words
from ipu_isa.formats import ElementFormat


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("program", type=Path)
    parser.add_argument("--format", default="int8", choices=[f.value for f in ElementFormat])
    parser.add_argument("--cr", action="append", default=[], help="INDEX=VALUE")
    parser.add_argument("--xmem", type=Path, help="starting XMEM image, at most 2 MiB")
    parser.add_argument("--xclbin", type=Path, help="run on the card with this xclbin")
    args = parser.parse_args(argv)

    words = unpack_words(args.program.read_bytes())
    fmt = ElementFormat(args.format)
    crs = {int(k): int(v, 0) for k, v in (item.split("=") for item in args.cr)}
    image = args.xmem.read_bytes() if args.xmem else b""

    if args.xclbin:
        from xrt_device import XrtDevice  # noqa: PLC0415 - only on a machine with XRT

        device = XrtDevice(args.xclbin)
    else:
        device = ModelDevice()
    run = run_flat(device, words, fmt, crs, image)

    model = ReferenceModel(fmt)
    model.load_program(words)
    for index, value in crs.items():
        model.write_register("cr", value, index=index)
    if image:
        model.write_xmem(0, image)
    result = model.run()
    xmem = model.read_xmem(0, FLAT_BYTES)
    lrs = [model.read_register("lr", i) for i in range(16)]

    print(
        f"status {run.status.run_state.name}, retired {run.retired_words} (model {result.cycles})"
    )
    ok = run.retired_words == result.cycles and run.lrs == lrs and run.xmem == xmem
    for index, (got, want) in enumerate(zip(run.lrs, lrs, strict=True)):
        if got != want:
            print(f"LR{index}: {got} (model {want})")
    for row in range(FLAT_BYTES // ROW_BYTES):
        span = slice(row * ROW_BYTES, (row + 1) * ROW_BYTES)
        if run.xmem[span] != xmem[span]:
            print(f"first differing XMEM row: {row}")
            break
    print("MATCH" if ok else "DIFFER")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
