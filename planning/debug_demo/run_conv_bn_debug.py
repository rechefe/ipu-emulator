"""Demo driver: run conv_universal_bn_activation and open the HTML debug window.

Not part of the app suite — a scratch script to exercise the new debug window
tool against a small, real conv_bn_activation instance.
"""
from __future__ import annotations

import webbrowser
from pathlib import Path

import numpy as np

from ipu_as.lark_tree import assemble_to_bin_file
from ipu_emu.debug_window import capture_trace
from ipu_emu.debug_window_html import write_html
from ipu_emu.ipu_state import IpuState

from ipu_apps.convolutions_universal.conv.conv_universal_bn_activation import (
    ConvUniversalBnActivationApp,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ASM_PATH = (
    REPO_ROOT
    / "src" / "tools" / "ipu-apps" / "src" / "ipu_apps"
    / "convolutions_universal" / "conv" / "conv_universal_bn_activation"
    / "conv_universal_bn_activation.asm"
)

OUT_DIR = Path(__file__).resolve().parent
BIN_PATH = OUT_DIR / "conv_bn.bin"
HTML_PATH = OUT_DIR / "conv_bn_trace.html"


def _pack_input_chunked(input_chw: np.ndarray, rows: int, cols: int) -> bytes:
    in_ch, ir, ic = input_chw.shape
    assert (ir, ic) == (rows, cols)
    rows_per_chunk = 128 // cols
    packed = bytearray((rows * cols // 128) * in_ch * 128)
    for ch in range(in_ch):
        for r in range(rows):
            for c in range(cols):
                chunk = r // rows_per_chunk
                local_row = r % rows_per_chunk
                offset = (chunk * in_ch + ch) * 128 + local_row * cols + c
                packed[offset] = np.uint8(input_chw[ch, r, c]).item()
    return bytes(packed)


def main() -> None:
    assemble_to_bin_file(ASM_PATH.read_text(encoding="utf-8"), str(BIN_PATH))

    rows = cols = 16
    in_ch = 4
    out_ch = 2

    rng = np.random.default_rng(0)
    weights = rng.integers(-4, 5, size=(out_ch, in_ch, 3, 3), dtype=np.int8)
    input_chw = rng.integers(-8, 9, size=(in_ch, rows, cols), dtype=np.int8)
    bias = np.array([0, -1], dtype=np.int8)

    input_path = OUT_DIR / "input.bin"
    input_path.write_bytes(_pack_input_chunked(input_chw, rows, cols))

    app = ConvUniversalBnActivationApp(
        inst_path=BIN_PATH,
        input_path=input_path,
        kernel=weights,
        bias=bias,
        output_path=None,
        dtype="INT8",
        rows=rows,
        cols=cols,
        in_channels=in_ch,
        out_channels=out_ch,
    )

    state = IpuState()
    from ipu_emu.emulator import load_program_from_binary
    load_program_from_binary(state, BIN_PATH)
    app.setup(state)

    snapshots = capture_trace(state, start=0, limit=400, max_cycles=200_000)
    write_html(snapshots, HTML_PATH, title="conv_universal_bn_activation debug trace")
    print(f"wrote {HTML_PATH} ({len(snapshots)} cycles captured)")
    webbrowser.open(HTML_PATH.resolve().as_uri())


if __name__ == "__main__":
    main()
