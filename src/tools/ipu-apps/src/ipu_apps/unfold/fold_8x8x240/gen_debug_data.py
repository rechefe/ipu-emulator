"""FP32 debug-input generation for fold_8x8x240.

Fold's natural input is unfold_8x8x240's RAW, UNCROPPED per-channel row
output (the ``.rows.bin`` sibling file that kernel's own teardown() writes,
not its cropped ``[N_STREAMS, N_OUT, N_TOK]`` convenience array), so this
generator builds a random spatial tensor, packs it via
``unfold_8x8x240.pack_input_rows`` (the documented INPUT CONTRACT for that
kernel), runs the REAL unfold_8x8x240 kernel binary on it (via the emulator,
in-process), and writes its raw rows as fold's input. This gives every
consumer of ``generate()`` -- ``__main__.py`` here and the debug runner -- a
free round-trip flavour without duplicating unfold's own generation recipe.

Self-contained aside from importing unfold_8x8x240 itself (deliberately, to
avoid re-deriving its row-packing recipe by hand and risking drift).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.unfold.unfold_8x8x240 import (
    H, W, C, N_STREAMS, N_OUT, LANES, pack_input_rows, Unfold8x8x240App,
)

_SEED = 0x058

_UNFOLD_ASM_PATH = (
    Path(__file__).resolve().parents[1] / "unfold_8x8x240" / "unfold_8x8x240.asm"
)


def _write(path: Path, arr: np.ndarray) -> Path:
    path.write_bytes(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
    return path


def generate(out_dir: Path) -> dict[str, Path]:
    """Write FP32 inputs into ``out_dir``; return app-constructor kwargs.

    Also writes ``spatial_reference_fp32.bin`` (the original [C, H, W]
    tensor, NAIVE row-major layout -- matches this kernel's OUTPUT contract)
    alongside the app inputs, for callers that want to check a round-trip
    without re-deriving the random tensor themselves.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(_SEED)

    x = rng.uniform(-1.0, 1.0, size=(C, H, W)).astype(np.float32)
    _write(out_dir / "spatial_reference_fp32.bin", x)

    packed = pack_input_rows(x)
    unfold_input_path = _write(out_dir / "_unfold_input_fp32.bin", packed)

    # Run the real unfold_8x8x240 kernel to get fold's natural input (the
    # RAW uncropped rows, not the cropped convenience array).
    with tempfile.TemporaryDirectory() as tmp:
        unfold_bin = Path(tmp) / "unfold_8x8x240.bin"
        assemble_to_bin_file(_UNFOLD_ASM_PATH.read_text(encoding="utf-8"), str(unfold_bin))

        unfold_output_path = Path(tmp) / "unfold_output.bin"
        state = IpuState(
            wide_vector_debug=True,
            wide_vector_arithmetic=WideVectorArithmetic.FP32,
        )
        app = Unfold8x8x240App(
            inst_path=unfold_bin,
            input_path=unfold_input_path,
            output_path=unfold_output_path,
        )
        app.run(max_cycles=20_000_000, state=state)

        # teardown() wrote the raw uncropped rows alongside the cropped
        # output, at <output_path>.rows.bin -- that IS fold's input format.
        rows_path = unfold_output_path.with_suffix(".rows.bin")
        fold_input_path = out_dir / "input_fp32.bin"
        fold_input_path.write_bytes(rows_path.read_bytes())

    return {"input_path": out_dir / "input_fp32.bin"}
