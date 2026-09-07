"""FP32 debug-input generation for fold_16x16x192.

Fold's natural input is unfold_16x16x192's output, so this generator builds a
random spatial tensor, runs the REAL unfold_16x16x192 kernel binary on it (via
the emulator, in-process), and writes its output as fold's input. This gives
every consumer of ``generate()`` -- ``__main__.py`` here and the debug runner
-- a free round-trip flavour without duplicating unfold's own generation
recipe.

Self-contained aside from importing unfold_16x16x192 itself (deliberately, to
avoid re-deriving its NHCW-striping recipe by hand and risking drift).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.unfold.unfold_16x16x192 import H, W, C, N_STRIPES, LANES, Unfold16x16x192App

_SEED = 0x016

_UNFOLD_ASM_PATH = (
    Path(__file__).resolve().parents[1] / "unfold_16x16x192" / "unfold_16x16x192.asm"
)


def _write(path: Path, arr: np.ndarray) -> Path:
    path.write_bytes(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
    return path


def _spatial_to_nhcw_striped(x: np.ndarray) -> np.ndarray:
    """[C, H, W] -> NHCW-striped rows, matching unfold_16x16x192's own input recipe."""
    stripe_h = H // N_STRIPES
    src = np.zeros((N_STRIPES * C, LANES), dtype=np.float32)
    for stripe in range(N_STRIPES):
        r0 = stripe * stripe_h
        for ch in range(C):
            src[stripe * C + ch, : stripe_h * W] = x[ch, r0 : r0 + stripe_h, :].reshape(-1)
    return src


def generate(out_dir: Path) -> dict[str, Path]:
    """Write FP32 inputs into ``out_dir``; return app-constructor kwargs.

    Also writes ``spatial_reference_fp32.bin`` (the original [C, H, W]
    tensor) alongside the app inputs, for callers that want to check a
    round-trip without re-deriving the random tensor themselves.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(_SEED)

    x = rng.uniform(-1.0, 1.0, size=(C, H, W)).astype(np.float32)
    _write(out_dir / "spatial_reference_fp32.bin", x)

    striped = _spatial_to_nhcw_striped(x)
    unfold_input_path = _write(out_dir / "_unfold_input_fp32.bin", striped)

    # Run the real unfold_16x16x192 kernel to get fold's natural input.
    with tempfile.TemporaryDirectory() as tmp:
        unfold_bin = Path(tmp) / "unfold_16x16x192.bin"
        assemble_to_bin_file(_UNFOLD_ASM_PATH.read_text(encoding="utf-8"), str(unfold_bin))

        unfold_output_path = out_dir / "input_fp32.bin"
        state = IpuState(
            wide_vector_debug=True,
            wide_vector_arithmetic=WideVectorArithmetic.FP32,
        )
        app = Unfold16x16x192App(
            inst_path=unfold_bin,
            input_path=unfold_input_path,
            output_path=unfold_output_path,
        )
        app.run(max_cycles=20_000_000, state=state)

    return {"input_path": out_dir / "input_fp32.bin"}
