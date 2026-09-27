"""Pooling family tests: routing between maxpool2d_stride2 and its tail variant."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from ipu_emu.ipu import LANES
from ipu_apps.kernel_registry import resolve
from ipu_apps.kernel_registry.cases import assemble_kernel


def test_registry_resolve_and_run_across_tile_widths() -> None:
    """Exactly one of maxpool2d_stride2 / maxpool2d_stride2_tail takes each
    width: resolve, assemble, run, and verify compactly tiled input widths."""
    shapes = [
        (1, 2, 2),
        (1, 2, 127),
        (1, 2, 128),
        (1, 2, 129),
        (1, 2, 255),
        (1, 2, 256),
        (1, 2, 257),
        (1, 2, 258),
        (2, 4, 260),
        (1, 2, 511),
        (1, 2, 512),
        (64, 2, 511),
        (64, 2, 512),
    ]
    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        assembled: dict[str, Path] = {}

        for case, shape in enumerate(shapes):
            channels, height, width = shape
            out_height, out_width = height // 2, width // 2
            input_tiles = (width + LANES - 1) // LANES
            out_tiles = (out_width + LANES - 1) // LANES
            verdict = resolve(
                "maxpool2d",
                shape=shape,
                kernel_size=2,
                stride=2,
                padding=0,
            )
            assert verdict.supported
            expected_name = (
                "maxpool2d_stride2_tail"
                if input_tiles < 2 * out_tiles
                else "maxpool2d_stride2"
            )
            assert verdict.app_name == expected_name

            if verdict.app_name not in assembled:
                assembled[verdict.app_name] = assemble_kernel(verdict.app_name, tmp_path)
            inst_path = assembled[verdict.app_name]

            logical_input = rng.standard_normal(shape, dtype=np.float32)
            tiled_input = np.full(
                (channels, height, input_tiles * LANES),
                np.finfo(np.float32).min,
                dtype="<f4",
            )
            tiled_input[:, :, :width] = logical_input
            trimmed = logical_input[:, : 2 * out_height, : 2 * out_width]
            expected = trimmed.reshape(
                channels, out_height, 2, out_width, 2
            ).max(axis=(2, 4))

            input_path = tmp_path / f"input-{case}.bin"
            output_path = tmp_path / f"output-{case}.bin"
            input_path.write_bytes(tiled_input.tobytes())
            app = verdict.kernel.app_class(
                inst_path=inst_path,
                input_path=input_path,
                output_path=output_path,
                **verdict.kwargs,
            )
            assert app.layout.crs[5] == input_tiles
            assert app.layout.input_rows == channels * height * input_tiles
            state, cycles = app.run(max_cycles=2_000_000)
            raw_output = np.frombuffer(output_path.read_bytes(), dtype="<f4")
            tiled_output = raw_output.reshape(
                channels, out_height, out_tiles * LANES
            )

            assert cycles > 0
            expected_cr12 = out_tiles - 1 if expected_name.endswith("_tail") else 0
            assert state.regfile.get_cr(12) == expected_cr12
            reads_per_output_row = (
                6 * (out_tiles - 1) + 3
                if expected_name == "maxpool2d_stride2_tail"
                else 6 * out_tiles
            )
            assert state.stats.xmem_reads == channels * out_height * reads_per_output_row
            assert np.array_equal(tiled_output[:, :, :out_width], expected)

