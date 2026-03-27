"""End-to-end tests for emulator core: binary I/O, FP8 conversion,
and the run_test orchestration pipeline.

These tests exercise:
  * Binary load / dump round-trip through XMEM
  * FP32→FP8 conversion + load
  * Full ``run_test`` orchestration
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ipu_emu.ipu_state import IpuState
from ipu_emu.ipu_math import DType, fp32_to_fp8_bytes, fp8_bytes_to_fp32
from ipu_emu.emulator import (
    load_binary_to_xmem,
    dump_xmem_to_binary,
    load_fp32_as_fp8_to_xmem,
    run_test,
    DebugAction,
)


@pytest.fixture(autouse=True)
def _reset_assembler_labels():
    """Reset the global label registry between tests."""
    from ipu_as.label import reset_labels
    reset_labels()
    yield
    reset_labels()


# ---------------------------------------------------------------------------
# Binary I/O unit tests
# ---------------------------------------------------------------------------


class TestLoadBinaryToXmem:
    """Tests for ``load_binary_to_xmem``."""

    def test_load_exact_chunks(self, tmp_path: Path) -> None:
        """Load a file that is an exact multiple of chunk_size."""
        chunk = bytes(range(128))
        data = chunk * 4  # 4 chunks
        f = tmp_path / "data.bin"
        f.write_bytes(data)

        state = IpuState()
        loaded = load_binary_to_xmem(state, f, 0x1000, 128)
        assert loaded == 4

        for i in range(4):
            got = state.xmem.read_address(0x1000 + i * 128, 128)
            assert bytes(got) == chunk

    def test_max_chunks_limit(self, tmp_path: Path) -> None:
        """Respect *max_chunks* even if more data is available."""
        f = tmp_path / "data.bin"
        f.write_bytes(b"\xAA" * 512)

        state = IpuState()
        loaded = load_binary_to_xmem(state, f, 0, 128, max_chunks=2)
        assert loaded == 2

    def test_partial_trailing_data_ignored(self, tmp_path: Path) -> None:
        """Trailing bytes smaller than chunk_size are ignored."""
        f = tmp_path / "data.bin"
        f.write_bytes(b"\xBB" * (128 * 3 + 50))

        state = IpuState()
        loaded = load_binary_to_xmem(state, f, 0, 128)
        assert loaded == 3

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")

        state = IpuState()
        loaded = load_binary_to_xmem(state, f, 0, 128)
        assert loaded == 0


class TestDumpXmemToBinary:
    """Tests for ``dump_xmem_to_binary``."""

    def test_round_trip(self, tmp_path: Path) -> None:
        """Write data to XMEM, dump, verify binary matches."""
        state = IpuState()
        pattern = bytes(range(256)) * 2  # 512 bytes = 4 chunks of 128
        for i in range(4):
            state.xmem.write_address(
                0x2000 + i * 128, pattern[i * 128 : (i + 1) * 128]
            )

        out = tmp_path / "dump.bin"
        written = dump_xmem_to_binary(state, out, 0x2000, 128, 4)
        assert written == 4
        assert out.read_bytes() == pattern

    def test_load_dump_round_trip(self, tmp_path: Path) -> None:
        """Load binary → dump binary → files should be identical."""
        original = bytes(range(128)) * 5
        src = tmp_path / "src.bin"
        src.write_bytes(original)

        state = IpuState()
        load_binary_to_xmem(state, src, 0, 128, max_chunks=5)

        dst = tmp_path / "dst.bin"
        dump_xmem_to_binary(state, dst, 0, 128, 5)
        assert dst.read_bytes() == original


class TestFP32ToFP8Loading:
    """Tests for FP32→FP8 conversion and loading."""

    def test_fp32_to_fp8_e4m3_round_trip(self) -> None:
        """Convert FP32 → fp8_e4 → FP32, check values are close."""
        fp32 = np.array([0.0, 1.0, -1.0, 0.5, 2.0], dtype=np.float32)
        raw = fp32_to_fp8_bytes(fp32, DType.E4)
        assert len(raw) == 5
        back = fp8_bytes_to_fp32(raw, DType.E4)
        np.testing.assert_allclose(back, fp32, rtol=0.1, atol=0.1)

    def test_fp32_to_fp8_e5m2_round_trip(self) -> None:
        fp32 = np.array([0.0, 1.0, -1.0, 0.25], dtype=np.float32)
        raw = fp32_to_fp8_bytes(fp32, DType.E5)
        assert len(raw) == 4
        back = fp8_bytes_to_fp32(raw, DType.E5)
        np.testing.assert_allclose(back, fp32, rtol=0.2, atol=0.1)

    def test_load_fp32_as_fp8_to_xmem(self, tmp_path: Path) -> None:
        """Full pipeline: save FP32 to file → load as FP8 → verify in XMEM."""
        fp32 = np.arange(128, dtype=np.float32)
        f = tmp_path / "fp32.bin"
        f.write_bytes(fp32.tobytes())

        state = IpuState()
        count = load_fp32_as_fp8_to_xmem(state, f, 0x5000, DType.E4)
        assert count == 128

        # Read back and verify
        raw = state.xmem.read_address(0x5000, 128)
        back = fp8_bytes_to_fp32(bytes(raw), DType.E4)
        # fp8_e4 saturates above ~448, so check representable range
        np.testing.assert_allclose(back[:10], fp32[:10], rtol=0.1, atol=0.5)

    def test_int8_dtype_rejected(self) -> None:
        """INT8 is not valid for FP8 conversion."""
        fp32 = np.array([1.0], dtype=np.float32)
        with pytest.raises(ValueError, match="FP8"):
            fp32_to_fp8_bytes(fp32, DType.INT8)


# ---------------------------------------------------------------------------
# run_test orchestration tests
# ---------------------------------------------------------------------------


class TestRunTest:
    """Test the run_test high-level orchestrator."""

    def test_run_test_with_bkpt_program(self, tmp_path: Path) -> None:
        """A program that immediately breaks should run in 1 cycle."""
        from ipu_as.lark_tree import assemble_to_bin_file

        asm = "bkpt;;\n"
        inst_path = tmp_path / "bkpt.bin"
        assemble_to_bin_file(asm, str(inst_path))

        setup_called = [False]
        teardown_called = [False]

        def my_setup(state: IpuState) -> None:
            setup_called[0] = True

        def my_teardown(state: IpuState) -> None:
            teardown_called[0] = True

        state, cycles = run_test(
            inst_path=inst_path,
            setup=my_setup,
            teardown=my_teardown,
        )

        assert setup_called[0]
        assert teardown_called[0]
        # bkpt in run_until_complete skips the break, advances PC
        assert cycles >= 1

    def test_run_test_with_debug_callback(self, tmp_path: Path) -> None:
        """run_test with debug_callback should invoke it on break."""
        from ipu_as.lark_tree import assemble_to_bin_file

        # Use a BREAK instruction (break slot) followed by bkpt (cond halt)
        asm = "break;;\nbkpt;;\n"
        inst_path = tmp_path / "break.bin"
        assemble_to_bin_file(asm, str(inst_path))

        cb_calls = []

        def my_cb(state: IpuState, cycle: int) -> DebugAction:
            cb_calls.append(cycle)
            return DebugAction.QUIT

        state, cycles = run_test(
            inst_path=inst_path,
            debug_callback=my_cb,
        )
        assert len(cb_calls) > 0


# ---------------------------------------------------------------------------
# End-to-end: assemble → load → run → validate
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Simple end-to-end tests using the emulator core."""

    def test_simple_program_end_to_end(self, tmp_path: Path) -> None:
        """Assemble a trivial program, run it, verify register state."""
        from ipu_as.lark_tree import assemble_to_bin_file

        # Simple program: set lr0 = 42, then breakpoint
        asm = "set lr0 42;;\nbkpt;;\n"
        inst_path = tmp_path / "simple.bin"
        assemble_to_bin_file(asm, str(inst_path))

        state, cycles = run_test(inst_path=inst_path)

        assert state.regfile.get_lr(0) == 42
        assert cycles >= 2  # set + bkpt
