"""End-to-end tests for Phase 4: binary I/O, fully_connected app, and
the assemble→load→run→validate pipeline.

These tests exercise:
  * Binary load / dump round-trip through XMEM
  * FP32→FP8 conversion + load
  * Weight transpose logic in fully_connected harness
  * Full ``run_test`` orchestration
  * End-to-end: assemble ASM → run in emulator → compare output vs golden
"""

from __future__ import annotations

import os
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from ipu_emu.ipu_state import IpuState
from ipu_emu.ipu_math import DType, fp32_to_fp8_bytes, fp8_bytes_to_fp32
from ipu_emu.emulator import (
    load_binary_to_xmem,
    dump_xmem_to_binary,
    load_fp32_as_fp8_to_xmem,
    load_program,
    run_until_complete,
    run_test,
    DebugAction,
)
from ipu_emu.execute import decode_instruction_word
from ipu_emu.apps.fully_connected import (
    parse_dtype,
    setup_fully_connected,
    teardown_fully_connected,
    run_fully_connected,
    _load_and_transpose_weights,
    SAMPLES_NUM,
    INPUT_BASE_ADDR,
    INPUT_NEURONS,
    WEIGHTS_BASE_ADDR,
    OUTPUT_BASE_ADDR,
    OUTPUT_NEURONS,
)


@pytest.fixture(autouse=True)
def _reset_assembler_labels():
    """Reset the global label registry between tests."""
    from ipu_as.label import reset_labels
    reset_labels()
    yield
    reset_labels()


def _runfiles_dir() -> Path | None:
    """Return the Bazel runfiles root, or None if not running under Bazel."""
    srcdir = os.environ.get("TEST_SRCDIR")
    workspace = os.environ.get("TEST_WORKSPACE", "_main")
    if srcdir:
        return Path(srcdir) / workspace
    return None


def _resolve_path(*parts: str) -> Path:
    """Resolve a workspace-relative path, supporting both Bazel and uv/pytest.

    Under Bazel, files are found via runfiles.  Under uv/pytest, we resolve
    relative to the workspace root (4 parents up from this test file).
    """
    rf = _runfiles_dir()
    if rf:
        return rf.joinpath(*parts)
    workspace_root = Path(__file__).resolve().parents[4]
    return workspace_root.joinpath(*parts)


# Pre-assembled instruction binary (built by assemble_asm rule under Bazel,
# assembled at test time under uv/pytest).
_FC_INST_BIN = _resolve_path(
    "src", "tools", "ipu-emu-py", "assemble_fully_connected.bin"
)

_FC_DATA_DIR = _resolve_path(
    "src", "apps", "fully_connected", "test_data_format"
)

_FC_ASM_FILE = _resolve_path(
    "src", "apps", "fully_connected", "fully_connected.asm"
)

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
        """Convert FP32 → FP8 → FP32, check values are close."""
        fp32 = np.array([0.0, 1.0, -1.0, 0.5, 2.0], dtype=np.float32)
        raw = fp32_to_fp8_bytes(fp32, DType.FP8_E4M3)
        assert len(raw) == 5
        back = fp8_bytes_to_fp32(raw, DType.FP8_E4M3)
        np.testing.assert_allclose(back, fp32, rtol=0.1, atol=0.1)

    def test_fp32_to_fp8_e5m2_round_trip(self) -> None:
        fp32 = np.array([0.0, 1.0, -1.0, 0.25], dtype=np.float32)
        raw = fp32_to_fp8_bytes(fp32, DType.FP8_E5M2)
        assert len(raw) == 4
        back = fp8_bytes_to_fp32(raw, DType.FP8_E5M2)
        np.testing.assert_allclose(back, fp32, rtol=0.2, atol=0.1)

    def test_load_fp32_as_fp8_to_xmem(self, tmp_path: Path) -> None:
        """Full pipeline: save FP32 to file → load as FP8 → verify in XMEM."""
        fp32 = np.arange(128, dtype=np.float32)
        f = tmp_path / "fp32.bin"
        f.write_bytes(fp32.tobytes())

        state = IpuState()
        count = load_fp32_as_fp8_to_xmem(state, f, 0x5000, DType.FP8_E4M3)
        assert count == 128

        # Read back and verify
        raw = state.xmem.read_address(0x5000, 128)
        back = fp8_bytes_to_fp32(bytes(raw), DType.FP8_E4M3)
        # FP8 E4M3 saturates above ~448, so check representable range
        np.testing.assert_allclose(back[:10], fp32[:10], rtol=0.1, atol=0.5)

    def test_int8_dtype_rejected(self) -> None:
        """INT8 is not valid for FP8 conversion."""
        fp32 = np.array([1.0], dtype=np.float32)
        with pytest.raises(ValueError, match="FP8"):
            fp32_to_fp8_bytes(fp32, DType.INT8)


# ---------------------------------------------------------------------------
# Fully-connected app harness tests
# ---------------------------------------------------------------------------


class TestParseDtype:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("INT8", DType.INT8),
            ("int8", DType.INT8),
            ("FP8_E4M3", DType.FP8_E4M3),
            ("fp8_e4m3", DType.FP8_E4M3),
            ("FP8_E5M2", DType.FP8_E5M2),
            ("fp8_e5m2", DType.FP8_E5M2),
        ],
    )
    def test_valid(self, name: str, expected: DType) -> None:
        assert parse_dtype(name) == expected

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid dtype"):
            parse_dtype("FLOAT16")


class TestWeightTranspose:
    """Test the weight transpose logic."""

    def test_transpose_identity(self) -> None:
        """Build a known matrix, transpose, verify layout in XMEM."""
        state = IpuState()

        # Create weights where w[j][i] = j * INPUT_NEURONS + i
        weights = bytearray(OUTPUT_NEURONS * INPUT_NEURONS)
        for j in range(OUTPUT_NEURONS):
            for i in range(INPUT_NEURONS):
                weights[j * INPUT_NEURONS + i] = (j + i) & 0xFF

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(weights)
            f.flush()
            _load_and_transpose_weights(state, f.name)

        # Verify: transposed_vector[i][j] should equal original[j][i]
        for i in range(INPUT_NEURONS):
            row_data = state.xmem.read_address(
                WEIGHTS_BASE_ADDR + i * INPUT_NEURONS, INPUT_NEURONS
            )
            for j in range(OUTPUT_NEURONS):
                expected = (j + i) & 0xFF
                assert row_data[j] == expected, f"Mismatch at [{i}][{j}]"
            # Padding should be zero
            for k in range(OUTPUT_NEURONS, INPUT_NEURONS):
                assert row_data[k] == 0, f"Non-zero padding at [{i}][{k}]"


class TestSetupFullyConnected:
    """Test the setup callback in isolation."""

    def test_cr_registers_set(self, tmp_path: Path) -> None:
        """Verify CR registers are set correctly after setup."""
        # Create dummy input and weight files
        inputs = bytes(range(256)) * (SAMPLES_NUM * INPUT_NEURONS // 256 + 1)
        inputs = inputs[: SAMPLES_NUM * INPUT_NEURONS]
        (tmp_path / "inputs.bin").write_bytes(inputs)

        weights = bytes([0]) * (OUTPUT_NEURONS * INPUT_NEURONS)
        (tmp_path / "weights.bin").write_bytes(weights)

        state = IpuState()
        setup_fully_connected(
            state,
            tmp_path / "inputs.bin",
            tmp_path / "weights.bin",
            DType.INT8,
        )

        assert state.regfile.get_cr(0) == INPUT_BASE_ADDR
        assert state.regfile.get_cr(1) == WEIGHTS_BASE_ADDR
        assert state.regfile.get_cr(2) == OUTPUT_BASE_ADDR
        assert state.get_cr_dtype() == DType.INT8

    def test_inputs_loaded(self, tmp_path: Path) -> None:
        """Verify inputs are in XMEM after setup."""
        inputs = bytes([0x42]) * (SAMPLES_NUM * INPUT_NEURONS)
        (tmp_path / "inputs.bin").write_bytes(inputs)
        weights = bytes([0]) * (OUTPUT_NEURONS * INPUT_NEURONS)
        (tmp_path / "weights.bin").write_bytes(weights)

        state = IpuState()
        setup_fully_connected(
            state, tmp_path / "inputs.bin", tmp_path / "weights.bin", DType.INT8
        )

        first_row = state.xmem.read_address(INPUT_BASE_ADDR, INPUT_NEURONS)
        assert all(b == 0x42 for b in first_row)


class TestTeardownFullyConnected:
    """Test the teardown callback in isolation."""

    def test_output_dumped(self, tmp_path: Path) -> None:
        """Write known data to output area, verify it's dumped."""
        state = IpuState()
        # Write a known pattern to the output area
        chunk_size = OUTPUT_NEURONS * 4
        for s in range(SAMPLES_NUM):
            addr = OUTPUT_BASE_ADDR + s * chunk_size
            data = struct.pack(f"<{OUTPUT_NEURONS}i", *range(s * OUTPUT_NEURONS, (s + 1) * OUTPUT_NEURONS))
            state.xmem.write_address(addr, data)

        out = tmp_path / "output.bin"
        teardown_fully_connected(state, out)

        raw = out.read_bytes()
        assert len(raw) == SAMPLES_NUM * chunk_size


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
    """Full end-to-end tests using actual assembly and test data."""

    def _get_fc_inst_bin(self, tmp_path: Path) -> Path:
        """Return path to the assembled FC instruction binary.

        Under Bazel the binary is pre-built by the ``assemble_asm`` rule.
        Under uv/pytest we assemble it on the fly.
        """
        if _FC_INST_BIN.exists():
            return _FC_INST_BIN

        # Fallback: assemble at test time (uv / pytest)
        if not _FC_ASM_FILE.exists():
            pytest.skip(f"ASM file not found: {_FC_ASM_FILE}")

        from ipu_as.lark_tree import assemble_to_bin_file
        out = tmp_path / "fc.bin"
        assemble_to_bin_file(_FC_ASM_FILE.read_text(), str(out))
        return out

    def test_assemble_fc_program(self, tmp_path: Path) -> None:
        """Verify the FC assembly file assembles without error."""
        inst = self._get_fc_inst_bin(tmp_path)
        assert inst.stat().st_size > 0

    @pytest.mark.parametrize("dtype_dir,dtype_str", [
        ("int8", "INT8"),
    ])
    def test_fc_end_to_end(
        self, tmp_path: Path, dtype_dir: str, dtype_str: str
    ) -> None:
        """Assemble → setup → run → compare output against golden.

        Uses actual test data from the C test suite.
        """
        data_dir = _FC_DATA_DIR / dtype_dir
        if not data_dir.exists():
            pytest.skip(f"Test data not found: {data_dir}")

        inst_path = self._get_fc_inst_bin(tmp_path)

        # Locate data files
        inputs_path = data_dir / f"inputs_{dtype_dir}.bin"
        weights_path = data_dir / f"weights_{dtype_dir}.bin"
        golden_path = data_dir / f"out_{dtype_dir}_acc_int32.bin"

        if not inputs_path.exists() or not weights_path.exists():
            pytest.skip(f"Missing data files in {data_dir}")

        # Run
        output_path = tmp_path / "output.bin"
        state, cycles = run_fully_connected(
            inst_path=inst_path,
            inputs_path=inputs_path,
            weights_path=weights_path,
            output_path=output_path,
            dtype=dtype_str,
            max_cycles=2_000_000,
        )

        assert cycles > 0, "Program did not execute any cycles"

        # Compare with golden if available
        if golden_path.exists():
            expected = golden_path.read_bytes()
            actual = output_path.read_bytes()
            assert len(actual) == len(expected), (
                f"Output size mismatch: {len(actual)} vs {len(expected)}"
            )
            assert actual == expected, "Output does not match golden reference"

    @pytest.mark.parametrize("dtype_dir,dtype_str,golden_suffix", [
        ("fp8_e4m3", "FP8_E4M3", "out_fp8_e4m3_acc_fp32.bin"),
        ("fp8_e5m2", "FP8_E5M2", "out_fp8_e5m2_acc_fp32.bin"),
    ])
    def test_fc_end_to_end_fp8(
        self, tmp_path: Path, dtype_dir: str, dtype_str: str, golden_suffix: str
    ) -> None:
        """End-to-end test for FP8 data types."""
        data_dir = _FC_DATA_DIR / dtype_dir
        if not data_dir.exists():
            pytest.skip(f"Test data not found: {data_dir}")

        inst_path = self._get_fc_inst_bin(tmp_path)

        inputs_path = data_dir / f"inputs_{dtype_dir}.bin"
        weights_path = data_dir / f"weights_{dtype_dir}.bin"
        golden_path = data_dir / golden_suffix

        if not inputs_path.exists() or not weights_path.exists():
            pytest.skip(f"Missing data files in {data_dir}")

        output_path = tmp_path / "output.bin"
        state, cycles = run_fully_connected(
            inst_path=inst_path,
            inputs_path=inputs_path,
            weights_path=weights_path,
            output_path=output_path,
            dtype=dtype_str,
            max_cycles=2_000_000,
        )

        assert cycles > 0

        if golden_path.exists():
            expected = golden_path.read_bytes()
            actual = output_path.read_bytes()
            assert len(actual) == len(expected), (
                f"Output size mismatch: {len(actual)} vs {len(expected)}"
            )
            # For FP8, compare as float32 arrays with tolerance
            expected_arr = np.frombuffer(expected, dtype=np.float32)
            actual_arr = np.frombuffer(actual, dtype=np.float32)
            np.testing.assert_allclose(
                actual_arr, expected_arr, rtol=1e-3, atol=1e-3,
                err_msg="FP8 output does not match golden reference",
            )

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
