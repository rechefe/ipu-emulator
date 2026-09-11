"""Drive the ETISS-based IPU simulator from Python.

The ETISS backend runs out of process: this module serialises an
:class:`~ipu_emu.ipu_state.IpuState` into the runner's state blob, hands it the
program image and XMEM, and reads the result back into the same ``IpuState``,
so callers (``run_test``, ``IpuApp``, the test suite) see a state object that is
indistinguishable from one the Python emulator produced.

The address map, word geometry and blob layout all come from
:mod:`ipu_as.gen_etiss`, which is also what generated the C side — there is no
second copy of any of it here.

Wide-vector debug mode is supported: the flags travel in the state blob and the
backend switches to 4-byte lanes exactly as the Python emulator does.
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ipu_as.compound_inst import CompoundInst
from ipu_as import gen_etiss

from ipu_emu.errors import EmulatorError
from ipu_emu.ipu_state import INST_MEM_SIZE, IpuState, WideVectorArithmetic
from ipu_emu.ipu_math import DType
from ipu_emu.xmem import XMEM_SIZE_BYTES

#: Environment variable naming the ``ipu_etiss_run`` binary.
RUNNER_ENV = "IPU_ETISS_RUN"

WORD_BYTES = gen_etiss.instruction_aligned_bytes()
_STATE_FIELDS = gen_etiss.state_field_list()

_SCALAR_FMT = {"U32": "<I", "U64": "<Q", "F64": "<d"}

#: wide_vector_arithmetic <-> the integer the state blob carries.
_WIDE_ARITH_TO_INT = {WideVectorArithmetic.FP32: 0, WideVectorArithmetic.INT32: 1}
_INT_TO_WIDE_ARITH = {v: k for k, v in _WIDE_ARITH_TO_INT.items()}

#: Blob field name -> the register file blob it maps to. Scalars and stats are
#: handled separately; everything else is a raw register image.
_BLOB_REGISTERS = {
    name: name.lower()
    for name, kind, _count in _STATE_FIELDS
    if kind in ("BLOB", "U32ARR")
}


class EtissNotAvailable(RuntimeError):
    """Raised when the ETISS runner binary cannot be located."""


def runner_path() -> Path:
    """Locate ``ipu_etiss_run``.

    Looks at ``$IPU_ETISS_RUN`` first, then falls back to ``PATH``.
    """
    explicit = os.environ.get(RUNNER_ENV)
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise EtissNotAvailable(f"{RUNNER_ENV}={explicit} does not exist")
        return path
    found = shutil.which("ipu_etiss_run")
    if found:
        return Path(found)
    raise EtissNotAvailable(
        "ipu_etiss_run not found; build src/tools/ipu-etiss and set "
        f"{RUNNER_ENV} to the built binary"
    )


def is_available() -> bool:
    """True when the ETISS backend can be used in this environment."""
    try:
        runner_path()
    except EtissNotAvailable:
        return False
    return True


# ---------------------------------------------------------------------------
# Instruction image
# ---------------------------------------------------------------------------

def _nop_word() -> int:
    """The encoding of a compound instruction with every slot NOP."""
    from ipu_as.lark_tree import assemble

    return assemble("NOP;;")[0]


_NOP_WORD_CACHE: int | None = None


def encode_instruction_word(fields: dict[str, int]) -> int:
    """Inverse of :func:`ipu_emu.execute.decode_instruction_word`."""
    word = 0
    shift = 0
    for name, width in CompoundInst.get_fields():
        word |= (fields[name] & ((1 << width) - 1)) << shift
        shift += width
    return word


def build_imem_image(state: IpuState) -> bytes:
    """Serialise ``state.inst_mem`` into the assembler's ``--format bin`` layout.

    Unprogrammed slots become the all-NOP word.  The Python emulator treats a
    ``None`` entry as "NOP, advance the PC", and an encoded all-NOP word does
    exactly the same thing on both backends -- including not bumping any
    ``RunStats`` counter -- so the padded image runs off the end of the program
    with the same cycle count.
    """
    global _NOP_WORD_CACHE
    if _NOP_WORD_CACHE is None:
        _NOP_WORD_CACHE = _nop_word()

    out = bytearray()
    for entry in state.inst_mem[:INST_MEM_SIZE]:
        word = _NOP_WORD_CACHE if entry is None else encode_instruction_word(entry)
        out += int(word).to_bytes(WORD_BYTES, "little")
    return bytes(out)


# ---------------------------------------------------------------------------
# State blob
# ---------------------------------------------------------------------------

def serialize_state(state: IpuState, *, max_cycles: int, break_mode: int) -> bytes:
    """Pack an :class:`IpuState` into the runner's input blob."""
    values: dict[str, Any] = {
        "dtype": int(state.dtype),
        "elu_alpha": float(state.elu_alpha),
        "break_mode": int(break_mode),
        "max_cycles": int(max_cycles),
        "wide_vector_debug": int(bool(state.wide_vector_debug)),
        "wide_vector_arithmetic": _WIDE_ARITH_TO_INT[
            WideVectorArithmetic(state.wide_vector_arithmetic)
        ],
        "wide_vector_quantize_output": int(bool(state.wide_vector_quantize_output)),
        "cycles": 0,
        "mult_active_cycles": 0,
        "acc_active_cycles": 0,
        "xmem_reads": 0,
        "xmem_writes": 0,
        "error_code": 0,
        "error_detail": 0,
    }

    out = bytearray(gen_etiss.STATE_MAGIC)
    out += struct.pack("<I", gen_etiss.STATE_VERSION)
    out += struct.pack("<Q", int(state.program_counter))
    for name, kind, count in _STATE_FIELDS:
        if kind in _SCALAR_FMT:
            out += struct.pack(_SCALAR_FMT[kind], values[name])
        else:
            raw = state.regfile.raw(_BLOB_REGISTERS[name])
            expected = count * (4 if kind == "U32ARR" else 1)
            if len(raw) != expected:
                raise EmulatorError(
                    f"register {_BLOB_REGISTERS[name]!r} is {len(raw)} bytes, "
                    f"the ETISS state blob expects {expected}"
                )
            out += bytes(raw)
    return bytes(out)


def deserialize_state(state: IpuState, blob: bytes) -> dict[str, int]:
    """Write the runner's output blob back into *state*; return its stats."""
    if not blob.startswith(gen_etiss.STATE_MAGIC):
        raise EmulatorError("ETISS runner produced a state blob with a bad magic")
    pos = len(gen_etiss.STATE_MAGIC)
    (version,) = struct.unpack_from("<I", blob, pos)
    pos += 4
    if version != gen_etiss.STATE_VERSION:
        raise EmulatorError(
            f"ETISS state blob version {version}, expected {gen_etiss.STATE_VERSION}"
        )
    (pc,) = struct.unpack_from("<Q", blob, pos)
    pos += 8
    state.program_counter = int(pc)

    result: dict[str, int] = {}
    for name, kind, count in _STATE_FIELDS:
        if kind in _SCALAR_FMT:
            fmt = _SCALAR_FMT[kind]
            (value,) = struct.unpack_from(fmt, blob, pos)
            pos += struct.calcsize(fmt)
            result[name] = value
        else:
            size = count * (4 if kind == "U32ARR" else 1)
            state.regfile.raw(_BLOB_REGISTERS[name])[:] = blob[pos:pos + size]
            pos += size

    state.dtype = DType(result["dtype"])
    state.elu_alpha = result["elu_alpha"]
    state.wide_vector_debug = bool(result["wide_vector_debug"])
    state.wide_vector_arithmetic = _INT_TO_WIDE_ARITH[result["wide_vector_arithmetic"]]
    state.wide_vector_quantize_output = bool(result["wide_vector_quantize_output"])
    state.stats.total_cycles = result["cycles"]
    state.stats.mult_active_cycles = result["mult_active_cycles"]
    state.stats.acc_active_cycles = result["acc_active_cycles"]
    state.stats.xmem_reads = result["xmem_reads"]
    state.stats.xmem_writes = result["xmem_writes"]
    return result


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

class EtissRunner:
    """Runs one program on the ETISS backend and applies the result to a state."""

    def __init__(self, *, jit: str = "tcc", keep_workdir: bool = False):
        self.jit = jit
        self.keep_workdir = keep_workdir

    def run(self, state: IpuState, *, max_cycles: int = 100_000, break_mode: int = 0) -> int:
        """Execute ``state``'s loaded program; update it in place, return cycles."""
        binary = runner_path()
        workdir = Path(tempfile.mkdtemp(prefix="ipu-etiss-"))
        try:
            imem = workdir / "imem.bin"
            xmem_in = workdir / "xmem_in.bin"
            xmem_out = workdir / "xmem_out.bin"
            state_in = workdir / "state_in.bin"
            state_out = workdir / "state_out.bin"

            imem.write_bytes(build_imem_image(state))
            xmem_in.write_bytes(bytes(state.xmem.read_address(0, XMEM_SIZE_BYTES)))
            state_in.write_bytes(
                serialize_state(state, max_cycles=max_cycles, break_mode=break_mode)
            )

            proc = subprocess.run(
                [
                    str(binary),
                    "--imem", str(imem),
                    "--state-in", str(state_in),
                    "--state-out", str(state_out),
                    "--xmem-in", str(xmem_in),
                    "--xmem-out", str(xmem_out),
                    "--jit", self.jit,
                ],
                capture_output=True,
                text=True,
            )
            if proc.returncode not in (0, 2) or not state_out.exists():
                raise EmulatorError(
                    f"ipu_etiss_run failed (exit {proc.returncode}):\n"
                    f"{proc.stdout}\n{proc.stderr}"
                )

            stats = deserialize_state(state, state_out.read_bytes())
            state.xmem.write_address(0, xmem_out.read_bytes())

            error_code = stats["error_code"]
            if error_code:
                name = gen_etiss.ERROR_NAMES.get(error_code, f"code {error_code}")
                raise EmulatorError(
                    f"IPU error {name} (detail {stats['error_detail']}) "
                    f"after {stats['cycles']} cycles"
                )
            return int(stats["cycles"])
        finally:
            if not self.keep_workdir:
                shutil.rmtree(workdir, ignore_errors=True)
