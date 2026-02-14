"""VLIW instruction decoder and legacy execution API.

This module provides:
- Instruction word decoding (decode_instruction_word)
- Binary instruction file loading (load_binary_instructions)
- Legacy execution functions for backward compatibility

The actual instruction execution is now handled by the Ipu class in ipu.py,
which uses automatic dispatch based on instruction_spec metadata.

Instruction format
~~~~~~~~~~~~~~~~~~
An instruction is stored as a ``dict[str, int]`` whose keys match the
C struct field names produced by ``CompoundInst.get_fields()``, e.g.::

    {
        "break_inst_token_0_break_inst_opcode": 2,   # break_nop
        "xmem_inst_token_0_xmem_inst_opcode": 4,     # xmem_nop
        ...
    }

These dicts are produced by :func:`decode_instruction_word`.
"""

from __future__ import annotations

# Re-export for convenience
from ipu_as.compound_inst import CompoundInst

# Re-export Ipu class and BreakResult for backward compatibility
from ipu_emu.ipu import Ipu, BreakResult
from ipu_emu.ipu_state import IpuState


# ---------------------------------------------------------------------------
# Instruction word decoder
# ---------------------------------------------------------------------------


def decode_instruction_word(word: int) -> dict[str, int]:
    """Decode a raw integer instruction word into a field dict.

    Uses ``CompoundInst.get_fields()`` which returns ``[(name, bits), ...]``
    ordered from MSB to LSB.  We iterate in reverse (LSB first) to extract
    each field.
    """
    fields = CompoundInst.get_fields()  # LSB-first list
    result: dict[str, int] = {}
    shift = 0
    for name, width in fields:
        mask = (1 << width) - 1
        result[name] = (word >> shift) & mask
        shift += width
    return result


def _instruction_aligned_bytes() -> int:
    """Number of bytes per instruction in binary format (32-bit word aligned).

    Must match ``instruction_aligned_bytes_len()`` in the assembler.
    """
    bits = CompoundInst.bits()
    word_size = 32
    if bits % word_size != 0:
        bits += word_size - (bits % word_size)
    return (bits // word_size) * 4


def load_binary_instructions(data: bytes) -> list[dict[str, int]]:
    """Load a binary file (output of ``ipu-as assemble --format bin``) into
    a list of decoded instruction dicts.

    Each instruction is word-aligned to 32-bit boundaries (matching the
    assembler's ``assemble_to_bin_file``).
    """
    inst_bytes = _instruction_aligned_bytes()
    instructions: list[dict[str, int]] = []
    for offset in range(0, len(data), inst_bytes):
        chunk = data[offset : offset + inst_bytes]
        if len(chunk) < inst_bytes:
            break
        word = int.from_bytes(chunk, byteorder="little")
        instructions.append(decode_instruction_word(word))
    return instructions


# ---------------------------------------------------------------------------
# Legacy execution API (backward compatibility wrappers)
# ---------------------------------------------------------------------------
# These functions maintain the old API but delegate to the Ipu class.
# New code should use the Ipu class directly.
# ---------------------------------------------------------------------------


def execute_next_instruction(state: IpuState) -> BreakResult:
    """Execute one VLIW cycle.

    Legacy wrapper — new code should use Ipu class directly.

    1. Fetch instruction at program counter
    2. Snapshot the register file
    3. Execute BREAK first (before side effects)
    4. Execute XMEM, LR, MULT, ACC, COND in parallel from the snapshot

    Returns:
        BreakResult.BREAK if break condition occurred, CONTINUE otherwise
    """
    ipu = Ipu(state)
    return ipu.execute_vliw_cycle()


def execute_instruction_skip_break(state: IpuState) -> None:
    """Execute the current instruction without re-checking break.

    Legacy wrapper — new code should use Ipu class directly.
    Used after returning from a debug break to complete the cycle.
    """
    ipu = Ipu(state)
    ipu.execute_vliw_cycle_skip_break()
