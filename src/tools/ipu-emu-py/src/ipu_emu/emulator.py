"""High-level emulator runner — Python equivalent of emulator.c.

Provides two modes:
  * ``run_until_complete`` — runs the program silently, ignoring break-
    points, until the PC falls off the end of instruction memory.
  * ``run_with_debug``   — honours breakpoints.  On break, invokes a
    user-supplied callback that decides whether to step, continue, or
    quit.

Both functions operate on an :class:`IpuState` that has already been
loaded with an assembled program (via :func:`load_program`).
"""

from __future__ import annotations

from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ipu_emu.ipu_state import IpuState, INST_MEM_SIZE
from ipu_emu.ipu_math import DType, fp32_to_fp8_bytes, fp8_bytes_to_fp32
from ipu_emu.execute import (
    BreakResult,
    decode_instruction_word,
    execute_next_instruction,
    execute_instruction_skip_break,
    load_binary_instructions,
)


class DebugAction(Enum):
    """Action returned by a debug callback on breakpoint."""
    CONTINUE = auto()
    STEP = auto()
    QUIT = auto()


# Type alias for the debug callback.
# Receives (state, cycle_count) and returns a DebugAction.
DebugCallback = Callable[[IpuState, int], DebugAction]


def load_program(state: IpuState, instructions: list[dict[str, int]]) -> None:
    """Load decoded instruction dicts into the state's instruction memory."""
    if len(instructions) > INST_MEM_SIZE:
        raise ValueError(
            f"Program too large: {len(instructions)} instructions "
            f"(max {INST_MEM_SIZE})"
        )
    for i, inst in enumerate(instructions):
        state.inst_mem[i] = inst


def load_program_from_binary(state: IpuState, path: str | Path) -> None:
    """Load a binary file produced by ``ipu-as assemble --format bin``."""
    data = Path(path).read_bytes()
    instructions = load_binary_instructions(data)
    load_program(state, instructions)


def run_until_complete(state: IpuState, max_cycles: int = 100_000) -> int:
    """Run the program ignoring breakpoints until it halts.

    Returns the number of cycles executed.

    Raises ``RuntimeError`` if *max_cycles* is exceeded (likely infinite loop).
    """
    cycles = 0
    while not state.is_halted:
        if cycles >= max_cycles:
            raise RuntimeError(
                f"Exceeded {max_cycles} cycles — possible infinite loop "
                f"(PC={state.program_counter})"
            )
        result = execute_next_instruction(state)
        if result == BreakResult.BREAK:
            # In "run" mode, skip the break and execute the instruction anyway
            execute_instruction_skip_break(state)
        cycles += 1
    return cycles


def run_with_debug(
    state: IpuState,
    debug_callback: DebugCallback,
    max_cycles: int = 100_000,
) -> int:
    """Run the program honouring breakpoints.

    When a breakpoint fires, *debug_callback(state, cycle_count)* is called.
    The callback returns a :class:`DebugAction`:

    - ``CONTINUE`` — resume execution (ignore further breaks until the next
      explicit breakpoint).
    - ``STEP`` — execute the current instruction, then break again.
    - ``QUIT`` — halt immediately.

    Returns the total number of cycles executed.
    """
    cycles = 0
    stepping = False

    while not state.is_halted:
        if cycles >= max_cycles:
            raise RuntimeError(
                f"Exceeded {max_cycles} cycles — possible infinite loop "
                f"(PC={state.program_counter})"
            )

        result = execute_next_instruction(state)

        if result == BreakResult.BREAK or stepping:
            action = debug_callback(state, cycles)
            if action == DebugAction.QUIT:
                break
            stepping = action == DebugAction.STEP
            # Execute the instruction (skipping the break re-check)
            if result == BreakResult.BREAK:
                execute_instruction_skip_break(state)

        cycles += 1

    return cycles


# ---------------------------------------------------------------------------
# Binary I/O — mirrors emulator__load_binary_to_xmem / dump_xmem_to_binary
# ---------------------------------------------------------------------------


def load_binary_to_xmem(
    state: IpuState,
    path: str | Path,
    base_addr: int,
    chunk_size: int,
    max_chunks: int = 0,
) -> int:
    """Load a binary file into XMEM in chunks.

    Reads *chunk_size* bytes at a time from *path* and writes them
    sequentially into XMEM starting at *base_addr*.

    Args:
        state:      The IPU state (provides ``xmem``).
        path:       Path to the binary file.
        base_addr:  Starting XMEM address.
        chunk_size: Bytes per chunk (typically 128 for one R-register row).
        max_chunks: Stop after this many chunks (0 = read until EOF).

    Returns:
        Number of chunks loaded.
    """
    data = Path(path).read_bytes()
    addr = base_addr
    chunks_loaded = 0
    offset = 0

    while offset + chunk_size <= len(data):
        state.xmem.write_address(addr, data[offset : offset + chunk_size])
        addr += chunk_size
        offset += chunk_size
        chunks_loaded += 1
        if max_chunks > 0 and chunks_loaded >= max_chunks:
            break

    return chunks_loaded


def dump_xmem_to_binary(
    state: IpuState,
    path: str | Path,
    base_addr: int,
    chunk_size: int,
    num_chunks: int,
) -> int:
    """Dump XMEM contents to a binary file in chunks.

    Args:
        state:      The IPU state (provides ``xmem``).
        path:       Output file path.
        base_addr:  Starting XMEM address.
        chunk_size: Bytes per chunk.
        num_chunks: Number of chunks to write.

    Returns:
        Number of chunks written.
    """
    parts: list[bytes] = []
    addr = base_addr
    for _ in range(num_chunks):
        parts.append(bytes(state.xmem.read_address(addr, chunk_size)))
        addr += chunk_size

    Path(path).write_bytes(b"".join(parts))
    return num_chunks


def load_fp32_as_fp8_to_xmem(
    state: IpuState,
    path: str | Path,
    base_addr: int,
    dtype: DType = DType.E4,
) -> int:
    """Load a FP32 binary file, convert to FP8, and store in XMEM.

    Reads N×4-byte float32 values from *path*, converts each to a single
    FP8 byte using ``ml_dtypes``, and writes the result contiguously into
    XMEM starting at *base_addr*.

    Args:
        state:     The IPU state (provides ``xmem``).
        path:      Path to a raw FP32 binary file.
        base_addr: Starting XMEM address.
        dtype:     Target FP8 variant (E4M3 or E5M2).

    Returns:
        Number of FP32 values converted.
    """
    raw = Path(path).read_bytes()
    fp32_values = np.frombuffer(raw, dtype=np.float32)
    fp8_data = fp32_to_fp8_bytes(fp32_values, dtype)
    state.xmem.write_address(base_addr, fp8_data)
    return len(fp32_values)


# ---------------------------------------------------------------------------
# run_test — high-level test orchestrator (mirrors emulator__run_test)
# ---------------------------------------------------------------------------


def run_test(
    *,
    inst_path: str | Path,
    setup: Callable[[IpuState], None] | None = None,
    teardown: Callable[[IpuState], None] | None = None,
    max_cycles: int = 1_000_000,
    debug_callback: DebugCallback | None = None,
) -> tuple[IpuState, int]:
    """Full test harness matching the C ``emulator__run_test`` pattern.

    1. Creates a fresh :class:`IpuState`.
    2. Loads the instruction binary.
    3. Calls *setup(state)* (e.g. to load data into XMEM).
    4. Runs the program.
    5. Calls *teardown(state)* (e.g. to dump XMEM results).

    Returns ``(state, cycles)`` so callers can inspect final state.
    """
    state = IpuState()
    load_program_from_binary(state, inst_path)

    if setup is not None:
        setup(state)

    if debug_callback is not None:
        cycles = run_with_debug(state, debug_callback, max_cycles)
    else:
        cycles = run_until_complete(state, max_cycles)

    if teardown is not None:
        teardown(state)

    return state, cycles
