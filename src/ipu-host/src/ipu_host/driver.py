"""Drive an IPU and the prototype platform through their register maps.

A ``Device`` is the two things the host software needs from a card: a 32-bit
``Bus`` onto the kernel's control address space, and ``DeviceMemory`` (the
HBM the cache unit's DMA reads and writes). The card (``ipu_host.xrt``) and
the model (``ipu_host.model``) are both Devices, so the same host code runs on
either.

``IpuHost`` drives the IPU host port (``ipu_host.rdl``); ``Platform`` drives
the prototype platform (``proto_platform.rdl``); ``run_flat`` runs one program
in flat mode (docs/content/specs/fpga-prototype.md, "Flat mode").
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol

from ipu_host.regmap import RegisterMap, load
from ipu_isa.codec.word import hardware_word
from ipu_isa.formats import FORMATS, ElementFormat
from ipu_isa.geometry import INST_MEM_WORDS
from ipu_isa.tags import Tag

#: Bytes per instruction word in an IMEM window (ADR 0007).
IMEM_STRIDE = 32
#: 32-bit beats that carry a word's 160 padded bits; the rest of the stride reads 0.
IMEM_BEATS = 5
#: Base of each map in the kernel's control address space.
IPU_BASE = 0x00000
PLATFORM_BASE = 0x20000
#: XMEM geometry (cache-unit.md, section 4).
BANKS = 16
ROWS_PER_BANK = 1024
ROW_BYTES = 128
BANK_BYTES = ROWS_PER_BANK * ROW_BYTES
#: The bytes flat mode maps: every bank.
FLAT_BYTES = BANKS * BANK_BYTES


class BusError(RuntimeError):
    """The device answered an access with an AXI error (SLVERR)."""


class Bus(Protocol):
    """32-bit accesses to the kernel's control address space."""

    def read32(self, address: int) -> int:
        """The 32-bit value at byte ``address``. Raises BusError on SLVERR."""
        ...

    def write32(self, address: int, value: int) -> None:
        """Write ``value`` at byte ``address``. Raises BusError on SLVERR."""
        ...


class DeviceMemory(Protocol):
    """The device memory the cache unit's DMA reaches (HBM on the U55C)."""

    def alloc(self, size: int) -> int:
        """A new buffer of ``size`` bytes; returns its device address (4 KiB aligned)."""
        ...

    def write(self, address: int, data: bytes) -> None:
        """Copy ``data`` from the host to device memory."""
        ...

    def read(self, address: int, size: int) -> bytes:
        """Copy ``size`` bytes from device memory to the host."""
        ...


class Device(Bus, DeviceMemory, Protocol):
    """A card, or a model of one."""


class RunState(IntEnum):
    """``STATUS.RUN_STATE`` (execution-flow.md, Run states)."""

    IDLE = 0
    RUNNING_UNARMED = 1
    RUNNING_ARMED = 2
    HALTED = 3


class HaltCause(IntEnum):
    """``STATUS.HALT_CAUSE``."""

    NONE = 0
    HALT_REQUEST = 1
    STEP_DONE = 2
    BREAKPOINT = 3
    FAULT = 4


class TableType(IntEnum):
    """A table's type, relative to the IPU (cache-unit.md, section 5.1)."""

    READ_ONLY = 0
    WRITE_ONLY = 1
    READ_AFTER_WRITE = 2
    SCRATCH_PAD = 3


#: ``FORMAT.FMT`` encodings: the hardware formats, in ``ElementFormat`` order.
FORMAT_CODES: Mapping[ElementFormat, int] = {
    fmt: code for code, fmt in enumerate(f for f in ElementFormat if FORMATS[f].tag is Tag.HARDWARE)
}


@dataclass(frozen=True)
class Status:
    """``STATUS``, decoded."""

    run_state: RunState
    active_ctx: int
    armed: bool
    tables_valid: int
    ctx_locked: int
    halt_cause: HaltCause
    bp_index: int
    fault_code: int
    stalled: bool


@dataclass(frozen=True)
class Table:
    """One table of a table set (``proto_platform.rdl``, ``TABLE_SET[k].TABLE[t]``)."""

    type: TableType
    dram: int
    size: int
    first_bank: int = 0
    jump_back: int = 0
    repetitions: int = 0
    format: int = 0


class IpuHost:
    """The IPU host port of one device."""

    def __init__(self, bus: Bus, base: int = IPU_BASE) -> None:
        self.bus = bus
        self.base = base
        self.map: RegisterMap = load("ipu_host")

    def read(self, path: str) -> int:
        """The register at ``path``."""
        return self.bus.read32(self.base + self.map.reg(path).address)

    def write(self, path: str, value: int) -> None:
        """Write the register at ``path``."""
        self.bus.write32(self.base + self.map.reg(path).address, value)

    def write_fields(self, path: str, **fields: int) -> None:
        """Write the register at ``path`` with the named fields; the rest 0."""
        self.write(path, self.map.reg(path).encode(**fields))

    def identify(self) -> int:
        """The ISA digest the IPU was built from (``ISA_DIGEST``).

        Raises:
            BusError: The ID register does not hold the IPU's magic.
        """
        magic = self.map.reg("REGS.ID").reset
        if self.read("REGS.ID") != magic:
            raise BusError(f"no IPU host port at 0x{self.base:X}")
        return self.read("REGS.ISA_DIGEST")

    def status(self) -> Status:
        """``STATUS``, decoded."""
        f = self.map.reg("REGS.STATUS").decode(self.read("REGS.STATUS"))
        return Status(
            run_state=RunState(f["RUN_STATE"]),
            active_ctx=f["ACTIVE_CTX"],
            armed=bool(f["ARMED"]),
            tables_valid=f["TABLES_VALID"],
            ctx_locked=f["CTX_LOCKED"],
            halt_cause=HaltCause(f["HALT_CAUSE"]),
            bp_index=f["BP_INDEX"],
            fault_code=f["FAULT_CODE"],
            stalled=bool(f["STALLED"]),
        )

    def inactive_ctx(self) -> int:
        """The context the next ``ARM`` from ``IDLE`` starts."""
        return 1 - self.status().active_ctx

    def load_program(self, ctx: int, words: Sequence[int], *, emulator_words: bool = True) -> None:
        """Write a program to IMEM bank ``ctx`` and set its program length.

        Args:
            ctx: The context (0 or 1); it must not be locked.
            words: The program's words, from word 0.
            emulator_words: The words are emulator words (``ipu-as`` default
                output): keep their hardware part. False: hardware words.

        Raises:
            ValueError: The program is larger than IMEM.
        """
        if len(words) > INST_MEM_WORDS:
            raise ValueError(f"{len(words)} words do not fit IMEM ({INST_MEM_WORDS})")
        window = self.base + self.map.mem(f"IMEM{ctx}").address
        for index, word in enumerate(words):
            hw = hardware_word(word) if emulator_words else word
            for beat in range(IMEM_BEATS):
                value = (hw >> (32 * beat)) & 0xFFFF_FFFF
                self.bus.write32(window + IMEM_STRIDE * index + 4 * beat, value)
        self.write_fields(f"REGS.PROG_LEN[{ctx}]", WORDS=len(words))

    def read_program(self, ctx: int, count: int) -> list[int]:
        """The first ``count`` hardware words of IMEM bank ``ctx``."""
        window = self.base + self.map.mem(f"IMEM{ctx}").address
        words = []
        for index in range(count):
            word = 0
            for beat in range(IMEM_BEATS):
                word |= self.bus.read32(window + IMEM_STRIDE * index + 4 * beat) << (32 * beat)
            words.append(word)
        return words

    def set_format(self, ctx: int, fmt: ElementFormat) -> None:
        """Set the element format of context ``ctx``.

        Raises:
            ValueError: ``fmt`` is an emulator-only format.
        """
        if fmt not in FORMAT_CODES:
            raise ValueError(f"{fmt} is not a hardware format")
        self.write_fields(f"REGS.FORMAT[{ctx}]", FMT=FORMAT_CODES[fmt])

    def write_crs(self, ctx: int, values: Mapping[int, int]) -> None:
        """Write CRs of bank ``ctx``; ``values`` maps CR index (2-15) to value."""
        for index, value in sorted(values.items()):
            self.write(f"REGS.CR_BANK[{ctx}].CR[{index}]", value & 0xFFFF_FFFF)

    def read_lrs(self) -> list[int]:
        """``LR0``-``LR15`` (IPU-DBG-13)."""
        return [self.read(f"REGS.DEBUG.LR[{i}]") for i in range(16)]

    def pc(self) -> int:
        """The program counter."""
        return self.read("REGS.DEBUG.PC")

    def command(self, name: str) -> None:
        """Issue one command (``ARM``, ``DISARM``, ``HALT``, ``STEP``, ``RESUME``, ``ABORT``)."""
        self.write_fields("REGS.COMMAND", **{name: 1})

    def wait(
        self,
        stop: Callable[[Status], bool],
        timeout: float = 10.0,
        poll: float = 0.001,
    ) -> Status:
        """Poll ``STATUS`` until ``stop`` holds.

        Raises:
            TimeoutError: ``stop`` did not hold within ``timeout`` seconds.
        """
        deadline = time.monotonic() + timeout
        while True:
            status = self.status()
            if stop(status):
                return status
            if time.monotonic() > deadline:
                raise TimeoutError(f"IPU still {status.run_state.name} after {timeout} s")
            time.sleep(poll)


def stopped(status: Status) -> bool:
    """The IPU is ``IDLE`` or ``HALTED``."""
    return status.run_state in (RunState.IDLE, RunState.HALTED)


class Platform:
    """The prototype platform of one device: control, cache unit, exerciser."""

    def __init__(self, bus: Bus, base: int = PLATFORM_BASE) -> None:
        self.bus = bus
        self.base = base
        self.map: RegisterMap = load("proto_platform")

    def read(self, path: str) -> int:
        """The register at ``path``."""
        return self.bus.read32(self.base + self.map.reg(path).address)

    def write(self, path: str, value: int) -> None:
        """Write the register at ``path``."""
        self.bus.write32(self.base + self.map.reg(path).address, value)

    def field(self, path: str, name: str) -> int:
        """One field of the register at ``path``."""
        return self.map.reg(path).decode(self.read(path))[name]

    def write_table(self, table_set: int, table_id: int, table: Table | None) -> None:
        """Write one table of a set; ``None`` disables it."""
        prefix = f"TABLE_SET[{table_set}].TABLE[{table_id}]"
        if table is None:
            self.write(f"{prefix}.CFG", 0)
            return
        cfg = self.map.reg(f"{prefix}.CFG").encode(ENABLE=1, TYPE=table.type, FORMAT=table.format)
        self.write(f"{prefix}.CFG", cfg)
        self.write(f"{prefix}.BANKS", table.first_bank)
        self.write(f"{prefix}.DRAM_LO", table.dram & 0xFFFF_FFFF)
        self.write(f"{prefix}.DRAM_HI", table.dram >> 32)
        self.write(f"{prefix}.SIZE", table.size)
        self.write(f"{prefix}.JUMP_BACK", table.jump_back)
        self.write(f"{prefix}.REPETITIONS", table.repetitions)

    def commit(self, table_set: int) -> None:
        """Commit table set ``table_set`` (CU-11).

        Raises:
            BusError: The cache unit refused the commit (``CU_STATUS.COMMIT_ERR``).
        """
        self.write("COMMIT", 1 << table_set)
        if self.field("CU_STATUS", "COMMIT_ERR"):
            raise BusError(f"the cache unit refused the commit of table set {table_set}")

    def wait_drained(self, ctx: int, timeout: float = 10.0, poll: float = 0.001) -> None:
        """Wait until the outputs of the last kernel of ``ctx`` are in DRAM (CU-18).

        Raises:
            TimeoutError: Not drained within ``timeout`` seconds.
        """
        deadline = time.monotonic() + timeout
        while not (self.field("CU_STATUS", "DRAINED") >> ctx) & 1:
            if time.monotonic() > deadline:
                raise TimeoutError(f"context {ctx} not drained after {timeout} s")
            time.sleep(poll)


@dataclass(frozen=True)
class FlatRun:
    """What ``run_flat`` read back."""

    status: Status
    retired_words: int
    pc: int
    lrs: list[int]
    xmem: bytes


def run_flat(
    device: Device,
    program: Sequence[int],
    fmt: ElementFormat,
    crs: Mapping[int, int],
    xmem: bytes,
    *,
    timeout: float = 10.0,
) -> FlatRun:
    """Run one program in flat mode and read back XMEM.

    Flat mode is one ``scratch_pad`` table, table 0, over all 16 banks: row
    ``r`` of the emulator's XMEM is bank ``r >> 10``, row ``r & 1023``. The
    program runs in the inactive context from ``IDLE``.

    Args:
        device: The card or the model.
        program: The program's emulator words (``ipu-as`` default output).
        fmt: A hardware element format.
        crs: CR index (2-15) to value.
        xmem: The starting XMEM image, at most 2 MiB; the rest is zero.

    Returns:
        The status, counters and LRs after the run, and the 2 MiB XMEM image
        the cache unit drained.

    Raises:
        ValueError: ``xmem`` is larger than flat mode maps.
    """
    if len(xmem) > FLAT_BYTES:
        raise ValueError(f"flat mode maps {FLAT_BYTES} bytes; the image has {len(xmem)}")
    ipu = IpuHost(device)
    platform = Platform(device)
    ctx = ipu.inactive_ctx()
    ipu.load_program(ctx, program)
    ipu.set_format(ctx, fmt)
    ipu.write_crs(ctx, crs)
    dram = device.alloc(FLAT_BYTES)
    device.write(dram, xmem.ljust(FLAT_BYTES, b"\0"))
    platform.write_table(ctx, 0, Table(TableType.SCRATCH_PAD, dram, size=BANKS))
    for table_id in range(1, 16):
        platform.write_table(ctx, table_id, None)
    platform.commit(ctx)
    ipu.command("ARM")
    status = ipu.wait(stopped, timeout)
    if status.run_state is RunState.IDLE:
        platform.wait_drained(ctx, timeout)
    return FlatRun(
        status=status,
        retired_words=ipu.read("REGS.RETIRED_WORDS"),
        pc=ipu.pc(),
        lrs=ipu.read_lrs(),
        xmem=device.read(dram, FLAT_BYTES),
    )
