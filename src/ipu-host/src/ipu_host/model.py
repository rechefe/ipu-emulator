"""A model of the prototype card: the IPU host port and the platform, over the
reference model.

``ModelDevice`` is a ``Device``: host software written against
``ipu_host.driver`` runs on it unchanged, in CI, with no card. It follows the
register maps and the execution flow (execution-flow.md) at the level a host
sees, with these limits (fpga-prototype.md, "The host-port model"):

- A kernel runs to its end inside the ``ARM`` (or ``RESUME``) write, so the
  host never sees a running state and the ``HALT`` command is never accepted.
- Each kernel starts from a reset reference model: LR and the vector
  registers do not carry over between kernels (IPU-EXEC-5; SW-4 says kernels
  do not rely on them).
- A table's page ``p`` is emulator XMEM row ``(table_id << 20) + (p << 10)``,
  so tables 0 to 3 are modelled; bank capacity and streaming are not.
- PC writes, the socket exerciser and the fault behaviours the reference
  model does not define (clearing F-2 to F-6) answer with an AXI error.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from ipu_emu.errors import ConfigurationError
from ipu_emu.reference import ReferenceModel, RunState
from ipu_host import driver
from ipu_host.driver import (
    BANK_BYTES,
    FORMAT_CODES,
    IMEM_BEATS,
    IMEM_STRIDE,
    PLATFORM_BASE,
    BusError,
    HaltCause,
)
from ipu_host.driver import RunState as HostState
from ipu_host.regmap import Register, RegisterMap, load
from ipu_isa.codec.slot import WORD_CODEC
from ipu_isa.faults import FaultId
from ipu_isa.geometry import INST_MEM_WORDS, LANES
from ipu_isa.tags import Tag
from ipu_isa.view import isa

#: Tables the model maps into the emulator's XMEM (512 MiB = 4 tables of 2^20 rows).
MODELLED_TABLES = 4

_EMULATOR_BITS = isa().layout.word_bits(Tag.EMULATOR)
_EMULATOR_NOPS = WORD_CODEC.pack({}) & ((1 << _EMULATOR_BITS) - 1)
_FORMATS = {code: fmt for fmt, code in FORMAT_CODES.items()}


def emulator_word(hardware: int) -> int:
    """The emulator word of a hardware word: its EMULATOR slots hold NOP."""
    return (hardware << _EMULATOR_BITS) | _EMULATOR_NOPS


@dataclass
class _Context:
    imem: list[int] = field(default_factory=lambda: [0] * INST_MEM_WORDS)
    crs: list[int] = field(default_factory=lambda: [0, 1] + [0] * 14)
    prog_len: int = 0
    fmt: int = 0
    tables: dict[int, driver.Table] = field(default_factory=dict)  # working copy
    tables_valid: bool = False
    drained: bool = True


class _Registers:
    """Plain storage for one map's registers: software access rules only."""

    def __init__(self, regmap: RegisterMap, base: int) -> None:
        self.map = regmap
        self.base = base
        self.values = {r.address: r.reset for r in regmap.registers.values()}

    def reg(self, address: int) -> Register:
        reg = self.map.at(address - self.base)
        if reg is None:
            raise BusError(f"no register at 0x{address:05X}")
        return reg

    def write(self, reg: Register, value: int) -> None:
        fields = reg.fields.values()
        if not any(f.sw_writable for f in fields):
            raise BusError(f"{reg.path} is read-only")
        old = self.values[reg.address]
        new = old
        for f in fields:
            if not f.sw_writable:
                continue
            bits = value & f.mask
            if f.name in _W1C.get(reg.path, ()):
                new &= ~bits
            else:
                new = (new & ~f.mask) | bits
        self.values[reg.address] = new

    def set_bits(self, path: str, **fields: int) -> None:
        reg = self.map.reg(path)
        self.values[reg.address] |= reg.encode(**fields)

    def get(self, path: str) -> dict[str, int]:
        reg = self.map.reg(path)
        return reg.decode(self.values[reg.address])


_W1C = {
    "REGS.INT_STATUS": ("KERNEL_START", "KERNEL_DONE", "HALTED", "HOST_ERR"),
    "REGS.FAULT_FLAGS": tuple(f"F{n}" for n in range(1, 10))
    + ("ACC_OVERFLOW", "NAN_RESULT", "QUANT_SATURATION"),
}


class ModelDevice:
    """The card, modelled. A ``driver.Device``."""

    def __init__(self, isa_digest: int = 0) -> None:
        self.ipu_map = load("ipu_host")
        self.ipu = _Registers(self.ipu_map, 0)
        self.platform = _Registers(load("proto_platform"), PLATFORM_BASE)
        self.ctx = [_Context(), _Context()]
        self.run_state = HostState.IDLE
        self.active = 0
        self.armed = False
        self.cause = HaltCause.NONE
        self.bp_index = 0
        self.fault_code = 0
        self.fault_pc = 0
        self.retired = 0
        self.skip_bp_once = False
        self.model: ReferenceModel | None = None
        self.isa_digest = isa_digest
        self.memory: dict[int, bytearray] = {}
        self._next_buffer = 0x1000_0000

    # -- DeviceMemory ----------------------------------------------------------

    def alloc(self, size: int) -> int:
        address = self._next_buffer
        self.memory[address] = bytearray(size)
        self._next_buffer += (size + 0xFFFF) & ~0xFFFF
        return address

    def _buffer(self, address: int, size: int) -> tuple[bytearray, int]:
        for base, buf in self.memory.items():
            if base <= address and address + size <= base + len(buf):
                return buf, address - base
        raise BusError(f"device address 0x{address:X} (+{size}) is not allocated")

    def write(self, address: int, data: bytes) -> None:
        buf, offset = self._buffer(address, len(data))
        buf[offset : offset + len(data)] = data

    def read(self, address: int, size: int) -> bytes:
        buf, offset = self._buffer(address, size)
        return bytes(buf[offset : offset + size])

    # -- Bus -------------------------------------------------------------------

    def read32(self, address: int) -> int:
        try:
            return self._read(address)
        except BusError:
            self.ipu.set_bits("REGS.INT_STATUS", HOST_ERR=1)
            raise

    def write32(self, address: int, value: int) -> None:
        try:
            self._write(address, value & 0xFFFF_FFFF)
        except BusError:
            self.ipu.set_bits("REGS.INT_STATUS", HOST_ERR=1)
            raise

    def _read(self, address: int) -> int:
        if address >= PLATFORM_BASE:
            return self._platform_read(address)
        imem = self._imem(address)
        if imem is not None:
            ctx, index, beat = imem
            return (self.ctx[ctx].imem[index] >> (32 * beat)) & 0xFFFF_FFFF if beat < 5 else 0
        reg = self.ipu.reg(address)
        live = self._live(reg.path)
        return live() if live is not None else self.ipu.values[reg.address]

    def _write(self, address: int, value: int) -> None:
        if address >= PLATFORM_BASE:
            self._platform_write(address, value)
            return
        imem = self._imem(address)
        if imem is not None:
            ctx, index, beat = imem
            self._unlocked(ctx)
            if beat < IMEM_BEATS:
                word = self.ctx[ctx].imem[index] & ~(0xFFFF_FFFF << (32 * beat))
                self.ctx[ctx].imem[index] = word | (value << (32 * beat))
            return
        reg = self.ipu.reg(address)
        external = self._external_write(reg, value)
        if not external:
            self.ipu.write(reg, value)

    def _imem(self, address: int) -> tuple[int, int, int] | None:
        for ctx in (0, 1):
            window = self.ipu_map.mem(f"IMEM{ctx}")
            if window.contains(address):
                offset = address - window.address
                return ctx, offset // IMEM_STRIDE, (offset % IMEM_STRIDE) // 4
        return None

    # -- IPU registers with behaviour -----------------------------------------

    def _live(self, path: str) -> Callable[[], int] | None:
        reg = self.ipu_map.reg(path)
        if path.startswith("REGS.CR_BANK["):
            ctx, index = _indices(path)
            return lambda: self.ctx[ctx].crs[index]
        if path.startswith("REGS.DEBUG.LR["):
            (index,) = _indices(path)
            return lambda: self._lr(index)
        lookups: dict[str, Callable[[], int]] = {
            "REGS.STATUS": self._status_word,
            "REGS.ISA_DIGEST": lambda: self.isa_digest,
            "REGS.PARAMS": lambda: reg.encode(IMEM_DEPTH_LOG2=10, BREAKPOINTS=4, LANES=LANES),
            "REGS.FAULT_PC": lambda: self.fault_pc,
            "REGS.RETIRED_WORDS": lambda: self.retired & 0xFFFF_FFFF,
            "REGS.KERNEL_CYCLES": lambda: self.retired & 0xFFFF_FFFF,
            "REGS.DEBUG.DBG_CAP": lambda: reg.encode(BREAKPOINTS=4),
            "REGS.DEBUG.PC": lambda: self.model.status.pc if self.model else 0,
            "REGS.COMMAND": lambda: 0,
            "REGS.PROG_LEN[0]": lambda: self.ctx[0].prog_len,
            "REGS.PROG_LEN[1]": lambda: self.ctx[1].prog_len,
            "REGS.FORMAT[0]": lambda: self.ctx[0].fmt,
            "REGS.FORMAT[1]": lambda: self.ctx[1].fmt,
        }
        return lookups.get(path)

    def _lr(self, index: int) -> int:
        value = self.model.read_register("lr", index) if self.model else 0
        assert isinstance(value, int)
        return value

    def _status_word(self) -> int:
        return self.ipu_map.reg("REGS.STATUS").encode(
            RUN_STATE=self.run_state,
            ACTIVE_CTX=self.active,
            ARMED=int(self.armed),
            TABLES_VALID=self._tables_valid(),
            CTX_LOCKED=int(self._locked(0)) | int(self._locked(1)) << 1,
            HALT_CAUSE=self.cause,
            BP_INDEX=self.bp_index,
            FAULT_CODE=self.fault_code,
        )

    def _tables_valid(self) -> int:
        return int(self.ctx[0].tables_valid) | int(self.ctx[1].tables_valid) << 1

    def _locked(self, ctx: int) -> bool:
        active = ctx == self.active and self.run_state is not HostState.IDLE
        return active or (self.armed and ctx != self.active)

    def _unlocked(self, ctx: int) -> None:
        if self._locked(ctx):
            raise BusError(f"context {ctx} is locked")

    def _external_write(self, reg: Register, value: int) -> bool:
        path = reg.path
        if path == "REGS.COMMAND":
            self._command(reg, value)
        elif path.startswith("REGS.CR_BANK["):
            ctx, index = _indices(path)
            if index < 2:
                raise BusError(f"CR{index} is a constant")
            self._unlocked(ctx)
            self.ctx[ctx].crs[index] = value
        elif path.startswith("REGS.PROG_LEN["):
            (ctx,) = _indices(path)
            self._unlocked(ctx)
            self.ctx[ctx].prog_len = reg.decode(value)["WORDS"]
        elif path.startswith("REGS.FORMAT["):
            (ctx,) = _indices(path)
            self._unlocked(ctx)
            self.ctx[ctx].fmt = reg.decode(value)["FMT"]
        elif path == "REGS.DEBUG.PC":
            raise BusError("the model cannot write the PC")
        else:
            return False
        return True

    # -- execution flow --------------------------------------------------------

    def _command(self, reg: Register, value: int) -> None:
        names = [name for name, bit in reg.decode(value).items() if bit]
        if len(names) != 1:
            raise BusError(f"a command write sets exactly one bit, not {names}")
        name = names[0]
        state = self.run_state
        debug_halt = state is HostState.HALTED and self.cause is not HaltCause.FAULT
        if name == "ARM" and state in (HostState.IDLE, HostState.HALTED) and not self.armed:
            if not self.ctx[1 - self.active].tables_valid:
                raise BusError("ARM: the inactive context's table set is not committed")
            if state is HostState.IDLE:
                self._start(1 - self.active)
            else:
                self.armed = True
        elif name == "DISARM" and state is HostState.HALTED and self.armed:
            self.armed = False
        elif name in ("STEP", "RESUME") and debug_halt:
            self.skip_bp_once = True
            self._continue(step=name == "STEP")
        elif name == "ABORT" and state is HostState.HALTED:
            self._abort()
        else:
            raise BusError(f"{name} is not accepted in {state.name}")

    def _start(self, ctx: int, *, run: bool = True) -> None:
        context = self.ctx[ctx]
        self.active = ctx
        self.run_state = HostState.RUNNING_ARMED if self.armed else HostState.RUNNING_UNARMED
        self.cause = HaltCause.NONE
        self.retired = 0
        self.ipu.set_bits("REGS.INT_STATUS", KERNEL_START=1)
        model = ReferenceModel(_FORMATS[context.fmt])
        model.load_program([emulator_word(w) for w in context.imem[: context.prog_len]])
        for index in range(2, 16):
            model.write_register("cr", context.crs[index], index=index)
        enables = self.ipu.get("REGS.FAULT_ENABLE")
        try:
            for fault in FaultId:
                model.set_fault_enable(fault, bool(enables[fault.name]))
        except ConfigurationError as error:
            raise BusError(f"ARM: {error}") from None
        for table_id, table in context.tables.items():
            if table.type is not driver.TableType.WRITE_ONLY:
                model.write_xmem(
                    _xmem_address(table_id), self.read(table.dram, table.size * BANK_BYTES)
                )
        self.model = model
        if run:
            self._continue(step=False)
        else:
            self._halt(HaltCause.STEP_DONE)

    def _continue(self, *, step: bool) -> None:
        model = self.model
        assert model is not None
        self.run_state = HostState.RUNNING_ARMED if self.armed else HostState.RUNNING_UNARMED
        breakpoints = self._breakpoints()
        if step or breakpoints:
            stopped = self._step_until(model, breakpoints, single=step)
        else:
            limit = self.ipu.get("REGS.CYCLE_LIMIT")["VALUE"]
            self.retired += model.run(max_cycles=limit).cycles
            stopped = False
        if stopped:
            return
        status = model.status
        if status.state is RunState.FAULTED:
            assert status.fault is not None
            self._halt(HaltCause.FAULT, fault=status.fault, pc=status.pc)
        elif status.state is RunState.HALTED:
            self._done(after_step=step)
        elif step:
            self._halt(HaltCause.STEP_DONE)

    def _step_until(
        self, model: ReferenceModel, breakpoints: dict[int, int], *, single: bool
    ) -> bool:
        while True:
            pc = model.status.pc
            if pc in breakpoints and not self.skip_bp_once:
                self._halt(HaltCause.BREAKPOINT, bp=breakpoints[pc])
                return True
            self.skip_bp_once = False
            record = model.step()
            if record.outcome != "fault":
                self.retired += 1
            if record.outcome != "retired" or single:
                return False

    def _breakpoints(self) -> dict[int, int]:
        found: dict[int, int] = {}
        for index in range(4):
            bp = self.ipu.get(f"REGS.DEBUG.BP[{index}]")
            if bp["ENABLE"] and bp["CTX"] == self.active:
                found.setdefault(bp["PC"], index)
        return found

    def _halt(
        self, cause: HaltCause, *, bp: int = 0, fault: FaultId | None = None, pc: int = 0
    ) -> None:
        self.run_state = HostState.HALTED
        self.cause = cause
        self.bp_index = bp
        if fault is not None:
            self.fault_code = int(fault)
            self.fault_pc = pc
            self.ipu.set_bits("REGS.FAULT_FLAGS", **{fault.name: 1})
        self.ipu.set_bits("REGS.INT_STATUS", HALTED=1)

    def _done(self, *, after_step: bool) -> None:
        ctx = self.active
        context = self.ctx[ctx]
        model = self.model
        assert model is not None
        for table_id, table in context.tables.items():
            if table.type is not driver.TableType.READ_ONLY:
                data = model.read_xmem(_xmem_address(table_id), table.size * BANK_BYTES)
                self.write(table.dram, data)
        context.tables_valid = False
        context.drained = True
        self.ipu.set_bits("REGS.INT_STATUS", KERNEL_DONE=1)
        if self.armed:
            self.armed = False
            # IPU-DBG-8: after a stepped handoff, halt at PC 0 of the next kernel.
            self._start(1 - ctx, run=not after_step)
        else:
            self.run_state = HostState.IDLE

    def _abort(self) -> None:
        context = self.ctx[self.active]
        context.tables_valid = False
        context.drained = True  # OQ-12: the aborted context's banks are freed, none drained
        self.armed = False
        self.run_state = HostState.IDLE
        self.cause = HaltCause.NONE

    # -- platform --------------------------------------------------------------

    def _platform_read(self, address: int) -> int:
        reg = self.platform.reg(address)
        if reg.path == "CU_STATUS":
            stored = self.platform.values[reg.address]
            drained = int(self.ctx[0].drained) | int(self.ctx[1].drained) << 1
            return reg.encode(TABLES_VALID=self._tables_valid(), DRAINED=drained) | (stored & 0x30)
        if reg.path == "CLK_KHZ":
            return 100_000
        return self.platform.values[reg.address]

    def _platform_write(self, address: int, value: int) -> None:
        reg = self.platform.reg(address)
        if reg.path.startswith("EX_CMD"):
            raise BusError("the model has no socket exerciser")
        if reg.path == "COMMIT":
            self._commit(reg.decode(value))
            return
        self.platform.write(reg, value)

    def _commit(self, sets: dict[str, int]) -> None:
        status = self.platform.map.reg("CU_STATUS")
        self.platform.values[status.address] &= ~status.field("COMMIT_ERR").mask
        for name, bit in sets.items():
            ctx = int(name[-1])
            context = self.ctx[ctx]
            if not bit:
                continue
            if self._locked(ctx) or not context.drained:
                self.platform.set_bits("CU_STATUS", COMMIT_ERR=1)
                continue
            context.tables = self._read_tables(ctx)
            context.tables_valid = True
            context.drained = False

    def _read_tables(self, ctx: int) -> dict[int, driver.Table]:
        tables = {}
        for table_id in range(16):
            prefix = f"TABLE_SET[{ctx}].TABLE[{table_id}]"
            cfg = self.platform.get(f"{prefix}.CFG")
            if not cfg["ENABLE"]:
                continue
            if table_id >= MODELLED_TABLES:
                self.platform.set_bits("CU_STATUS", CONFIG_ERR=1)
                continue
            dram = self.platform.get(f"{prefix}.DRAM_LO")["VALUE"]
            dram |= self.platform.get(f"{prefix}.DRAM_HI")["VALUE"] << 32
            tables[table_id] = driver.Table(
                type=driver.TableType(cfg["TYPE"]),
                dram=dram,
                size=self.platform.get(f"{prefix}.SIZE")["VALUE"],
                first_bank=self.platform.get(f"{prefix}.BANKS")["FIRST_BANK"],
                jump_back=self.platform.get(f"{prefix}.JUMP_BACK")["VALUE"],
                repetitions=self.platform.get(f"{prefix}.REPETITIONS")["VALUE"],
                format=cfg["FORMAT"],
            )
        return tables


def _indices(path: str) -> list[int]:
    """The array indices in a register path: ``CR_BANK[1].CR[3]`` gives [1, 3]."""
    return [int(i) for i in re.findall(r"\[(\d+)\]", path)]


def _xmem_address(table_id: int) -> int:
    """The emulator XMEM byte address of page 0 of ``table_id`` (8-bit rows)."""
    return (table_id << 20) * 128
