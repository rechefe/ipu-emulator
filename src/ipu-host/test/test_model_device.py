"""Host software on the modelled card: the driver's flows, and their results
against a direct reference-model run of the same program."""

import pytest

from ipu_as.lark_tree import assemble
from ipu_emu.reference import ReferenceModel
from ipu_host import driver
from ipu_host.driver import (
    BusError,
    HaltCause,
    IpuHost,
    Platform,
    RunState,
    Table,
    TableType,
    run_flat,
    stopped,
)
from ipu_host.model import ModelDevice
from ipu_isa.formats import ElementFormat

PROGRAM = """
SET LR0 CR2 ;;
ADD LR1 LR0 CR3 ;;
ACTIVATE.QUANTIZE relu CR15 ;;
ST.AAQ LR0 CR4 ;;
HALT ;;
"""
CRS = {2: 3, 3: 40, 4: 100}
IMAGE = bytes([0xAA]) * (200 * 128)


def reference_run(words, fmt, crs, image):
    model = ReferenceModel(fmt)
    model.load_program(words)
    for index, value in crs.items():
        model.write_register("cr", value, index=index)
    model.write_xmem(0, image)
    result = model.run()
    lrs = [model.read_register("lr", i) for i in range(16)]
    return result, lrs, model.read_xmem(0, driver.FLAT_BYTES)


def test_flat_run_matches_the_reference_model():
    words = assemble(PROGRAM)
    run = run_flat(ModelDevice(), words, ElementFormat.INT8, CRS, IMAGE)
    result, lrs, xmem = reference_run(words, ElementFormat.INT8, CRS, IMAGE)
    assert run.status.run_state is RunState.IDLE
    assert run.retired_words == result.cycles == 5
    assert run.lrs == lrs
    assert run.lrs[:2] == [3, 43]
    assert run.xmem == xmem
    assert run.xmem[103 * 128 : 104 * 128] == bytes(128)  # ST.AAQ at CR4 + LR0


def test_imem_round_trips_hardware_words_at_a_32_byte_stride():
    device = ModelDevice()
    ipu = IpuHost(device)
    words = assemble(PROGRAM)
    ipu.load_program(1, words)
    assert ipu.read_program(1, len(words)) == [driver.hardware_word(w) for w in words]
    assert device.read32(32 * 1 + 4 * 5) == 0  # beats 5 to 7 read 0
    assert ipu.read("REGS.PROG_LEN[1]") == len(words)


def test_cr0_and_cr1_are_constants():
    ipu = IpuHost(ModelDevice())
    assert ipu.read("REGS.CR_BANK[0].CR[1]") == 1
    with pytest.raises(BusError):
        ipu.write("REGS.CR_BANK[0].CR[0]", 5)
    assert ipu.map.reg("REGS.INT_STATUS").decode(ipu.read("REGS.INT_STATUS"))["HOST_ERR"]


def test_arm_needs_a_committed_table_set():
    ipu = IpuHost(ModelDevice())
    with pytest.raises(BusError):
        ipu.command("ARM")
    assert ipu.status().run_state is RunState.IDLE


def test_a_fault_halts_with_its_code_and_pc():
    device = ModelDevice()
    ipu, platform = IpuHost(device), Platform(device)
    words = assemble("SET LR0 CR2 ;;\nLD.MULT R0 LR0 CR3 ;;\nHALT ;;")
    ipu.load_program(1, words[:2])  # no HALT: fetching word 2 faults F-2
    ipu.set_format(1, ElementFormat.INT8)
    platform.write_table(1, 0, Table(TableType.SCRATCH_PAD, device.alloc(driver.FLAT_BYTES), 16))
    platform.commit(1)
    ipu.command("ARM")
    status = ipu.wait(stopped)
    assert (status.run_state, status.halt_cause, status.fault_code) == (
        RunState.HALTED,
        HaltCause.FAULT,
        2,
    )
    assert ipu.read("REGS.FAULT_PC") == 2
    with pytest.raises(BusError):
        ipu.command("RESUME")  # IPU-FLT-7
    ipu.command("ABORT")
    assert ipu.status().run_state is RunState.IDLE


def test_breakpoint_step_and_resume():
    device = ModelDevice()
    ipu, platform = IpuHost(device), Platform(device)
    ipu.load_program(1, assemble(PROGRAM))
    ipu.write_crs(1, CRS)
    ipu.write_fields("REGS.DEBUG.BP[0]", ENABLE=1, CTX=1, PC=1)
    platform.write_table(1, 0, Table(TableType.SCRATCH_PAD, device.alloc(driver.FLAT_BYTES), 16))
    platform.commit(1)
    ipu.command("ARM")
    status = ipu.status()
    assert (status.run_state, status.halt_cause, status.bp_index) == (
        RunState.HALTED,
        HaltCause.BREAKPOINT,
        0,
    )
    assert ipu.pc() == 1
    assert ipu.read_lrs()[:2] == [3, 0]
    with pytest.raises(BusError):
        ipu.write("REGS.CR_BANK[1].CR[2]", 9)  # the active context is locked
    ipu.command("STEP")
    assert ipu.status().halt_cause is HaltCause.STEP_DONE
    assert ipu.read_lrs()[:2] == [3, 43]
    ipu.command("RESUME")
    assert ipu.status().run_state is RunState.IDLE
    assert ipu.read("REGS.RETIRED_WORDS") == 5


def test_armed_kernel_starts_when_the_first_is_done():
    device = ModelDevice()
    ipu, platform = IpuHost(device), Platform(device)
    for ctx, crs in ((1, CRS), (0, {**CRS, 2: 7})):
        ipu.load_program(ctx, assemble(PROGRAM))
        ipu.write_crs(ctx, crs)
    ipu.write_fields("REGS.DEBUG.BP[0]", ENABLE=1, CTX=1, PC=4)
    dram = device.alloc(driver.FLAT_BYTES)
    for ctx in (1, 0):
        platform.write_table(ctx, 0, Table(TableType.SCRATCH_PAD, dram, 16))
    platform.commit(1)
    ipu.command("ARM")  # context 1 runs to its breakpoint
    platform.commit(0)
    ipu.command("ARM")  # context 0 is armed
    assert ipu.status().armed
    ipu.command("RESUME")  # context 1 is done; context 0 runs to its end
    status = ipu.status()
    assert (status.run_state, status.active_ctx, status.armed) == (RunState.IDLE, 0, False)
    assert ipu.read_lrs()[:2] == [7, 47]
