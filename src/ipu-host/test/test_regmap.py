"""The register maps compile, and the facts other repositories rely on hold.

The offsets pinned here are wire format for the RTL repository (ADR 0007): a
change to one is a change to the IPU's host port. Update the pin and call the
change out in the PR.
"""

import pytest

from ipu_host import regmap
from ipu_host.regmap import RegisterMapError


@pytest.fixture(scope="module")
def host():
    return regmap.load("ipu_host")


@pytest.fixture(scope="module")
def platform():
    return regmap.load("proto_platform")


def test_imem_windows_hold_1024_words_at_a_32_byte_stride(host):
    assert host.mem("IMEM0").address == 0x00000
    assert host.mem("IMEM1").address == 0x08000
    assert host.mem("IMEM0").size == host.mem("IMEM1").size == 1024 * 32


def test_register_file_offsets(host):
    pinned = {
        "REGS.ID": 0x10000,
        "REGS.COMMAND": 0x10010,
        "REGS.STATUS": 0x10014,
        "REGS.INT_STATUS": 0x10020,
        "REGS.PROG_LEN[1]": 0x10034,
        "REGS.FAULT_ENABLE": 0x10040,
        "REGS.RETIRED_WORDS": 0x10064,
        "REGS.CR_BANK[0].CR[0]": 0x10100,
        "REGS.CR_BANK[1].CR[15]": 0x1017C,
        "REGS.DEBUG.PC": 0x10804,
        "REGS.DEBUG.LR[15]": 0x1087C,
        "REGS.DEBUG.BP[3]": 0x1088C,
    }
    assert {path: host.reg(path).address for path in pinned} == pinned


def test_commands_are_one_bit_each(host):
    command = host.reg("REGS.COMMAND")
    bits = {name: f.lsb for name, f in command.fields.items()}
    assert bits == {"ARM": 0, "DISARM": 1, "HALT": 2, "STEP": 3, "RESUME": 4, "ABORT": 5}
    assert all(f.width == 1 for f in command.fields.values())


def test_status_decodes(host):
    status = host.reg("REGS.STATUS")
    word = status.encode(RUN_STATE=3, ACTIVE_CTX=1, HALT_CAUSE=4, FAULT_CODE=6)
    assert status.decode(word)["FAULT_CODE"] == 6
    assert status.decode(word)["RUN_STATE"] == 3


def test_every_fault_is_enabled_after_reset(host):
    # IPU-FLT-6: bit n is F-n, n = 1..9.
    assert host.reg("REGS.FAULT_ENABLE").reset == 0x3FE


def test_encode_rejects_unknown_fields_and_overflow(host):
    with pytest.raises(RegisterMapError):
        host.reg("REGS.STATUS").encode(NOPE=1)
    with pytest.raises(RegisterMapError):
        host.reg("REGS.DEBUG.PC").encode(PC=1024)


def test_platform_tables(platform):
    assert platform.reg("TABLE_SET[0].TABLE[0].CFG").address == 0x1000
    assert platform.reg("TABLE_SET[1].TABLE[15].REPETITIONS").address == 0x1200 + 15 * 0x20 + 0x18
    assert platform.reg("EX_DATA[31]").address == 0x1FC


def test_maps_fit_the_prototype_control_space(host, platform):
    # docs/content/specs/fpga-prototype.md, "Control address map".
    assert host.size <= 0x20000
    assert platform.size <= 0x20000
