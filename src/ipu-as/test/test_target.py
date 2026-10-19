"""`--target hardware`: the hardware slots only, emulator-only instructions refused (E05-15)."""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pytest
from click.testing import CliRunner

from ipu_as import diagnostics
from ipu_as.cli import cli
from ipu_as.errors import SlotError
from ipu_as.session import AssemblySession
from ipu_isa.codec.word import hardware_word, word_bytes
from ipu_isa.slots import LAYOUT_ORDER, SLOTS
from ipu_isa.tags import Tag
from ipu_isa.view import isa

PROGRAM = """\
SET LR0 CR1; LD.MULT R0 LR0 CR0; MULT.RC.VV LR0 R0 0 LR2 CR15; ACC.ADD;;
ADD LR0 LR1 LR2; SUB LR3 LR1 CR5; ST.AAQ LR0 CR0; ACC.MAX;;
ACTIVATE.QUANTIZE relu CR15; BNE LR0 CR0 +0;;
HALT;;
"""

#: NOP in every hardware slot: each slot's NOP opcode at its opcode field
#: (word_fields.json), worked out by hand once; every other bit is 0.
HARDWARE_NOP = 0x600000800008000080000C00180000300000100

SV_GOLDEN = Path("src/ipu-gen/test/baselines/ipu_instr_pkg.sv")


def _emulator_words(source: str) -> tuple[int, ...]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return AssemblySession(source).assemble().words


def test_hardware_words_are_the_top_bits_of_the_emulator_words():
    hardware = AssemblySession(PROGRAM).assemble("hardware")
    assert hardware.words == tuple(hardware_word(w) for w in _emulator_words(PROGRAM))
    assert all(w < 1 << isa().layout.word_bits(Tag.HARDWARE) for w in hardware.words)


def test_the_hardware_bin_is_one_hardware_word_per_instruction():
    hardware = AssemblySession(PROGRAM).assemble("hardware")
    assert len(hardware.to_bin()) == len(hardware.words) * word_bytes(Tag.HARDWARE)
    emulator = AssemblySession(PROGRAM).assemble()
    assert len(emulator.to_bin()) == len(emulator.words) * word_bytes()


def test_an_emulator_only_instruction_fails_at_its_mnemonic():
    source = "NOP;;\nSET LR0 CR1; BREAK.IFEQ LR0 10;;\n"
    with pytest.raises(SlotError, match="BREAK.IFEQ is emulator-only") as info:
        AssemblySession(source).assemble("hardware")
    assert (info.value.location.line, info.value.location.column) == (2, 14)
    assert len(_emulator_words(source)) == 2  # the default target assembles it


def test_check_reports_emulator_only_instructions_for_the_hardware():
    source = "BREAK;;\n"
    assert diagnostics.check(source) == []
    (found,) = diagnostics.check(source, target="hardware")
    assert "emulator-only" in found.message and found.line == 0


@pytest.mark.parametrize("target", ["assembler", "rtl"])
def test_an_unknown_target_is_refused(target):
    with pytest.raises(ValueError, match="emulator or hardware"):
        AssemblySession(PROGRAM).assemble(target)


def test_an_empty_hardware_word_is_every_hardware_slot_nop():
    # The nine hardware slot instances' NOP opcodes, packed by hand from
    # word_fields.json: every other bit is 0.
    (word,) = AssemblySession("NOP;;").assemble("hardware").words
    assert word == HARDWARE_NOP


def test_disassembling_the_other_targets_words_is_an_error(tmp_path):
    source = tmp_path / "p.asm"
    source.write_text(PROGRAM)
    binary, out = tmp_path / "h.bin", tmp_path / "out.txt"
    runner = CliRunner()
    argv = ["assemble", "--input", str(source), "--output", str(binary), "--target", "hardware"]
    assert runner.invoke(cli, argv).exit_code == 0
    result = runner.invoke(cli, ["disassemble", "--input", str(binary), "--output", str(out)])
    assert result.exit_code == 1, result.output
    assert "not a file of emulator words" in result.stdout


def test_the_cli_assembles_and_disassembles_hardware_words(tmp_path):
    source = tmp_path / "p.asm"
    source.write_text(PROGRAM)
    runner = CliRunner()
    hardware_bin, emulator_bin = tmp_path / "h.bin", tmp_path / "e.bin"
    argv = ["assemble", "--input", str(source), "--output"]
    assert runner.invoke(cli, [*argv, str(hardware_bin), "--target", "hardware"]).exit_code == 0
    assert runner.invoke(cli, [*argv, str(emulator_bin)]).exit_code == 0
    assert hardware_bin.read_bytes() == AssemblySession(PROGRAM).assemble("hardware").to_bin()

    text = {}
    for name, binary, target in (
        ("hardware", hardware_bin, "hardware"),
        ("emulator", emulator_bin, "emulator"),
    ):
        out = tmp_path / f"{name}.txt"
        argv = ["disassemble", "--input", str(binary), "--output", str(out), "--target", target]
        assert runner.invoke(cli, argv).exit_code == 0
        # One "MNEMONIC fields" group per slot instance, each word's prefixed by its index.
        text[name] = [g.split(":\t\t")[-1].strip() for g in out.read_text().split(";") if g.strip()]
    # The hardware text is the emulator text without the emulator-only slots.
    hardware = sum(SLOTS[s].count for s in LAYOUT_ORDER if SLOTS[s].tag is Tag.HARDWARE)
    every = sum(SLOTS[s].count for s in LAYOUT_ORDER)
    assert len(text["hardware"]) == 4 * hardware and len(text["emulator"]) == 4 * every
    for word in range(4):
        hw = text["hardware"][word * hardware : (word + 1) * hardware]
        assert hw == text["emulator"][word * every : word * every + hardware]


#: A backward branch (a label before its use) and a forward one past the end.
LOOP = """\
SET LR0 CR1;;
top: ADD LR0 LR0 CR1; ACC.ADD;;
BNE LR0 CR2 top;;
BEQ LR0 CR0 +3;;
HALT;;
"""


def test_hardware_words_disassemble_to_source_that_assembles_back(tmp_path):
    source = tmp_path / "p.asm"
    source.write_text(PROGRAM + LOOP)
    binary, text, again = tmp_path / "h.bin", tmp_path / "h.asm", tmp_path / "again.bin"
    runner = CliRunner()
    hardware = ["--target", "hardware"]
    argv = ["assemble", "--input", str(source), "--output", str(binary), *hardware]
    assert runner.invoke(cli, argv).exit_code == 0
    argv = ["disassemble", "--input", str(binary), "--output", str(text), "--syntax", "assembly"]
    assert runner.invoke(cli, [*argv, *hardware]).exit_code == 0
    written = text.read_text()
    assert "w0005: " in written and "BNE lr0 cr2 w0005" in written  # the label is generated
    assert not re.search(r"\bNOP [^;]", written)  # NOP takes no operands
    argv = ["assemble", "--input", str(text), "--output", str(again), *hardware]
    assert runner.invoke(cli, argv).exit_code == 0
    assert again.read_bytes() == binary.read_bytes()


def test_check_cli_takes_the_target():
    result = CliRunner().invoke(
        cli, ["check", "--json", "--input", "-", "--target", "hardware"], input="BREAK;;\n"
    )
    assert result.exit_code == 1
    assert "emulator-only" in result.stdout


def test_the_hardware_width_is_the_sv_package_width():
    # E03-22: the SV package describes the hardware word only.
    params = dict(
        (name, int(value))
        for name, value in re.findall(
            r"localparam int unsigned (\w+) = (\d+);", SV_GOLDEN.read_text()
        )
    )
    hardware_slots = sum(
        params[f"{slot.value.upper()}_SLOT_T_WIDTH"] * SLOTS[slot].count
        for slot in LAYOUT_ORDER
        if SLOTS[slot].tag is Tag.HARDWARE
    )
    hardware = isa().layout.word_bits(Tag.HARDWARE)
    assert params["IPU_COMPOUND_INST_WIDTH"] == hardware_slots == hardware == 155
