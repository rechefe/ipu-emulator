"""Tests for the debug CLI — auto-generated commands, JSON export, disassembly."""

from __future__ import annotations

import json
import struct
from io import StringIO

import pytest

from ipu_emu.ipu_state import IpuState
from ipu_emu.debug_cli import (
    DebugCLI,
    DebugAction,
    debug_prompt,
    format_register,
    print_all_registers,
    state_to_json_dict,
    save_state_json,
    disassemble_current,
    _resolve_register,
)
from ipu_emu.descriptors import REGFILE_SCHEMA
from ipu_emu.execute import decode_instruction_word
from ipu_emu.emulator import load_program

from ipu_as.lark_tree import assemble


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cli(state: IpuState, commands: str) -> tuple[DebugCLI, StringIO]:
    """Create a CLI with pre-canned input and capture output."""
    inp = StringIO(commands)
    out = StringIO()
    cli = DebugCLI(state, out=out, inp=inp)
    return cli, out


def _run_cli(state: IpuState, commands: str, level: int = 0) -> tuple[DebugAction, str]:
    """Run the debug CLI with *commands* and return (action, output)."""
    cli, out = _make_cli(state, commands)
    action = cli.run(level=level)
    return action, out.getvalue()


def _make_state_with_program(asm_code: str) -> IpuState:
    encoded = assemble(asm_code)
    decoded = [decode_instruction_word(w) for w in encoded]
    state = IpuState()
    load_program(state, decoded)
    return state


# ============================================================================
# Register resolution
# ============================================================================

class TestResolveRegister:
    def test_lr_by_index(self):
        desc, idx = _resolve_register("lr5")
        assert desc is not None
        assert desc.name == "lr"
        assert idx == 5

    def test_cr_by_index(self):
        desc, idx = _resolve_register("cr15")
        assert desc is not None
        assert desc.name == "cr"
        assert idx == 15

    def test_acc_alias(self):
        desc, idx = _resolve_register("acc")
        assert desc is not None
        assert desc.name == "r_acc"

    def test_rcyclic_alias(self):
        desc, idx = _resolve_register("rcyclic")
        assert desc is not None
        assert desc.name == "r_cyclic"

    def test_rmask_alias(self):
        desc, idx = _resolve_register("rmask")
        assert desc is not None
        assert desc.name == "r_mask"

    def test_r0_alias(self):
        desc, idx = _resolve_register("r0")
        assert desc is not None
        assert desc.name == "r"
        assert idx == 0

    def test_r1_alias(self):
        desc, idx = _resolve_register("r1")
        assert desc is not None
        assert desc.name == "r"
        assert idx == 1

    def test_unknown(self):
        desc, _ = _resolve_register("xyz")
        assert desc is None


# ============================================================================
# CLI commands
# ============================================================================

class TestCLICommands:
    def test_continue(self):
        state = IpuState()
        action, output = _run_cli(state, "continue\n")
        assert action == DebugAction.CONTINUE
        assert "Continuing" in output

    def test_continue_shortcut(self):
        state = IpuState()
        action, _ = _run_cli(state, "c\n")
        assert action == DebugAction.CONTINUE

    def test_quit(self):
        state = IpuState()
        action, output = _run_cli(state, "quit\n")
        assert action == DebugAction.QUIT
        assert "Halting" in output

    def test_quit_shortcut(self):
        state = IpuState()
        action, _ = _run_cli(state, "q\n")
        assert action == DebugAction.QUIT

    def test_step(self):
        state = IpuState()
        action, output = _run_cli(state, "step\n")
        assert action == DebugAction.STEP
        assert "Stepping" in output

    def test_help(self):
        state = IpuState()
        action, output = _run_cli(state, "help\ncontinue\n")
        assert action == DebugAction.CONTINUE
        # cmd.Cmd built-in help lists documented commands
        assert "help" in output

    def test_regs(self):
        state = IpuState()
        state.regfile.set_lr(3, 42)
        action, output = _run_cli(state, "regs\ncontinue\n")
        assert "42" in output
        assert "Program Counter" in output

    def test_pc(self):
        state = IpuState()
        state.program_counter = 7
        action, output = _run_cli(state, "pc\ncontinue\n")
        assert "PC = 7" in output

    def test_get_lr(self):
        state = IpuState()
        state.regfile.set_lr(5, 0xDEAD)
        action, output = _run_cli(state, "get lr5\ncontinue\n")
        assert "0x0000dead" in output

    def test_get_cr(self):
        state = IpuState()
        state.regfile.set_cr(2, 999)
        action, output = _run_cli(state, "get cr2\ncontinue\n")
        assert "999" in output

    def test_get_acc_bytes(self):
        state = IpuState()
        state.regfile.raw("r_acc")[0] = 0xAB
        action, output = _run_cli(state, "get acc\ncontinue\n")
        assert "ab" in output

    def test_getw_acc(self):
        state = IpuState()
        struct.pack_into("<I", state.regfile.raw("r_acc"), 0, 0x12345678)
        action, output = _run_cli(state, "getw acc\ncontinue\n")
        assert "12345678" in output

    def test_get_pc(self):
        state = IpuState()
        state.program_counter = 42
        action, output = _run_cli(state, "get pc\ncontinue\n")
        assert "42" in output

    def test_set_lr(self):
        state = IpuState()
        action, output = _run_cli(state, "set lr7 0xFF\ncontinue\n")
        assert state.regfile.get_lr(7) == 0xFF
        assert "Set lr7" in output

    def test_set_cr(self):
        state = IpuState()
        action, output = _run_cli(state, "set cr0 12345\ncontinue\n")
        assert state.regfile.get_cr(0) == 12345

    def test_set_pc(self):
        state = IpuState()
        action, output = _run_cli(state, "set pc 100\ncontinue\n")
        assert state.program_counter == 100

    def test_unknown_command(self):
        state = IpuState()
        action, output = _run_cli(state, "foobar\ncontinue\n")
        assert "Unknown command" in output

    def test_lr_shortcut(self):
        state = IpuState()
        state.regfile.set_lr(0, 77)
        action, output = _run_cli(state, "lr\ncontinue\n")
        assert "77" in output
        assert "LR Registers" in output

    def test_acc_shortcut(self):
        state = IpuState()
        action, output = _run_cli(state, "acc\ncontinue\n")
        assert "r_acc" in output or "acc" in output.lower()

    def test_eof_quits(self):
        state = IpuState()
        action, output = _run_cli(state, "")  # empty = EOF
        assert action == DebugAction.QUIT


# ============================================================================
# Debug levels
# ============================================================================

class TestDebugLevels:
    def test_level0_shows_lr(self):
        state = IpuState()
        state.regfile.set_lr(0, 42)
        action, output = _run_cli(state, "continue\n", level=0)
        assert "42" in output

    def test_level1_shows_disasm(self):
        state = _make_state_with_program("set lr0 100;;\nbkpt;;")
        action, output = _run_cli(state, "continue\n", level=1)
        assert "Current Instruction" in output

    def test_level2_saves_json(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        state = IpuState()
        action, output = _run_cli(state, "continue\n", level=2)
        assert "saved to" in output
        assert (tmp_path / "ipu_debug_pc0.json").exists()


# ============================================================================
# Disassembly
# ============================================================================

class TestDisassembly:
    def test_disasm_set_lr(self):
        state = _make_state_with_program("set lr0 100;;\nbkpt;;")
        text = disassemble_current(state)
        assert "set" in text.lower()
        assert "lr0" in text.lower()

    def test_disasm_out_of_bounds(self):
        state = IpuState()
        state.program_counter = 9999
        text = disassemble_current(state)
        assert "out of bounds" in text.lower()

    def test_disasm_nop(self):
        state = IpuState()
        text = disassemble_current(state)
        assert "nop" in text.lower()


# ============================================================================
# JSON export
# ============================================================================

class TestJsonExport:
    def test_state_to_json_has_all_keys(self):
        state = IpuState()
        d = state_to_json_dict(state)
        assert "pc" in d
        for desc in REGFILE_SCHEMA:
            assert desc.name in d

    def test_json_lr_values(self):
        state = IpuState()
        state.regfile.set_lr(0, 100)
        state.regfile.set_lr(15, 999)
        d = state_to_json_dict(state)
        assert d["lr"][0] == 100
        assert d["lr"][15] == 999

    def test_json_pc(self):
        state = IpuState()
        state.program_counter = 42
        d = state_to_json_dict(state)
        assert d["pc"] == 42

    def test_save_file(self, tmp_path):
        state = IpuState()
        state.regfile.set_lr(3, 77)
        path = tmp_path / "test_dump.json"
        save_state_json(state, path)
        loaded = json.loads(path.read_text())
        assert loaded["lr"][3] == 77

    def test_json_r_regs_is_list_of_lists(self):
        state = IpuState()
        d = state_to_json_dict(state)
        assert isinstance(d["r"], list)
        assert len(d["r"]) == 2
        assert isinstance(d["r"][0], list)
        assert len(d["r"][0]) == 128


# ============================================================================
# format_register
# ============================================================================

class TestFormatRegister:
    def test_scalar(self):
        state = IpuState()
        state.regfile.set_lr(5, 0xCAFE)
        desc = next(d for d in REGFILE_SCHEMA if d.name == "lr")
        text = format_register(state, desc, index=5)
        assert "0x0000cafe" in text

    def test_byte_array(self):
        state = IpuState()
        state.regfile.raw("r_acc")[0] = 0xFF
        desc = next(d for d in REGFILE_SCHEMA if d.name == "r_acc")
        text = format_register(state, desc, offset=0, count=4)
        assert "ff" in text

    def test_word_view(self):
        state = IpuState()
        struct.pack_into("<I", state.regfile.raw("mult_res"), 0, 0xDEADBEEF)
        desc = next(d for d in REGFILE_SCHEMA if d.name == "mult_res")
        text = format_register(state, desc, as_words=True, count=1)
        assert "deadbeef" in text


# ============================================================================
# print_all_registers
# ============================================================================

class TestPrintAllRegisters:
    def test_returns_string(self):
        state = IpuState()
        text = print_all_registers(state)
        assert isinstance(text, str)
        assert "Program Counter" in text

    def test_writes_to_stream(self):
        state = IpuState()
        buf = StringIO()
        print_all_registers(state, out=buf)
        assert "Program Counter" in buf.getvalue()
