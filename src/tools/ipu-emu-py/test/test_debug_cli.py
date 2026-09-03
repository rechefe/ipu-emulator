"""Tests for the debug CLI — auto-generated commands, JSON export, disassembly."""

from __future__ import annotations

import json
import struct
from io import StringIO

import pytest

import ipu_emu.debug_cli as debug_cli_module
from ipu_emu.ipu_state import IpuState
from ipu_emu.debug_cli import (
    DebugCLI,
    DebugAction,
    VISIBLE_ALPHABET,
    debug_prompt,
    decode_16bit_cell,
    encode_16bit_cell,
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
from ipu_emu.xmem import XMEM_SIZE_BYTES

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


class _TTYStringIO(StringIO):
    def isatty(self) -> bool:
        return True


def _run_cli(state: IpuState, commands: str, level: int = 0) -> tuple[DebugAction, str]:
    """Run the debug CLI with *commands* and return (action, output)."""
    cli, out = _make_cli(state, commands)
    action = cli.run(level=level)
    return action, out.getvalue()


def _make_state_with_program(asm_code: str, *, cr: dict[int, int] | None = None) -> IpuState:
    encoded = assemble(asm_code)
    decoded = [decode_instruction_word(w) for w in encoded]
    state = IpuState()
    if cr:
        for idx, val in cr.items():
            state.regfile.set_cr(idx, val)
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
    def test_terminal_streams_enable_readline_history(self):
        state = IpuState()
        cli = DebugCLI(state, inp=_TTYStringIO(), out=_TTYStringIO())
        assert cli.use_rawinput is (debug_cli_module.readline is not None)

    def test_non_terminal_streams_keep_scripted_input(self):
        state = IpuState()
        cli, _ = _make_cli(state, "continue\n")
        assert not cli.use_rawinput

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
        action, output = _run_cli(state, "GET lr5\ncontinue\n")
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
        action, output = _run_cli(state, "SET lr7 0xFF\ncontinue\n")
        assert state.regfile.get_lr(7) == 0xFF
        assert "Set lr7" in output

    def test_set_cr(self):
        state = IpuState()
        action, output = _run_cli(state, "set cr3 12345\ncontinue\n")
        assert state.regfile.get_cr(3) == 12345

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


# ============================================================================
# XMEM inspection
# ============================================================================

class TestXmemCommand:
    def test_byte_address_hex(self):
        state = IpuState()
        state.xmem.write_address(0x100, bytes([0x12, 0x34, 0xAB, 0xCD]))
        action, output = _run_cli(state, "xmem byte 0x100 0 4 hex\ncontinue\n")
        assert action == DebugAction.CONTINUE
        assert "byte address 0x00000100" in output
        assert "12 34 ab cd" in output

    def test_wide_mode_hex_groups_every_four_bytes(self):
        state = IpuState(wide_vector_debug=True)
        state.xmem.write_address(0, bytes(range(8)))

        _, output = _run_cli(state, "xmem row 0 0 8 hex\ncontinue\n")

        assert "00 01 02 03  04 05 06 07" in output

    def test_int8_and_u8_formats(self):
        state = IpuState()
        state.xmem.write_address(0, bytes([0, 127, 128, 255]))

        _, int8_output = _run_cli(state, "xmem byte 0 0 4 int8\ncontinue\n")
        _, u8_output = _run_cli(state, "xmem byte 0 0 4 u8\ncontinue\n")

        assert f"{0:>4d} {127:>4d} {-128:>4d} {-1:>4d}" in int8_output
        assert "000 127 128 255" in u8_output

    def test_cell16_renders_character_and_ansi_colors(self):
        state = IpuState()
        cell = 200 | (14 << 8) | (4 << 12)
        state.xmem.write_address(0, struct.pack("<H", cell))

        _, output = _run_cli(state, "xmem byte 0 0 1 cell16\ncontinue\n")

        assert encode_16bit_cell(200, 14, 4) in output
        assert "XMEM row 0 starts at byte address 0x00000000" in output

    def test_cell16_separates_every_two_characters(self):
        state = IpuState()
        cells = [
            0 | (1 << 8) | (2 << 12),
            1 | (1 << 8) | (2 << 12),
            2 | (1 << 8) | (2 << 12),
            3 | (1 << 8) | (2 << 12),
        ]
        state.xmem.write_address(0, struct.pack("<4H", *cells))

        _, output = _run_cli(state, "xmem byte 0 0 4 cell16\ncontinue\n")

        expected = (
            encode_16bit_cell(0, 1, 2)
            + encode_16bit_cell(1, 1, 2)
            + " "
            + encode_16bit_cell(2, 1, 2)
            + encode_16bit_cell(3, 1, 2)
        )
        assert expected in output

    def test_cell16_displays_32_characters_per_line(self):
        state = IpuState(wide_vector_debug=True)
        state.xmem.write_address(0, struct.pack("<33H", *range(33)))

        _, output = _run_cli(state, "xmem row 0 0 33 cell16\ncontinue\n")

        value_lines = [line for line in output.splitlines() if line.startswith("  0x")]
        assert len(value_lines) == 2
        assert value_lines[0].startswith("  0x00000000:")
        assert value_lines[1].startswith("  0x00000040:")

    def test_row_address_from_cr_and_lr_narrow(self):
        state = IpuState()
        state.regfile.set_cr(4, 2)
        state.regfile.set_lr(0, 1)
        state.xmem.write_address(3 * 128, struct.pack("<2f", 1.5, -2.25))
        _, output = _run_cli(state, "xmem row cr4 lr0 2 f32\ncontinue\n")
        assert "XMEM row 3" in output
        assert "byte address 0x00000180" in output
        assert "XMEM row 3 starts at byte address 0x00000180" in output
        assert f"{1.5:>15.9g} {-2.25:>15.9g}" in output

    def test_f32_preserves_distinct_float32_values(self):
        state = IpuState()
        state.xmem.write_address(0, struct.pack("<2I", 0x3F800001, 0x3F800002))

        _, output = _run_cli(state, "xmem byte 0 0 2 f32\ncontinue\n")

        assert "1.00000012" in output
        assert "1.00000024" in output

    def test_row_address_uses_512_bytes_in_wide_mode(self):
        state = IpuState(wide_vector_debug=True)
        state.xmem.write_address(2 * 512, struct.pack("<2I", 1, 0xDEADBEEF))
        _, output = _run_cli(state, "xmem row 1 1 2 u32\ncontinue\n")
        assert "XMEM row 2" in output
        assert "byte address 0x00000400" in output
        assert f"{1:010d} {0xDEADBEEF:010d}" in output

    def test_byte_address_from_registers(self):
        state = IpuState()
        state.regfile.set_cr(3, 0x200)
        state.regfile.set_lr(2, 4)
        state.xmem.write_address(0x204, struct.pack("<I", 42))
        _, output = _run_cli(state, "xmem byte cr3 lr2 1 u32\ncontinue\n")
        assert "byte address 0x00000204" in output
        assert "0000000042" in output

    def test_command_and_operands_are_case_insensitive(self):
        state = IpuState()
        state.xmem.write_address(0, struct.pack("<f", 3.5))
        _, output = _run_cli(state, "XMEM ROW CR0 LR0 1 F32\ncontinue\n")
        assert "3.5" in output

    def test_display_marks_each_xmem_row_boundary(self):
        state = IpuState(wide_vector_debug=True)
        state.xmem.write_address(508, bytes(range(8)))

        _, output = _run_cli(state, "xmem byte 508 0 8 u8\ncontinue\n")

        assert (
            "XMEM row 0 starts at byte address 0x00000000; "
            "display starts at 0x000001fc"
        ) in output
        assert "0x000001fc: 000 001 002 003" in output
        assert "XMEM row 1 starts at byte address 0x00000200" in output
        assert "0x00000200: 004 005 006 007" in output

    @pytest.mark.parametrize(
        ("command", "message"),
        [
            ("xmem byte 1 0 1 f32", "4-byte aligned"),
            ("xmem byte 1 0 1 cell16", "2-byte aligned"),
            ("xmem byte 0 0 0 hex", "positive integer"),
            ("xmem byte 0 0 1 invalid", "format must"),
            ("xmem row rx0 0 1 hex", "Invalid address term"),
            (f"xmem byte {XMEM_SIZE_BYTES - 1} 0 2 hex", "out of bounds"),
            (f"xmem row {XMEM_SIZE_BYTES // 512} 0 1 hex", "out of range"),
        ],
    )
    def test_validation_errors_do_not_exit_debugger(self, command, message):
        state = IpuState()
        action, output = _run_cli(state, f"{command}\ncontinue\n")
        assert action == DebugAction.CONTINUE
        assert message in output


class Test16BitCellEncoding:
    def test_visible_alphabet_has_one_character_per_byte(self):
        assert len(VISIBLE_ALPHABET) == 256
        assert len(set(VISIBLE_ALPHABET)) == 256

    def test_example_index_200_is_valid(self):
        rendered = encode_16bit_cell(200, 14, 4)
        assert rendered == f"\033[96;44m{VISIBLE_ALPHABET[200]}\033[0m"
        assert decode_16bit_cell(200, 14, 4) == (
            VISIBLE_ALPHABET[200],
            14,
            4,
        )

    def test_inputs_are_normalized_to_their_encoded_bit_widths(self):
        assert decode_16bit_cell(0x100, 0x11, 0x22) == (
            VISIBLE_ALPHABET[0],
            1,
            2,
        )


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
        state = _make_state_with_program("SET lr0 cr8;;\nBKPT;;", cr={8: 100})
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
        state = _make_state_with_program("SET lr0 cr8;;\nBKPT;;", cr={8: 100})
        text = disassemble_current(state)
        tl = text.lower()
        assert "set" in tl
        assert "lr0" in tl
        assert "cr8" in tl

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

    def test_plain_save_remains_register_only(self, tmp_path):
        state = IpuState()
        path = tmp_path / "state.json"
        action, output = _run_cli(state, f"save {path}\ncontinue\n")
        assert action == DebugAction.CONTINUE
        assert "Registers saved" in output
        assert "xmem" not in json.loads(path.read_text())
        assert not (tmp_path / "state.xmem.bin").exists()

    def test_plain_save_preserves_unquoted_filename_with_spaces(self, tmp_path):
        state = IpuState()
        path = tmp_path / "state with spaces.json"

        _run_cli(state, f"save {path}\ncontinue\n")

        assert path.exists()
        assert "xmem" not in json.loads(path.read_text())

    def test_xmem_save_accepts_quoted_filename_with_spaces(self, tmp_path):
        state = IpuState()
        expected = b"test"
        state.xmem.write_address(0, expected)
        path = tmp_path / "range with spaces.json"

        _run_cli(
            state,
            f'save "{path}" xmem byte 0 0 {len(expected)}\ncontinue\n',
        )

        assert path.exists()
        assert (tmp_path / "range with spaces.xmem.bin").read_bytes() == expected

    def test_save_selected_row_range_to_binary_sidecar(self, tmp_path):
        state = IpuState(wide_vector_debug=True)
        state.regfile.set_cr(4, 2)
        state.regfile.set_lr(0, 1)
        expected = bytes((i % 251 for i in range(2 * 512)))
        state.xmem.write_address(3 * 512, expected)
        path = tmp_path / "row-state.json"

        action, output = _run_cli(
            state, f"save {path} xmem row cr4 lr0 2\ncontinue\n"
        )

        assert action == DebugAction.CONTINUE
        assert "XMEM saved" in output
        sidecar = tmp_path / "row-state.xmem.bin"
        assert sidecar.read_bytes() == expected
        metadata = json.loads(path.read_text())["xmem"]
        assert metadata == {
            "address_mode": "row",
            "row": 3,
            "row_size_bytes": 512,
            "byte_address": 3 * 512,
            "file": "row-state.xmem.bin",
            "byte_count": 2 * 512,
        }

    def test_save_selected_byte_range_to_binary_sidecar(self, tmp_path):
        state = IpuState()
        state.regfile.set_cr(3, 0x100)
        expected = bytes(range(16))
        state.xmem.write_address(0x108, expected)
        path = tmp_path / "byte-state.json"

        _run_cli(state, f"save {path} xmem byte cr3 8 16\ncontinue\n")

        assert (tmp_path / "byte-state.xmem.bin").read_bytes() == expected
        metadata = json.loads(path.read_text())["xmem"]
        assert metadata == {
            "address_mode": "byte",
            "byte_address": 0x108,
            "file": "byte-state.xmem.bin",
            "byte_count": 16,
        }

    def test_save_all_xmem_to_binary_sidecar(self, tmp_path):
        state = IpuState()
        state.xmem.write_address(0, b"start")
        state.xmem.write_address(XMEM_SIZE_BYTES - 3, b"end")
        path = tmp_path / "all-state.json"

        _run_cli(state, f"save {path} xmem all\ncontinue\n")

        sidecar = tmp_path / "all-state.xmem.bin"
        assert sidecar.stat().st_size == XMEM_SIZE_BYTES
        with sidecar.open("rb") as file:
            assert file.read(5) == b"start"
            file.seek(-3, 2)
            assert file.read() == b"end"
        metadata = json.loads(path.read_text())["xmem"]
        assert metadata == {
            "address_mode": "all",
            "file": "all-state.xmem.bin",
            "byte_address": 0,
            "byte_count": XMEM_SIZE_BYTES,
        }

    def test_invalid_xmem_save_writes_neither_file(self, tmp_path):
        state = IpuState()
        path = tmp_path / "invalid.json"
        action, output = _run_cli(
            state,
            f"save {path} xmem byte {XMEM_SIZE_BYTES - 1} 0 2\ncontinue\n",
        )
        assert action == DebugAction.CONTINUE
        assert "out of bounds" in output
        assert not path.exists()
        assert not (tmp_path / "invalid.xmem.bin").exists()


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
