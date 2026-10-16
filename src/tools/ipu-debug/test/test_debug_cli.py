"""Tests for the debug CLI — auto-generated commands, JSON export, disassembly."""

from __future__ import annotations

import json
import struct
from io import StringIO

import pytest
from emu_support import ElementFormat, make_state, new_state

import ipu_debug.cli as debug_cli_module
from ipu_debug.cli import (
    VISIBLE_ALPHABET,
    DebugAction,
    DebugCLI,
    # TODO(E08-05): DebugCLI._result is tested white-box until the mutation
    # path moves into DebugSession.
    debug_prompt,
    decode_16bit_cell,
    disassemble_current,
    encode_16bit_cell,
    format_register,
    print_all_registers,
    save_state_json,
    state_to_json_dict,
)
from ipu_debug.control import DebugControl
from ipu_emu.descriptors import REGFILE_SCHEMA
from ipu_emu.ipu_state import INST_MEM_SIZE, IpuState
from ipu_emu.xmem import XMEM_SIZE_BYTES

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cli(
    state: IpuState, commands: str, control: DebugControl | None = None
) -> tuple[DebugCLI, StringIO]:
    """Create a CLI with pre-canned input and capture output."""
    inp = StringIO(commands)
    out = StringIO()
    cli = DebugCLI(state, out=out, inp=inp, control=control)
    return cli, out


class _TTYStringIO(StringIO):
    def isatty(self) -> bool:
        return True


def _run_cli(state: IpuState, commands: str, level: int = 0) -> tuple[DebugAction, str]:
    """Run the debug CLI with *commands* and return (action, output)."""
    cli, out = _make_cli(state, commands)
    action = cli.run(level=level)
    return action, out.getvalue()


def test_active_tui_session_does_not_change_legacy_prompt(monkeypatch):
    import ipu_debug.tui as tui

    state = IpuState()
    tui.new_debug_view_session(state).active = True

    def forbidden(*args, **kwargs):
        pytest.fail("the legacy prompt must never open curses")

    monkeypatch.setattr(tui, "run_debug_tui", forbidden)
    output = StringIO()
    action = debug_prompt(state, cycle=4, inp=StringIO("step\n"), out=output)
    assert action == DebugAction.STEP
    assert "IPU Debug - Break at PC=0" in output.getvalue()


def test_legacy_cli_has_no_tui_command():
    cli, output = _make_cli(IpuState(), "")
    cli.onecmd("help")
    assert "tui" not in output.getvalue()
    cli.onecmd("tui")
    assert "Unknown command: tui" in output.getvalue()


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
        state.regfile.mutable("r_acc")[0] = 0xAB
        action, output = _run_cli(state, "get acc\ncontinue\n")
        assert "ab" in output

    def test_getw_acc(self):
        state = IpuState()
        struct.pack_into("<I", state.regfile.mutable("r_acc"), 0, 0x12345678)
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
        state = new_state(fmt=ElementFormat.FP32)
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
        state = new_state(fmt=ElementFormat.FP32)
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
        state = new_state(fmt=ElementFormat.FP32)
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
        state = new_state(fmt=ElementFormat.FP32)
        state.xmem.write_address(508, bytes(range(8)))

        _, output = _run_cli(state, "xmem byte 508 0 8 u8\ncontinue\n")

        assert (
            "XMEM row 0 starts at byte address 0x00000000; display starts at 0x000001fc"
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
        state = make_state("SET lr0 cr8;;\nBKPT;;", cr={8: 100})
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
        state = make_state("SET lr0 cr8;;\nBKPT;;", cr={8: 100})
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
        state = new_state(fmt=ElementFormat.FP32)
        state.regfile.set_cr(4, 2)
        state.regfile.set_lr(0, 1)
        expected = bytes(i % 251 for i in range(2 * 512))
        state.xmem.write_address(3 * 512, expected)
        path = tmp_path / "row-state.json"

        action, output = _run_cli(state, f"save {path} xmem row cr4 lr0 2\ncontinue\n")

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
        state.regfile.mutable("r_acc")[0] = 0xFF
        desc = next(d for d in REGFILE_SCHEMA if d.name == "r_acc")
        text = format_register(state, desc, offset=0, count=4)
        assert "ff" in text

    def test_word_view(self):
        state = IpuState()
        struct.pack_into("<I", state.regfile.mutable("mult_res"), 0, 0xDEADBEEF)
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


class TestRuntimeControls:
    def test_breakpoints_are_shared_between_prompt_instances(self):
        state = IpuState()
        control = DebugControl()
        cli, _ = _make_cli(state, "", control=control)
        cli.onecmd("break")
        cli.onecmd("break 0x10")
        other, output = _make_cli(state, "", control=control)
        other.onecmd("breaks")
        assert "Breakpoints: 0, 16" in output.getvalue()
        other.onecmd("delete 0")
        assert cli.control.breakpoints == {16}
        other.onecmd("delete all")
        assert not cli.control.breakpoints

    @pytest.mark.parametrize("argument", ["-1", "999999", "invalid", "1 2"])
    @pytest.mark.parametrize("command", ["break", "delete", "until"])
    def test_invalid_pc_does_not_change_controls(self, command, argument):
        cli, output = _make_cli(IpuState(), "")
        cli.control.add_breakpoint(2)
        assert not cli.onecmd(f"{command} {argument}")
        assert cli.control.breakpoints == {2}
        assert cli.control.run_target is None
        assert "PC must" in output.getvalue()

    def test_breakpoints_cannot_be_changed_from_outside(self):
        cli, _ = _make_cli(IpuState(), "")
        with pytest.raises(AttributeError):
            cli.control.breakpoints.add(3)
        assert not cli.control.breakpoints

    def test_out_of_range_pc_is_refused(self):
        state = IpuState()
        state.program_counter = 5
        cli, output = _make_cli(state, "")
        cli.onecmd("set pc 99999")
        assert f"PC must be between 0 and {INST_MEM_SIZE - 1}" in output.getvalue()
        assert state.program_counter == 5

    def test_until_current_pc_stays_paused(self):
        cli, output = _make_cli(IpuState(), "")
        assert cli.onecmd("until 0") is None
        assert cli.control.run_target is None
        assert "Already at PC 0" in output.getvalue()
        assert cli.onecmd("until 2") is True
        assert cli.control.run_target == 2
        assert cli._result == DebugAction.CONTINUE

    @pytest.mark.parametrize("register", ["cr0", "cr1"])
    def test_read_only_set_reports_error_without_exiting(self, register):
        state = IpuState()
        cli, output = _make_cli(state, "")
        assert cli.onecmd(f"set {register} 99") is None
        assert "read-only" in output.getvalue()
        assert state.regfile.get_cr(0) == 0
        assert state.regfile.get_cr(1) == 1


# ---------------------------------------------------------------------------
# Adapter conformance: the CLI only forwards to DebugSession
# ---------------------------------------------------------------------------


def _debug_state(state: IpuState, control: DebugControl) -> tuple:
    return (
        state.program_counter,
        state_to_json_dict(state),
        control.breakpoints,
        control.run_target,
    )


@pytest.mark.parametrize(
    ("command", "call"),
    [
        ("set lr3 5", lambda s: s.set_register("lr3", 5)),
        ("set cr7 0x10", lambda s: s.set_register("cr7", 0x10)),
        ("set pc 4", lambda s: s.set_register("pc", 4)),
        ("break 7", lambda s: s.add_breakpoint(7)),
        ("delete 2", lambda s: s.remove_breakpoint(2)),
        ("delete all", lambda s: s.clear_breakpoints()),
        ("until 9", lambda s: s.run_until(9)),
    ],
)
def test_cli_commands_leave_the_same_state_as_session_calls(command, call):
    from ipu_debug.session import DebugSession

    def prepared() -> tuple[IpuState, DebugControl]:
        state = IpuState()
        control = DebugControl()
        control.add_breakpoint(2)
        control.add_breakpoint(5)
        return state, control

    cli_state, cli_control = prepared()
    cli, _ = _make_cli(cli_state, "", control=cli_control)
    cli.onecmd(command)

    direct_state, direct_control = prepared()
    session = DebugSession(direct_state, direct_control)
    call(session)

    assert _debug_state(cli_state, cli_control) == _debug_state(direct_state, direct_control)
    assert cli.session.generation == session.generation == 1


def test_cli_set_bumps_generation_so_cached_views_refresh():
    cli, _ = _make_cli(IpuState(), "")
    cli.onecmd("set lr0 1")
    cli.onecmd("set r0 1")  # rejected: no mutation
    assert cli.session.generation == 1
