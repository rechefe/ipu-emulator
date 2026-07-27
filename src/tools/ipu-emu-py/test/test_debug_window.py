"""Tests for the cycle-by-cycle debug window (tracer + renderers)."""

from __future__ import annotations

import struct

import pytest

from ipu_as.lark_tree import assemble

from ipu_emu.debug_window import (
    capture_trace,
    compute_effective_mask,
    snapshot_to_dict,
)
from ipu_emu.debug_window_html import render_html, trace_to_json, write_html
from ipu_emu.debug_window_text import render_trace
from ipu_emu.emulator import load_program
from ipu_emu.execute import decode_instruction_word
from ipu_emu.ipu_config import Partition
from ipu_emu.ipu_math import DType, fp8_bytes_to_fp32
from ipu_emu.ipu_state import IpuState

_128_ONES = (1 << 128) - 1

R0_ADDR = 0x100
RC_ADDR = 0x200
OUT_ADDR = 0x400

ASM = """
SET lr0 cr2; SET lr1 cr0;;
LDR_MULT_REG r0 lr1 cr3;;
LDR_CYCLIC_MULT_REG lr1 cr4 lr1;;
MULT.RC.VV lr1 r0 0 lr1 cr15; ACC.ADD.FIRST;;
MULT.RC.VV lr1 r0 0 lr1 cr15; ACC.ADD;;
INC lr3 5; LDR_CYCLIC_MULT_REG lr3 cr4 lr1;;
STR_ACC_REG lr1 cr5;;
"""


def _make_state() -> IpuState:
    state = IpuState()
    state.regfile.set_cr(2, 7)
    state.regfile.set_cr(3, R0_ADDR)
    state.regfile.set_cr(4, RC_ADDR)
    state.regfile.set_cr(5, OUT_ADDR)
    state.xmem.write_address(R0_ADDR, b"\x02" * 128)
    state.xmem.write_address(RC_ADDR, b"\x03" * 128)
    decoded = [decode_instruction_word(w) for w in assemble(ASM)]
    load_program(state, decoded)
    return state


@pytest.fixture
def trace():
    with pytest.warns(UserWarning):  # STR_ACC_REG debug-only warning
        return capture_trace(_make_state())


# ---------------------------------------------------------------------------
# Tracer semantics
# ---------------------------------------------------------------------------


def test_trace_covers_program(trace):
    assert [s.program_counter for s in trace[:7]] == list(range(7))
    assert all(s.pc_after == s.program_counter + 1 for s in trace[:7])


def test_load_slot_xmem_address(trace):
    load = trace[1].slots["load"]
    assert load.instruction_name == "LDR_MULT_REG"
    xr = load.operands["xmem_read"]
    assert xr.register == "xmem" and xr.address_or_index == (R0_ADDR, 128)
    assert load.writes[0].register == "r0"


def test_mult_reads_are_snapshot_tagged(trace):
    mult = trace[3].slots["mult"]
    assert mult.instruction_name == "MULT.RC.VV"
    rc = mult.operands["rc_data"]
    assert rc.source == "snapshot"
    assert rc.address_or_index == (0, 128)
    assert rc.value == b"\x03" * 128
    ra = mult.operands["ra"]
    assert ra.source == "snapshot"
    assert ra.value == b"\x02" * 128
    assert mult.effective_mask == _128_ONES  # r_mask default all-ones, shift 0
    assert trace[3].slots["acc"].instruction_name == "ACC.ADD.FIRST"


def test_acc_add_reads_racc_from_snapshot(trace):
    acc = trace[4].slots["acc"]
    assert acc.instruction_name == "ACC.ADD"
    r_acc = acc.operands["r_acc"]
    assert r_acc.source == "snapshot"
    # snapshot r_acc = result of cycle 3's ACC.ADD.FIRST: every word 6
    words = struct.unpack_from("<128i", r_acc.value, 0)
    assert set(words) == {6}
    # live r_acc after this cycle: 12
    after = struct.unpack_from("<128i", trace[4].regfile["r_acc"], 0)
    assert set(after) == {12}


def test_live_lr_write_visible_to_same_cycle_load(trace):
    # INC LR3, 5 runs in the LR slot BEFORE the XMEM slot: the load's live
    # offset read must see the incremented value.
    cyc = trace[5]
    assert cyc.slots["lr0"].instruction_name == "INC"
    load = cyc.slots["load"]
    assert load.operands["offset"].value == 5
    assert load.operands["xmem_read"].address_or_index == (RC_ADDR + 5, 128)


def test_str_acc_reg_records_live_read_and_xmem_write(trace):
    acc_store = trace[6].slots["acc_store"]
    assert acc_store.operands["r_acc"].source == "live"
    assert acc_store.writes[0].register == "xmem"
    assert acc_store.writes[0].address_or_index == (OUT_ADDR, 512)


def test_regfile_snapshot_is_previous_cycle_after(trace):
    for prev, cur in zip(trace, trace[1:]):
        assert cur.regfile_snapshot == prev.regfile


def test_capture_window_start_limit():
    snaps = capture_trace(_make_state(), start=2, limit=2)
    assert [s.cycle for s in snaps] == [2, 3]


# ---------------------------------------------------------------------------
# Effective mask (pure replication of _mult_mask_and_shift steps 1-4)
# ---------------------------------------------------------------------------


def test_effective_mask_shift_left_zeroes_lane0():
    base, eff = compute_effective_mask(0, 1, Partition.P0, b"\xff" * 128)
    assert base == _128_ONES
    assert eff == _128_ONES & ~1  # lane 0 cleared by left shift

def test_effective_mask_shift_right_zeroes_lane127():
    _, eff = compute_effective_mask(0, (1 << 21) - 1, Partition.P0, b"\xff" * 128)
    # raw LR value 0x1FFFFF sign-extends to -1 at 21-bit width
    assert eff == _128_ONES & ~(1 << 127)

def test_effective_mask_partition_boundaries():
    _, eff = compute_effective_mask(0, 1, Partition.P2, b"\xff" * 128)
    assert eff & 1 == 0 and (eff >> 64) & 1 == 0  # lane 0 of each 64-lane group
    assert (eff >> 1) & 1 == 1


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def test_text_renderer(trace):
    text = render_trace(trace)
    assert "cycle 3 pc=3" in text
    assert "MULT.RC.VV" in text
    assert "SNAPSHOT" in text
    assert f"xmem@0x{R0_ADDR:x}" in text
    assert "effective_mask=0x" in text
    noted = render_trace(trace, notes=True)
    assert "issue #157" in noted


def test_json_serialisation(trace):
    d = snapshot_to_dict(trace[3])
    assert d["slots"]["mult"]["operands"]["rc_data"]["source"] == "snapshot"
    assert d["slots"]["mult"]["effective_mask"] == f"{_128_ONES:032x}"
    assert "regfile_snapshot" in d


def test_html_renderer(tmp_path, trace):
    html = render_html(trace, title="test trace")
    assert "test trace" in html and '"cycles":' in html
    out = write_html(trace, tmp_path / "trace.html")
    assert out.exists() and out.stat().st_size > 1000


# ---------------------------------------------------------------------------
# v2: dtype threading + decode tables + strip/mask rendering
# ---------------------------------------------------------------------------


def test_dtype_captured_per_cycle(trace):
    assert all(s.dtype == "INT8" for s in trace)
    assert snapshot_to_dict(trace[0])["dtype"] == "INT8"


def test_int8_trace_embeds_no_fp8_tables(trace):
    payload = trace_to_json(trace)
    assert payload["decode_tables"] == {}
    assert payload["cycles"][0]["dtype"] == "INT8"


def _fp8_trace():
    state = IpuState(dtype=DType.E4)
    state.regfile.set_cr(2, 7)
    decoded = [decode_instruction_word(w) for w in assemble("SET lr0 cr2;;\nBKPT;;")]
    load_program(state, decoded)
    return capture_trace(state)


def test_fp8_trace_embeds_decode_table():
    snaps = _fp8_trace()
    assert snaps[0].dtype == "E4"
    payload = trace_to_json(snaps)
    table = payload["decode_tables"]["E4"]
    assert len(table) == 256
    # table must come from the emulator's own decoder, byte-for-byte
    import math
    expected = [float(v) for v in fp8_bytes_to_fp32(bytes(range(256)), DType.E4)]
    for got, want in zip(table, expected):
        assert got == want or (math.isnan(got) and math.isnan(want))


def test_html_v2_markup(trace):
    html = render_html(trace)
    # strip renderer + always-expanded r_mask rows + dtype badge present
    assert "renderStrips" in html and "buildStrip" in html
    assert "renderMaskReg" in html and "maskRowBytes" in html
    assert '"decode_tables"' in html
    # R0/R1, r_cyclic, r_mask no longer in the generic hexdump VECTORS list
    vectors_block = html.split("const VECTORS = [", 1)[1].split("];", 1)[0]
    assert "'r'," not in vectors_block
    assert "r_cyclic" not in vectors_block and "r_mask" not in vectors_block
    assert "r_acc" in vectors_block and "mult_res" in vectors_block
