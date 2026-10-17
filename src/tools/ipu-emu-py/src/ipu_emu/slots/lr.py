"""LR slot handlers: the loop-register arithmetic of every LR sub-slot."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ipu_emu.errors import EmulatorFault
from ipu_emu.geometry import lrd_bytes
from ipu_isa.faults import FaultId
from ipu_isa.incr_mod_pow2_k import LR_MOD_POW2_K_ENCODED_MAX, LR_MOD_POW2_K_MIN
from ipu_isa.registers import lrd_pair

if TYPE_CHECKING:
    from ipu_emu.ipu import Ipu


def execute_lr_set(ipu: Ipu, *, reg: int, src: int) -> None:
    """Execute SET: Copy a 32-bit value from a configuration register into an LR."""
    ipu.state.regfile.set_lr(reg, src & 0xFFFFFFFF)


def execute_lr_add(ipu: Ipu, *, dest: int, src_a: int, src_b: int) -> None:
    """Execute ADD: uint32 ``dest = src_a + src_b``."""
    ipu.state.regfile.set_lr(dest, (src_a + src_b) & 0xFFFFFFFF)


def execute_lr_sub(ipu: Ipu, *, dest: int, src_a: int, src_b: int) -> None:
    """Execute SUB: uint32 ``dest = src_a - src_b``."""
    ipu.state.regfile.set_lr(dest, (src_a - src_b) & 0xFFFFFFFF)


def execute_lr_inc(ipu: Ipu, *, dest: int, imm: int) -> None:
    """Execute INC: uint32 ``dest = dest + imm`` (read-modify-write)."""
    cur = ipu._view.lr(dest, "dest")
    ipu.state.regfile.set_lr(dest, (cur + imm) & 0xFFFFFFFF)


def execute_lr_dec(ipu: Ipu, *, dest: int, imm: int) -> None:
    """Execute DEC: uint32 ``dest = dest - imm`` (read-modify-write)."""
    cur = ipu._view.lr(dest, "dest")
    ipu.state.regfile.set_lr(dest, (cur - imm) & 0xFFFFFFFF)


def _set_lrd_bytes(ipu: Ipu, n: int, lanes: bytes | bytearray) -> None:
    """Write LRDn's 8 byte elements to the live register file (inverse of ``lrd_bytes``)."""
    lo_idx, hi_idx = lrd_pair(n)
    ipu.state.regfile.set_lr(lo_idx, int.from_bytes(lanes[0:4], "little"))
    ipu.state.regfile.set_lr(hi_idx, int.from_bytes(lanes[4:8], "little"))


def _addb_broadcast(ipu: Ipu, dest: int, byte_val: int) -> None:
    """Broadcast-add a signed byte to all 8 elements of LRDn = LR(2n+1):LR(2n), clamped to [0, 255]."""
    lanes = bytearray(lrd_bytes(ipu._view.source("lr", "dest"), dest))
    signed = byte_val - 256 if byte_val >= 128 else byte_val
    for i in range(8):
        lanes[i] = max(0, min(255, lanes[i] + signed))
    _set_lrd_bytes(ipu, dest, lanes)


def execute_addb(ipu: Ipu, *, dest: int, src_b: int) -> None:
    """Execute ADDB: broadcast-add an LR/CR's low byte (signed) to LRDn's 8 byte elements."""
    _addb_broadcast(ipu, dest, src_b & 0xFF)


def execute_addbi(ipu: Ipu, *, dest: int, imm: int) -> None:
    """Execute ADDBI: broadcast-add an immediate byte (signed) to LRDn's 8 byte elements."""
    _addb_broadcast(ipu, dest, imm)


def execute_lr_inc_modp2(ipu: Ipu, *, dest: int, step: int, k: int) -> None:
    """INC.MODP2: dest <- (dest + step) mod 2^k.

    Old dest is taken from the cycle-start snapshot (read-before-write). Step is
    resolved from LcrIdx (snapshot), interpreted as uint32 like ``add``/``sub``.
    ``k`` is the raw encoded field (k_semantic − 1); semantic exponent is k + 1.
    """
    if k > LR_MOD_POW2_K_ENCODED_MAX:
        raise EmulatorFault(
            FaultId.F1,
            ipu.state.program_counter,
            f"INC.MODP2: invalid k encoding {k} (max {LR_MOD_POW2_K_ENCODED_MAX})",
        )
    k_exp = k + LR_MOD_POW2_K_MIN
    cur = ipu._view.lr(dest, "dest")
    step_u = step & 0xFFFFFFFF
    mask = (1 << k_exp) - 1
    ipu.state.regfile.set_lr(dest, ((cur + step_u) & 0xFFFFFFFF) & mask)


def execute_lr_nop(ipu: Ipu) -> None:
    """Execute NOP in LR slot: No operation."""
