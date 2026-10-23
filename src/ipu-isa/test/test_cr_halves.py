"""QF, DS and MULT-constant CR layouts (ADR 0007)."""

import pytest

from ipu_isa.cr_halves import (
    DEFAULT_DS,
    DataStructure,
    MaskType,
    constant,
    decode_constant,
    decode_ds,
    decode_qf,
    ds,
    half,
    pack_halves,
    qf,
    sign_extend_half,
)
from ipu_isa.errors import IsaValueError
from ipu_isa.nx8 import Nx8Format, decode_element


def test_qf_layout():
    fmt = Nx8Format(sign=1, exp_format=4, num_bits=8)
    assert qf(fmt) == 0x0148
    assert decode_qf(0x0148) == fmt
    assert qf(Nx8Format(sign=1, exp_format=0, num_bits=8, dynamic=True)) == 0x1108


def test_invalid_qf_raises():
    with pytest.raises(IsaValueError):
        decode_qf(0x0100)  # num_bits 0
    with pytest.raises(IsaValueError):
        decode_qf(0x2148)  # dynamic field 2


def test_ds_layout_and_active_elements():
    value = ds(10, num_partitions=4, valid_partitions=3)
    assert value == 0x320A
    shape = decode_ds(value)
    assert shape == DataStructure(10, 4, MaskType.ZERO, 3)
    active = shape.active()
    assert [i for i, on in enumerate(active) if on] == [
        *range(0, 10),
        *range(32, 42),
        *range(64, 74),
    ]


def test_default_ds_is_the_whole_vector():
    assert decode_ds(DEFAULT_DS).active() == (True,) * 128
    assert DEFAULT_DS == 0x1080


@pytest.mark.parametrize(
    "value",
    [
        0x0080,  # valid_partitions 0
        0x1041 | (1 << 8),  # 65 elements in groups of 64
        0x3180,  # valid_partitions 3 > num_partitions 2
        0x1C80,  # mask_type NaN
    ],
)
def test_invalid_ds_raises(value):
    with pytest.raises(IsaValueError):
        decode_ds(value)


def test_halves():
    word = pack_halves(0x1234, 0xFFFF)
    assert half(word, high=False) == 0x1234
    assert half(word, high=True) == 0xFFFF
    assert sign_extend_half(0xFFFF) == -1
    assert sign_extend_half(0x7FFF) == 0x7FFF


def test_constant_layout():
    word = constant(0x05, 128, exp_format=4)
    assert word == 0x00148005
    c = decode_constant(word)
    assert decode_element(c.value, c.fmt, c.scale) == 3.25


def test_invalid_constant_raises():
    with pytest.raises(IsaValueError):
        decode_constant(0x00000001)  # exp_format 0: CR1 is special-cased by the emulator
