import pytest

import ipu_as.ipu_token as ipu_token


def test_base_token_bits_raises():
    t = ipu_token.Token("x")
    with pytest.raises(NotImplementedError):
        _ = t.bits


def test_numbertoken_invalid_string():
    with pytest.raises(ValueError):
        ipu_token.NumberToken("not_a_number")


def test_numbertoken_direct_bits_not_implemented():
    # NumberToken.__init__ will try to use self.bits which is NotImplemented in the base
    with pytest.raises(NotImplementedError):
        ipu_token.NumberToken("3")


def test_numbertoken_subclass_range_and_encode():
    class FourBitNumber(ipu_token.NumberToken):
        @property
        def bits(self) -> int:
            return 4

    # in-range value
    n = FourBitNumber("3")
    assert n.bits == 4
    assert n.encode() == 3

    # out-of-range value (16 is equal to 1 << 4)
    with pytest.raises(ValueError):
        FourBitNumber("16")


def test_enumtoken_basic_and_len_bits_encode():
    class MyEnum(ipu_token.EnumToken):
        def __init__(self, token_str: str):
            # ensure _array exists before calling super().__init__
            self._array = ["nop", "halt"]
            super().__init__(token_str)

        @property
        def enum_array(self):
            return self._array

    e = MyEnum("nop")
    assert len(e) == 2
    # bits = (len(enum_array)-1).bit_length() == (2-1).bit_length() == 1
    assert e.bits == 1
    assert e.encode() == 0

    e2 = MyEnum("halt")
    assert e2.encode() == 1

    with pytest.raises(ValueError):
        MyEnum("unknown")


def test_labeltoken_bits_and_encode_and_undefined(monkeypatch):
    class DummyLabels:
        def __init__(self):
            self.labels = {"L1"}
            self._addr = {"L1": 123}

        def get_address(self, name):
            return self._addr[name]

    dummy = DummyLabels()
    # Monkeypatch the label.ipu_labels used inside the module under test
    monkeypatch.setattr(ipu_token.label, "ipu_labels", dummy, raising=True)

    lt = ipu_token.LabelToken("L1")
    assert lt.bits == (ipu_token.MAX_PROGRAM_SIZE - 1).bit_length()
    assert lt.encode() == 123

    with pytest.raises(ValueError):
        ipu_token.LabelToken("NO_SUCH_LABEL")