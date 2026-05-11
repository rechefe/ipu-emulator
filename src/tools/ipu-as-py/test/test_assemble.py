from ipu_as.inst import XmemInst


def test_instruction_reference_formats_syntax_and_examples():
    description = XmemInst.description()

    assert "**Syntax:** `LDR_MULT_REG dest, offset, base`" in description
    assert "LDR_MULT_REG R0, LR0, CR0;;" in description


def test_instruction_reference_uses_elements_and_uppercase_registers():
    description = XmemInst.description()

    assert "Memory[offset + base]  # 128 elements" in description
    assert "Memory[offset + base] = AAQ_RESULT  # 128 elements" in description
