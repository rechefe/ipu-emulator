"""ISA-wide checks of one assembled program, for every program in the repo.

The standalone corpus tests (``asm_corpus``) and the kernel tree's test
(``//src/ipu-harness:test_kernel_programs``, over every kernel's
rendered default programs) run the same checks. Each takes the program's
assembly text, already rendered (no Jinja left, or the defines it renders
with), or its words and ``.bin`` bytes, and raises ``AssertionError`` when the
program breaks the property.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence

from ipu_as.errors import SlotError
from ipu_as.session import AssemblySession
from ipu_isa.codec.disasm import program_text
from ipu_isa.codec.slot import SLOT_CODECS, WORD_CODEC
from ipu_isa.codec.word import DecodedWord, hardware_word, pack_program, unpack_program
from ipu_isa.slots import SLOTS
from ipu_isa.tags import Tag

__all__ = [
    "assembles_for_hardware",
    "check_disassembly_round_trip",
    "check_hardware_target",
    "check_slot_codec",
    "check_word_codec",
]

Defines = Mapping[str, object] | None


def _session(text: str, defines: Defines) -> AssemblySession:
    return (
        AssemblySession(text) if defines is None else AssemblySession(text, defines=dict(defines))
    )


def _emulator_words(text: str, defines: Defines) -> tuple[int, ...]:
    with warnings.catch_warnings():
        # An instruction of an emulator-only slot warns when assembled for the emulator.
        warnings.simplefilter("ignore")
        return tuple(_session(text, defines).assemble().words)


def check_disassembly_round_trip(text: str, target: str, defines: Defines = None) -> bool:
    """Disassembly of the program's words for ``target`` assembles back to the same words.

    Args:
        text: The program's assembly text.
        target: ``"emulator"`` or ``"hardware"``.
        defines: The Jinja context it renders with, if any.

    Returns:
        False when ``target`` is the hardware and the program uses an
        emulator-only instruction (it has no hardware words), else True.

    Raises:
        AssertionError: The words do not survive the round trip.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*-only .* slot")
        try:
            words = _session(text, defines).assemble(target).words
        except SlotError:
            assert target == "hardware", "an emulator program refused its own slots"
            return False
        source = program_text(words, target=target)
        assert AssemblySession(source).assemble(target).words == words
    return True


def check_slot_codec(words: Sequence[int]) -> None:
    """Each word's slots, decoded by the ISA's slot codec and encoded again, are the word.

    Args:
        words: The program's words.

    Raises:
        AssertionError: A slot or a word does not re-encode to its bits.
    """
    for word in words:
        slots = WORD_CODEC.unpack(word)
        fields = DecodedWord.from_word(word)
        values = {}
        for key, decoded in slots.items():
            assert decoded.unused_bits == 0, key
            values[key] = SLOT_CODECS[key[0]].encode(decoded.instruction.mnemonic, decoded.operands)
            assert values[key] == fields.slot_value(*key), key
        assert WORD_CODEC.pack(values) == word


def check_word_codec(words: Sequence[int], bin_bytes: bytes) -> None:
    """The word codec unpacks the ``.bin`` bytes to the words and packs them back.

    Args:
        words: The program's words.
        bin_bytes: Its ``.bin`` file, as the assembler writes it.

    Raises:
        AssertionError: The codec and the assembler disagree.
    """
    decoded = unpack_program(bin_bytes)
    assert [d.to_word() for d in decoded] == list(words)
    assert [DecodedWord.from_word(w) for w in words] == decoded
    assert pack_program(list(words)) == bin_bytes


def _uses_emulator_slots(word: int) -> bool:
    return any(
        SLOTS[slot].tag is Tag.EMULATOR and decoded.instruction.mnemonic != "NOP"
        for (slot, _), decoded in WORD_CODEC.unpack(word).items()
    )


def check_hardware_target(text: str, defines: Defines = None) -> bool:
    """The hardware words are the top bits of the emulator words.

    The program assembles for the hardware unless it uses an emulator-only
    instruction, and then its hardware words are the top bits of its emulator
    words.

    Args:
        text: The program's assembly text.
        defines: The Jinja context it renders with, if any.

    Returns:
        Whether the program has hardware words.

    Raises:
        AssertionError: The two targets disagree.
    """
    emulator = _emulator_words(text, defines)
    try:
        hardware = _session(text, defines).assemble("hardware").words
    except SlotError as error:
        assert "emulator-only" in str(error)
        assert any(_uses_emulator_slots(word) for word in emulator)
        return False
    assert not any(_uses_emulator_slots(word) for word in emulator)
    assert tuple(hardware) == tuple(hardware_word(word) for word in emulator)
    return True


def assembles_for_hardware(text: str, defines: Defines = None) -> bool:
    """Whether the program assembles for the hardware target at all.

    Args:
        text: The program's assembly text.
        defines: The Jinja context it renders with, if any.
    """
    try:
        _session(text, defines).assemble("hardware")
    except SlotError:
        return False
    return True
