from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import lark

import ipu_as.utils as utils
from ipu_as.errors import LabelError, OperandError, SourceLocation
from ipu_as.symbols import is_local_reference
from ipu_isa.codec.slot import field_code
from ipu_isa.geometry import INST_MEM_WORDS
from ipu_isa.operand_types import code_bits, operand_type

if TYPE_CHECKING:
    from ipu_as.symbols import SymbolTable

#: Instruction-memory depth, as the assembler names it.
MAX_PROGRAM_SIZE = INST_MEM_WORDS

#: The number literals a number operand accepts: exactly the strings Python's
#: ``int(v, 0)`` reads (NumberToken) among those the grammar's TOKEN matches, as
#: a regex JavaScript reads too. The editor takes it from isa.json
#: (``numberPattern``); test_isa_data.py holds it against ``int(v, 0)``.
NUMBER_PATTERN = (
    r"[+-]?(?:0[xX](?:_?[0-9a-fA-F])+|0[oO](?:_?[0-7])+|0[bB](?:_?[01])+"
    r"|0(?:_?0)*|[1-9](?:_?[0-9])*)"
)


@dataclasses.dataclass
class AnnotatedToken:
    token: lark.Token
    instr_id: int
    #: The label table of the assembly this token belongs to; None outside one.
    symbols: SymbolTable | None = None

    def location(self) -> SourceLocation:
        """Where this token is in the rendered source."""
        return SourceLocation.of(self.token)

    def get_location_string(self) -> str:
        return str(self.location())


class IpuToken:
    def __init__(self, token: AnnotatedToken):
        self.annotated_token = token
        self.token = token.token
        self.instr_id = token.instr_id

    @classmethod
    def bits(cls) -> int:
        raise NotImplementedError("bits property must be implemented by subclasses")

    @classmethod
    def completion_domain(cls) -> dict:
        """What this operand accepts, for editor completion, from the method its
        constructor validates with so the two cannot disagree: a number range from
        ``value_range()`` by default; else ``enum``, ``mixed`` or ``label``."""
        if hasattr(cls, "value_range"):
            low, high = cls.value_range()
            return {"kind": "number", "min": low, "max": high}
        raise NotImplementedError(f"{cls.__name__} does not describe its values")

    def _invalid_value_message(self, extra_msg: str) -> str:
        error_msg = (
            f"Invalid token value - {self.token.value} in token {self.__class__.__name__}\n"
            f"In {self.annotated_token.get_location_string()}"
        )
        if extra_msg:
            error_msg += f"\nAdditional Information: {extra_msg}"
        return error_msg

    def _raise_error(self, extra_msg: str = ""):
        """Raise an OperandError for this token's value."""
        raise OperandError(
            self._invalid_value_message(extra_msg),
            location=self.annotated_token.location(),
            token=self.token.value,
        )


class NumberToken(IpuToken):
    def __init__(self, token: AnnotatedToken):
        super().__init__(token)
        try:
            self.int = int(token.token.value, 0)
        except ValueError:
            self._raise_error(f"Value {self.token.value} is not a valid integer")
        min_val, max_val = self.value_range()
        if not (min_val <= self.int <= max_val):
            self._raise_error(
                f"Value {self.int} out of range [{min_val}, {max_val}] for {self.bits()} bits"
            )

    @classmethod
    def default(cls) -> IpuToken:
        return cls(AnnotatedToken(lark.Token("NUMBER", "0"), 0))

    @classmethod
    def value_range(cls) -> tuple[int, int]:
        """Accepts the signed and the unsigned reading of the field's bits."""
        return -(1 << (cls.bits() - 1)), (1 << cls.bits()) - 1

    @classmethod
    def bits(self) -> int:
        raise NotImplementedError("bits property must be implemented by subclasses")

    def encode(self) -> int:
        return field_code(self.int, self.bits())


#: Each EnumToken subclass's lower-case name -> value map, built on first use.
_REVERSE_MAPS: dict[type, dict[str, int]] = {}


class EnumToken(IpuToken):
    def __init__(self, token: AnnotatedToken):
        super().__init__(token)
        if self.token.value.lower() not in self._reverse_map():
            self._raise_error(
                f"Value {self.token.value} not in enum options\n"
                f"Available options: {self.enum_array()}"
            )

    def __len__(self):
        return len(self.enum_array())

    @classmethod
    def enum_array(cls) -> list[str]:
        raise NotImplementedError("enum_array property must be implemented by subclasses")

    @classmethod
    def default(cls) -> IpuToken:
        return cls(AnnotatedToken(lark.Token("ENUM", cls.enum_array()[0]), 0))

    @classmethod
    def completion_values(cls) -> list[str]:
        """The values worth offering: all, unless a table pads with unusable names."""
        return list(cls.enum_array())

    @classmethod
    def completion_domain(cls) -> dict:
        return {"kind": "enum", "values": cls.completion_values()}

    @classmethod
    def bits(cls) -> int:
        assert len(cls.enum_array()) > 1, (
            "EnumToken must have at least two values, check if you really need an Enum here if its "
            "just the one, consider adding 'nop' instruction instead."
        )
        return code_bits(len(cls.enum_array()))

    def encode(self) -> int:
        return self._reverse_map()[self.token.value.lower()]

    @classmethod
    def _reverse_map(cls) -> dict[str, int]:
        """Lower-case member name -> value, built once per class (the enum
        tables are static ISA data). Do not mutate: it is shared."""
        reverse = _REVERSE_MAPS.get(cls)
        if reverse is None:
            reverse = {name.lower(): idx for idx, name in enumerate(cls.enum_array())}
            _REVERSE_MAPS[cls] = reverse
        return reverse

    @classmethod
    def get_all_enum_descriptors(cls) -> dict[str, any]:
        enums = dict()

        def get_all_subclasses(base_class):
            """Recursively get all subclasses of a class"""
            all_subclasses = []
            for subclass in base_class.__subclasses__():
                all_subclasses.append(subclass)
                all_subclasses.extend(get_all_subclasses(subclass))
            return all_subclasses

        for subclass in get_all_subclasses(cls):
            try:
                if issubclass(subclass, EnumToken):
                    enums[utils.camel_case_to_snake_case(subclass.__name__)] = [
                        (idx, value.upper().replace(".", "_"))
                        for idx, value in enumerate(subclass.enum_array())
                    ]
            except NotImplementedError:
                continue
        return enums


class LabelToken(IpuToken):
    def __init__(
        self,
        token: AnnotatedToken,
    ):
        super().__init__(token)
        if self.token.value.startswith("+"):
            try:
                offset = int(self.token.value, 0)
            except ValueError:
                self._raise_error(f"Relative label value {self.token.value} is not a valid integer")
            target_address = self.instr_id + offset
            if not (0 <= target_address < MAX_PROGRAM_SIZE):
                self._raise_error(
                    f"Relative label target address {target_address} out of range for program size {MAX_PROGRAM_SIZE}"
                )
        elif self.annotated_token.symbols is not None and is_local_reference(self.token.value):
            try:
                self.annotated_token.symbols.address(self.token, at=self.instr_id)
            except LabelError as error:
                # In the token-error format, which carries "Line N, Column M".
                raise LabelError(
                    self._invalid_value_message(str(error)),
                    location=error.location,
                    token=error.token,
                ) from error
        elif (
            self.annotated_token.symbols is None
            or self.token.value not in self.annotated_token.symbols
        ):
            raise LabelError(
                self._invalid_value_message(f"Label {self.token.value} not defined"),
                location=self.annotated_token.location(),
                token=self.token.value,
            )

    @classmethod
    def default(cls) -> IpuToken:
        return cls(AnnotatedToken(lark.Token("LABEL", "+0"), 0))

    @classmethod
    def completion_domain(cls) -> dict:
        # A label in the program, or a `+N` offset whose target is below MAX_PROGRAM_SIZE.
        return {"kind": "label", "relative_max": MAX_PROGRAM_SIZE - 1}

    @classmethod
    def bits(cls) -> int:
        return operand_type("Label").bits

    def encode(self) -> int:
        if self.token.value.startswith("+"):
            offset = int(self.token.value, 0)
            return self.instr_id + offset
        return self.annotated_token.symbols.address(self.token, at=self.instr_id)
