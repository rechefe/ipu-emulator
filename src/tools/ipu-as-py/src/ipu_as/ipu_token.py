import lark
import dataclasses
import ipu_as.label as label

MAX_PROGRAM_SIZE = 1024


@dataclasses.dataclass
class AnnotatedToken:
    token: lark.Token
    instr_id: int

    def get_location_string(self) -> str:
        return f"Line {self.token.line}, Column {self.token.column}"


class IpuToken:
    def __init__(self, token: AnnotatedToken):
        self.annotated_token = token
        self.token = token.token
        self.instr_id = token.instr_id

    @classmethod
    def bits(cls) -> int:
        raise NotImplementedError("bits property must be implemented by subclasses")

    def _raise_error(self, extra_msg: str = ""):
        error_msg = (
            f"Invalid token value - {self.token.value} in token {self.__class__.__name__}\n"
            f"In {self.annotated_token.get_location_string()}"
        )
        if extra_msg:
            error_msg += f"\nAdditional Information: {extra_msg}"
        raise ValueError(error_msg)


class NumberToken(IpuToken):
    def __init__(self, token: AnnotatedToken):
        super().__init__(token)
        try:
            self.int = int(token.token.value, 0)
        except ValueError:
            self._raise_error(f"Value {self.token.value} is not a valid integer")
        if not (0 <= self.int < (1 << self.bits())):
            self._raise_error(f"Value {self.int} out of range for {self.bits()} bits")

    @classmethod
    def default(cls) -> "IpuToken":
        return cls.__init__(AnnotatedToken(lark.Token("NUMBER", "0"), 0))

    @classmethod
    def bits(self) -> int:
        raise NotImplementedError("bits property must be implemented by subclasses")

    def encode(self) -> int:
        return self.int


class EnumToken(IpuToken):
    def __init__(self, token: AnnotatedToken):
        super().__init__(token)
        if self.token.value.lower() not in self.enum_array():
            self._raise_error(
                (
                    f"Value {self.token.value} not in enum options\n"
                    f"Available options: {self.enum_array()}"
                )
            )

    def __len__(self):
        return len(self.enum_array())

    @classmethod
    def enum_array(cls) -> list[str]:
        raise NotImplementedError(
            "enum_array property must be implemented by subclasses"
        )

    @classmethod
    def default(cls) -> "IpuToken":
        return cls(
            AnnotatedToken(lark.Token("ENUM", cls.enum_array()[0]), 0)
        )

    @classmethod
    def bits(cls) -> int:
        assert len(cls.enum_array()) > 1, (
            "EnumToken must have at least two values, check if you really need an Enum here if its "
            "just the one, consider adding 'nop' instruction instead."
        )
        return (len(cls.enum_array()) - 1).bit_length()

    def encode(self) -> int:
        return self._reverse_map()[self.token.value.lower()]

    def _reverse_map(self) -> dict[str, int]:
        return {name.lower(): idx for idx, name in enumerate(self.enum_array())}


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
                self._raise_error(
                    f"Relative label value {self.token.value} is not a valid integer"
                )
            target_address = self.instr_id + offset
            if not (0 <= target_address < MAX_PROGRAM_SIZE):
                self._raise_error(
                    f"Relative label target address {target_address} out of range for program size {MAX_PROGRAM_SIZE}"
                )
        elif self.token.value not in label.ipu_labels.labels:
            self._raise_error(f"Label {self.token.value} not defined")

    @classmethod
    def default(cls) -> "IpuToken":
        return cls(AnnotatedToken(lark.Token("LABEL", "+0"), 0))

    @classmethod
    def bits(self) -> int:
        return (MAX_PROGRAM_SIZE - 1).bit_length()

    def encode(self) -> int:
        if self.token.value.startswith("+"):
            offset = int(self.token.value, 0)
            return self.instr_id + offset
        return label.ipu_labels.get_address(self.token)