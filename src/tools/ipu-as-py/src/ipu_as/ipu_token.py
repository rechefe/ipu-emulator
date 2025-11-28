import ipu_as.label as label

MAX_PROGRAM_SIZE = 1024


class Token:
    def __init__(self, token_str: str):
        pass

    @property
    def bits(self) -> int:
        raise NotImplementedError("bits property must be implemented by subclasses")

    def is_valid(self, value: str) -> bool:
        raise NotImplementedError("is_valid method must be implemented by subclasses")


class NumberToken(Token):
    def __init__(self, token_str: int):
        try:
            self.int = int(token_str)
        except ValueError:
            raise ValueError(f"Invalid number token value: {token_str}")
        if self.int < 0 or self.int >= (1 << self.bits):
            raise ValueError(f"Number token value out of range: {token_str}")

    @property
    def bits(self) -> int:
        raise NotImplementedError("bits property must be implemented by subclasses")

    @property 

    def encode(self) -> int:
        return self.int


class EnumToken(Token):
    def __init__(self, token_str: list[str]):
        self.token_str = token_str
        if self.token_str not in self.enum_array:
            raise ValueError(f"Invalid enum token value: {token_str}")

    def __len__(self):
        return len(self.enum_array)

    @property
    def enum_array(self):
        raise NotImplementedError(
            "enum_array property must be implemented by subclasses"
        )

    @property
    def bits(self) -> int:
        assert len(self.enum_array) > 1, (
            "EnumToken must have at least two values, check if you really need an Enum here if its "
            "just the one, consider adding 'nop' instruction instead."
        )
        return (len(self.enum_array) - 1).bit_length()

    def encode(self) -> int:
        return self._reverse_map()[self.token_str]

    def _reverse_map(self) -> dict[str, int]:
        return {name.lower(): idx for idx, name in enumerate(self.enum_array)}


class LabelToken(Token):
    def __init__(self, token_str: str):
        self.token_str = token_str
        if self.token_str not in label.ipu_labels.labels:
            print(label.ipu_labels.labels)
            raise ValueError(f"Undefined label token value: {token_str}")

    @property
    def bits(self) -> int:
        return (MAX_PROGRAM_SIZE - 1).bit_length()

    def encode(self) -> int:
        return label.ipu_labels.get_address(self.token_str)
