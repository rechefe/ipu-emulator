import ipu_as.opcodes as opcodes
import ipu_as.ipu_token as ipu_token
import ipu_as.reg as reg
import ipu_as.immediate as immediate


def validate_inst_structure(cls: type) -> type:
    """Class decorator to validate instruction structure."""
    cls._validate_instr_structure()
    return cls


@validate_inst_structure
class Inst:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._inst_mapping_table = dict()
        cls._validate_instr_structure_and_generate_mapping()

    def __init__(self, inst: str):
        self.inst_str = inst
        token_strs = self._inst_to_tokens_strs(inst)
        self.tokens = []

        opcode_str = token_strs[0].lower()
        self._add_token(opcode_str, self.opcode)

        for i, token_type in enumerate(self._non_opcode_tokens):
            if i in self._inst_mapping_table[opcode_str].keys():
                self._add_token(self._inst_mapping_table[opcode_str][i], token_type)
            else:
                self._add_token(token_type.default_str, token_type)

        for token_str, token_type in zip(token_strs, self.expected_tokens):
            self._add_token(token_str, token_type)

    def _add_token(self, token_str: str, token_type):
        try:
            self.tokens.append(token_type(token_str))
        except ValueError as e:
            raise ValueError(
                f"Invalid token value: {token_str} for token type: {token_type.__name__} in instruction - {self.inst_str} - \n{e}"
            ) from e

    def _inst_structure(self) -> dict[str, list[type[ipu_token.Token]]]:
        raise NotImplementedError(
            "instr_structure property must be implemented by subclasses"
        )

    def _validate_instr_structure_and_generate_mapping(self) -> bool:
        for opcode, token_list in self._inst_structure().items():
            assert isinstance(
                opcode, self.inst_tokens[0]
            ), f"Configuration of {self.__class__.__name__} is invalid, opcode key must be of type {self.inst_tokens[0].__name__}"

            self._inst_mapping_table[opcode] = self._reversed_inst_mapping_table(
                self._find_instruction_inst_mapping(token_list)
            )

    def _find_instruction_inst_mapping(self, token_list: list[type[ipu_token.Token]]):
        inst_mapping = [-1 for _ in token_list]
        full_token_list = [False for _ in self._non_opcode_tokens]
        for i, token in enumerate(token_list):
            for j, token_type in enumerate(self._non_opcode_tokens):
                if isinstance(token, token_type) and not full_token_list[i]:
                    full_token_list[j] = True
                    inst_mapping[i] = j
        assert (
            -1 not in inst_mapping
        ), "Configuration of {self.__class__.__name__} is invalid"
        return inst_mapping

    @property
    def inst_tokens(self) -> list[type[ipu_token.Token]]:
        raise NotImplementedError(
            "inst_tokens property must be implemented by subclasses"
        )

    @staticmethod
    def _reversed_inst_mapping_table(mapping: list[int]) -> dict[int, int]:
        return {(i, j) for j, i in enumerate(mapping)}

    @property
    def _non_opcode_tokens(self) -> list[type[ipu_token.Token]]:
        return self.inst_tokens[1:] if len(self.inst_tokens) > 1 else []

    @property
    def opcode(self) -> type[ipu_token.Token]:
        return self.inst_tokens[0]

    def _inst_to_tokens_strs(self, inst: str) -> list[str]:
        stripped_inst = inst.strip()
        self.inst = stripped_inst.split(" ")

        if len(self.inst) == len(self.expected_tokens):
            return self.inst
        raise ValueError(
            f"Instruction '{inst}' does not match expected token count of {len(self.expected_tokens)}"
        )

    def encode(self) -> int:
        encoded_inst = 0
        shift_amount = 0
        for token in reversed(self.tokens):
            encoded_inst |= token.encode() << shift_amount
            shift_amount += token.bits
        return encoded_inst

    @property
    def bits(self) -> int:
        return sum(token.bits for token in self.tokens)


class XmemInst(Inst):
    @property
    def inst_tokens(self) -> list[type[ipu_token.Token]]:
        return [opcodes.XmemInstOpcode, reg.LrRegField, reg.CrRegField]

    def _inst_structure(self) -> dict[str, list[type[ipu_token.Token]]]:
        return {
            "ldr": [reg.LrRegField, reg.CrRegField],
            "str": [reg.LrRegField, reg.CrRegField],
        }


class MacInst(Inst):
    @property
    def inst_tokens(self) -> list[type[ipu_token.Token]]:
        return [
            opcodes.MacInstOpcode,
            reg.RxRegField,
            reg.RxRegField,
            reg.RxRegField,
            reg.LrRegField,
        ]

    def _inst_structure(self) -> dict[str, list[type[ipu_token.Token]]]:
        return {
            "mac.ee": [reg.RxRegField, reg.RxRegField, reg.RxRegField],
            "mac.ev": [reg.RxRegField, reg.RxRegField, reg.RxRegField, reg.LrRegField],
        }


class LrInst(Inst):
    @property
    def inst_tokens(self) -> list[type[ipu_token.Token]]:
        return [opcodes.LrInstOpcode, reg.LrRegField, immediate.LrImmediateType]

    def _inst_structure(self) -> dict[str, list[type[ipu_token.Token]]]:
        return {
            "incr": [reg.LrRegField, immediate.LrImmediateType],
            "set": [reg.LrRegField, immediate.LrImmediateType],
        }


class CondInst(Inst):
    @property
    def inst_tokens(self) -> list[type[ipu_token.Token]]:
        return [
            opcodes.CondInstOpcode,
            reg.LrRegField,
            reg.LrRegField,
            ipu_token.LabelToken,
        ]

    def _inst_structure(self) -> dict[str, list[type[ipu_token.Token]]]:
        return {
            "beq": [reg.LrRegField, reg.LrRegField, ipu_token.LabelToken],
            "bne": [reg.LrRegField, reg.LrRegField, ipu_token.LabelToken],
            "blt": [reg.LrRegField, reg.LrRegField, ipu_token.LabelToken],
            "bnz": [reg.LrRegField, ipu_token.LabelToken],
            "bz": [reg.LrRegField, ipu_token.LabelToken],
            "b": [ipu_token.LabelToken],
            "bkpt": [],
        }
