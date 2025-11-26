import ipu_as.opcodes as opcodes
import ipu_as.ipu_token as ipu_token
import ipu_as.reg as reg
import ipu_as.immediate as immediate


class Inst:
    def __init__(self, inst: str):
        token_strs = self._inst_to_tokens_strs(inst)
        self.tokens = []
        for token_str, token_type in zip(token_strs, self.expected_tokens):
            try:
                self.tokens.append(token_type(token_str))
            except ValueError as e:
                raise ValueError(
                    f"Invalid token value: {token_str} for token type: {token_type.__name__} in instruction '{inst}' - \n{e}"
                ) from e

    @property
    def expected_tokens(self) -> list[type[ipu_token.Token]]:
        raise NotImplementedError(
            "expected_tokens property must be implemented by subclasses"
        )

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
        for token, token_type in reversed(
            list(zip(self.tokens, self.expected_tokens))
        ):
            encoded_inst |= token.encode() << shift_amount
            shift_amount += token.bits
        return encoded_inst

    @property
    def bits(self) -> int:
        return sum(token.bits for token in self.tokens)


class XmemInst(Inst):
    @property
    def expected_tokens(self) -> list[type[ipu_token.Token]]:
        return [opcodes.XmemInstOpcode, reg.LrRegField, reg.CrRegField]


class MacInst(Inst):
    @property
    def expected_tokens(self) -> list[type[ipu_token.Token]]:
        return [opcodes.MacInstOpcode, reg.RxRegField, reg.RxRegField, reg.RxRegField, reg.LrRegField]


class LrInst(Inst):
    @property
    def expected_tokens(self) -> list[type[ipu_token.Token]]:
        return [opcodes.LrInstOpcode, reg.LrRegField, immediate.LrImmediateType]


class CondInst(Inst):
    @property
    def expected_tokens(self) -> list[type[ipu_token.Token]]:
        return [
            opcodes.CondInstOpcode,
            reg.LrRegField,
            reg.LrRegField,
            ipu_token.LabelToken,
        ]
