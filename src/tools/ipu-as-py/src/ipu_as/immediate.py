import ipu_as.ipu_token as ipu_token


class LrImmediateType(ipu_token.NumberToken):
    @classmethod
    def bits(cls) -> int:
        return 16
