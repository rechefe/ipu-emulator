import ipu_as.ipu_token as ipu_token


class LrImmediateType(ipu_token.NumberToken):
    @property
    def bits(self) -> int:
        return 16
