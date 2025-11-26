import ipu_as.ipu_token as token


class RegFields(token.EnumToken):
    pass


IPU_R_REG_NUM = 12

IPU_RD_REG_NUM = IPU_R_REG_NUM // 2
IPU_RD_TO_R_RATIO = IPU_R_REG_NUM // IPU_RD_REG_NUM
IPU_RQ_REG_NUM = IPU_RQ_REG_NUM = IPU_RD_REG_NUM // 2
IPU_RQ_TO_R_RATIO = IPU_R_REG_NUM // IPU_RQ_REG_NUM
IPU_RO_REG_NUM = IPU_RO_REG_NUM = IPU_RQ_REG_NUM // 2
IPU_RO_TO_R_RATIO = IPU_R_REG_NUM // IPU_RO_REG_NUM

RX_REG_FIELDS = (
    [f"r{i}" for i in range(IPU_R_REG_NUM)]
    + [f"rd{i * IPU_RD_TO_R_RATIO}" for i in range(IPU_RD_REG_NUM)]
    + [f"rq{i * IPU_RQ_TO_R_RATIO}" for i in range(IPU_RQ_REG_NUM)]
    + [f"ro{i * IPU_RO_TO_R_RATIO}" for i in range(IPU_RO_REG_NUM)]
    + ["mem_bypass"]
)


class RxRegField(token.EnumToken):
    @property
    def enum_array(self):
        return RX_REG_FIELDS


IPU_LR_REG_NUM = 16
LR_REG_FIELDS = [f"lr{i}" for i in range(IPU_LR_REG_NUM)]


class LrRegField(token.EnumToken):
    @property
    def enum_array(self):
        return LR_REG_FIELDS


IPU_CR_REG_NUM = 16
CR_REG_FIELDS = [f"cr{i}" for i in range(IPU_CR_REG_NUM)]


class CrRegField(token.EnumToken):
    @property
    def enum_array(self):
        return CR_REG_FIELDS
