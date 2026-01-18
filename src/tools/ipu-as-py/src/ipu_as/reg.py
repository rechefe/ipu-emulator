import ipu_as.ipu_token as ipu_token

IPU_MULT_STAGE_REG_R_NUM = 2

MULT_STAGE_REG_R_FIELDS = [f"r{i}" for i in range(IPU_MULT_STAGE_REG_R_NUM)] + [
    "mem_bypass"
]

class MultStageRegField(ipu_token.EnumToken):
    @classmethod
    def enum_array(cls):
        return MULT_STAGE_REG_R_FIELDS


IPU_ACC_STAGE_REG_R_FIELDS = ["rt_tf32_high", "rt_tf32_low", "rt_fp32"]


class AccStageRegField(ipu_token.EnumToken):
    @classmethod
    def enum_array(cls):
        return IPU_ACC_STAGE_REG_R_FIELDS


IPU_LR_REG_NUM = 16
LR_REG_FIELDS = [f"lr{i}" for i in range(IPU_LR_REG_NUM)]


class LrRegField(ipu_token.EnumToken):
    @classmethod
    def enum_array(cls):
        return LR_REG_FIELDS


IPU_CR_REG_NUM = 16
CR_REG_FIELDS = [f"cr{i}" for i in range(IPU_CR_REG_NUM)]


class CrRegField(ipu_token.EnumToken):
    @classmethod
    def enum_array(cls):
        return CR_REG_FIELDS
