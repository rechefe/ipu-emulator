#include "ipu_mac_inst.h"
#include "ipu_base.h"
#include "ipu_regfile.h"
#include "logging/logger.h"
#include "ipu_math/ipu_math.h"
#include <assert.h>
#include <string.h>

static inline void ipu__mac_accumulate(ipu__obj_t *ipu, int i,
                                       uint8_t a, uint8_t b,
                                       ipu_math__dtype_t data_type,
                                       ipu__rq_reg_t *out_rq_reg)
{
    (void)ipu; // Unused parameter
    if (i >= IPU__RQ_REG_SIZE_WORDS)
    {
        LOG_ERROR("MAC accumulate index out of bounds: i=%d, max=%d", i, IPU__RQ_REG_SIZE_WORDS);
        assert(0 && "MAC accumulate index out of bounds");
    }

    // Use ipu_math module functions
    uint32_t result;
    uint32_t acc = out_rq_reg->words[i];
    ipu_math__mac(&a, &b, &acc, &result, data_type);
    out_rq_reg->words[i] = result;
}

void ipu__mac_element_element(ipu__obj_t *ipu,
                              int rx, int ry,
                              ipu_math__dtype_t data_type,
                              const ipu__regfile_t *regfile_snapshot,
                              ipu__rq_reg_t *out_rq_reg)
{
    ipu__r_reg_t r_reg_x;
    ipu__r_reg_t r_reg_y;

    ipu__get_r_register_for_mac_op(ipu, rx, regfile_snapshot, &r_reg_x);
    ipu__get_r_register_for_mac_op(ipu, ry, regfile_snapshot, &r_reg_y);

    // Read operands from snapshot - rx and ry are R registers, not RQ
    for (int i = 0; i < IPU__R_REG_SIZE_BYTES; i++)
    {
        uint8_t a = r_reg_x.bytes[i];
        uint8_t b = r_reg_y.bytes[i];
        ipu__mac_accumulate(ipu, i, a, b, data_type, out_rq_reg);
    }
}

void ipu__mac_element_vector(ipu__obj_t *ipu,
                             int r_source_0, int r_source_1,
                             int lr_idx,
                             ipu_math__dtype_t data_type,
                             const ipu__regfile_t *regfile_snapshot,
                             ipu__rq_reg_t *out_rq_reg)
{
    assert(lr_idx >= 0 && lr_idx < IPU__LR_REGS_NUM);

    ipu__r_reg_t r_source_0_value;
    ipu__r_reg_t r_source_1_value;

    ipu__get_r_register_for_mac_op(ipu, r_source_0, regfile_snapshot, &r_source_0_value);
    ipu__get_r_register_for_mac_op(ipu, r_source_1, regfile_snapshot, &r_source_1_value);

    // The LR value is the element we choose for MAC from RY reg (read from snapshot)
    uint32_t element_index = regfile_snapshot->lr_regfile.lr[lr_idx];
    assert(element_index < IPU__R_REG_SIZE_BYTES);

    // Read operands from snapshot
    for (int i = 0; i < IPU__R_REG_SIZE_BYTES; i++)
    {
        uint8_t a = r_source_0_value.bytes[i];
        uint8_t b = r_source_1_value.bytes[element_index];
        ipu__mac_accumulate(ipu, i, a, b, data_type, out_rq_reg);
    }
}

static void ipu__execute_mac_ee(ipu__obj_t *ipu, inst_parser__inst_t inst, const ipu__regfile_t *regfile_snapshot)
{
    // MAC element-element: RQ[rz] += R[rx] * R[ry]
    int rq_dest = ipu__get_rq_from_r_enum(inst.mac_inst_token_1_rx_reg_field);

    int r_source_0 = ipu__get_r_from_r_enum(inst.mac_inst_token_2_rx_reg_field);
    int r_source_1 = ipu__get_r_from_r_enum(inst.mac_inst_token_3_rx_reg_field);
    // Read data_type from CR[IPU__CR_DTYPE_REG]
    ipu_math__dtype_t data_type = ipu__get_cr_dtype(regfile_snapshot);
    ipu__mac_element_element(ipu, r_source_0, r_source_1, data_type, regfile_snapshot, &ipu->regfile.rx_regfile.rq_regs[rq_dest]);
}

static void ipu__execute_mac_ev(ipu__obj_t *ipu, inst_parser__inst_t inst, const ipu__regfile_t *regfile_snapshot)
{
    // MAC element-vector: RQ[rz][i] += R[rx][i] * R[ry][LR]
    int rq_dest = ipu__get_rq_from_r_enum(inst.mac_inst_token_1_rx_reg_field);

    int r_source_0 = ipu__get_r_from_r_enum(inst.mac_inst_token_2_rx_reg_field);
    int r_source_1 = ipu__get_r_from_r_enum(inst.mac_inst_token_3_rx_reg_field);
    int lr_idx = inst.mac_inst_token_4_lr_reg_field;

    // Read data_type from CR[IPU__CR_DTYPE_REG]
    ipu_math__dtype_t data_type = ipu__get_cr_dtype(regfile_snapshot);
    ipu__mac_element_vector(ipu, r_source_0, r_source_1, lr_idx, data_type, regfile_snapshot, &ipu->regfile.rx_regfile.rq_regs[rq_dest]);
}

static void ipu__execute_zero_rq(ipu__obj_t *ipu, inst_parser__inst_t inst)
{
    int rq_idx = ipu__get_rq_from_r_enum(inst.mac_inst_token_1_rx_reg_field);
    ipu__clear_rq_reg(ipu, rq_idx);
}

static void ipu__execute_mac_agg(ipu__obj_t *ipu, inst_parser__inst_t inst, const ipu__regfile_t *regfile_snapshot)
{
    int rq_dest = ipu__get_rq_from_r_enum(inst.mac_inst_token_1_rx_reg_field);

    int r_source_0 = ipu__get_r_from_r_enum(inst.mac_inst_token_2_rx_reg_field);
    int r_source_1 = ipu__get_r_from_r_enum(inst.mac_inst_token_3_rx_reg_field);

    int lr_idx = inst.mac_inst_token_4_lr_reg_field;

    // Read data_type from CR[IPU__CR_DTYPE_REG]
    ipu_math__dtype_t data_type = ipu__get_cr_dtype(regfile_snapshot);

    ipu__rq_reg_t r_mult_result;
    memset(&r_mult_result, 0, sizeof(ipu__rq_reg_t));

    // Multiply R source registers element-wise
    ipu__mac_element_element(ipu, r_source_0, r_source_1, data_type, regfile_snapshot, (ipu__rq_reg_t *)&r_mult_result);

    int32_t sum = 0;

    for (int i = 0; i < IPU__RQ_REG_SIZE_WORDS; i++)
    {
        int32_t add_result;
        int32_t word_val = (int32_t)r_mult_result.words[i];
        ipu_math__add(&sum, &word_val, &add_result, data_type);
        sum = add_result;
    }

    uint32_t rq_store_idx = regfile_snapshot->lr_regfile.lr[lr_idx];
    assert(rq_store_idx < IPU__RQ_REG_SIZE_WORDS);

    ipu->regfile.rx_regfile.rq_regs[rq_dest].words[rq_store_idx] = (uint32_t)sum;
}

void ipu__execute_mac_instruction(ipu__obj_t *ipu, inst_parser__inst_t inst, const ipu__regfile_t *regfile_snapshot)
{
    switch (inst.mac_inst_token_0_mac_inst_opcode)
    {
    case INST_PARSER__MAC_INST_OPCODE_MAC_EE:
        ipu__execute_mac_ee(ipu, inst, regfile_snapshot);
        break;
    case INST_PARSER__MAC_INST_OPCODE_MAC_EV:
        ipu__execute_mac_ev(ipu, inst, regfile_snapshot);
        break;
    case INST_PARSER__MAC_INST_OPCODE_MAC_AGG:
        ipu__execute_mac_agg(ipu, inst, regfile_snapshot);
        break;
    case INST_PARSER__MAC_INST_OPCODE_ZERO_RQ:
        ipu__execute_zero_rq(ipu, inst);
        break;
    case INST_PARSER__MAC_INST_OPCODE_MAC_NOP:
        // No operation for MAC
        break;
    default:
        assert(0 && "Unknown MAC instruction opcode");
    }
}
