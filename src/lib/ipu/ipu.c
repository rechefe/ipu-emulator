#include "ipu.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

ipu__obj_t *ipu__init_ipu()
{
    ipu__obj_t *ipu = malloc(sizeof(ipu__obj_t));
    memset(&ipu->regfile, 0, sizeof(ipu__regfile_t));
    xmem__obj_t *xmem = xmem__initialize_xmem();
    ipu->xmem = xmem;
    return ipu;
}

void ipu__load_r_reg(ipu__obj_t *ipu, int rx_idx, int cr_idx, int lr_idx)
{
    assert(rx_idx >= 0 && rx_idx < IPU__R_REGS_NUM);
    assert(cr_idx >= 0 && cr_idx < IPU__CR_REGS_NUM);
    assert(lr_idx >= 0 && lr_idx < IPU__CR_REGS_NUM);

    uint32_t xmem_addr =
        ipu->regfile.lr_regfile.lr[lr_idx] +
        ipu->regfile.cr_regfile.cr[cr_idx];
    xmem__read_address(ipu->xmem, xmem_addr, (uint8_t *)&ipu->regfile.rx_regfile.r_regs[rx_idx], IPU__R_REG_SIZE_BYTES);
}

void ipu__store_r_reg(ipu__obj_t *ipu, int rx_idx, int cr_idx, int lr_idx)
{
    assert(rx_idx >= 0 && rx_idx < IPU__R_REGS_NUM);
    assert(cr_idx >= 0 && cr_idx < IPU__CR_REGS_NUM);
    assert(lr_idx >= 0 && lr_idx < IPU__CR_REGS_NUM);

    uint32_t xmem_addr =
        ipu->regfile.lr_regfile.lr[lr_idx] +
        ipu->regfile.cr_regfile.cr[cr_idx];
    xmem__write_address(ipu->xmem, xmem_addr, (const uint8_t *)&ipu->regfile.rx_regfile.r_regs[rx_idx], IPU__R_REG_SIZE_BYTES);
}

void ipu__clear_reg(ipu__obj_t *ipu, int index)
{
    assert(index >= 0 && index < IPU__R_REGS_NUM);
    memset(&ipu->regfile.rx_regfile.r_regs[index], 0, sizeof(ipu__r_reg_t));
}

void ipu__clear_rq_reg(ipu__obj_t *ipu, int index)
{
    assert(index >= 0 && index < IPU__RQ_REGS_NUM);
    memset(&ipu->regfile.rx_regfile.rq_regs[index], 0, sizeof(ipu__rq_reg_t));
}

void ipu__mac_element_element(ipu__obj_t *ipu,
                              int rz, int rx, int ry,
                              ipu__data_type_t data_type)
{
    assert(rz >= 0 && rz < IPU__RQ_REGS_NUM);
    assert(rx >= 0 && rx < IPU__R_REGS_NUM);
    assert(ry >= 0 && ry < IPU__R_REGS_NUM);

    for (int i = 0; i < IPU__R_REG_SIZE_BYTES; i++)
    {
        uint8_t a = ipu->regfile.rx_regfile.rq_regs[rx].bytes[i];
        uint8_t b = ipu->regfile.rx_regfile.rq_regs[ry].bytes[i];
        uint32_t product = ipu__mult(a, b, data_type);
        uint32_t acc = ipu->regfile.rx_regfile.rq_regs[rz].words[i];
        uint32_t result = ipu__add(acc, product, data_type);
        ipu->regfile.rx_regfile.rq_regs[rz].words[i] = result;
    }
}

void ipu__mac_element_vector(ipu__obj_t *ipu,
                             int rz, int rx, int ry,
                             int lr_idx,
                             ipu__data_type_t data_type)
{
    assert(rz >= 0 && rz < IPU__RQ_REGS_NUM);
    assert(rx >= 0 && rx < IPU__R_REGS_NUM);
    assert(ry >= 0 && ry < IPU__R_REGS_NUM);
    assert(lr_idx >= 0 && lr_idx < IPU__LR_REGS_NUM);

    // The LR value is the element we choose for MAC from RY reg
    uint32_t element_index = ipu->regfile.lr_regfile.lr[lr_idx];

    assert(element_index < IPU__R_REG_SIZE_BYTES);

    for (int i = 0; i < IPU__R_REG_SIZE_BYTES; i++)
    {
        uint8_t a = ipu->regfile.rx_regfile.r_regs[rx].bytes[i];
        uint8_t b = ipu->regfile.rx_regfile.r_regs[ry].bytes[element_index];
        uint32_t product = ipu__mult(a, b, data_type);
        uint32_t acc = ipu->regfile.rx_regfile.rq_regs[rz].words[i];
        uint32_t result = ipu__add(acc, product, data_type);
        ipu->regfile.rx_regfile.rq_regs[rz].words[i] = result;
    }
}

void ipu__mac(ipu__obj_t *ipu,
              int r_reg_index,
              uint8_t multiplicand,
              ipu__data_type_t data_type)
{
    (void)ipu;
    (void)r_reg_index;
    (void)multiplicand;
    (void)data_type;
    for (int i = 0; i < IPU__R_REG_SIZE_BYTES; i++)
    {
    }
}

uint32_t ipu__add_twice_uint4_t(uint8_t a, uint8_t b)
{
    ipu__uint8_t_as_uint4_t_t a_as_two = (ipu__uint8_t_as_uint4_t_t) a;
    ipu__uint8_t_as_uint4_t_t b_as_two = (ipu__uint8_t_as_uint4_t_t) b;

    uint16_t res_low = (a_as_two.f.low + b_as_two.f.low);
    uint16_t res_high = (b_as_two.f.high + b_as_two.f.high);
    return (res_high << IPU__UINT16T_BITS) | res_low;
}

uint32_t ipu__add(uint32_t a, uint32_t b, ipu__data_type_t data_type)
{
    switch (data_type)
    {
    case IPU__DATA_TYPE_INT8:
        return (uint32_t)(a + b);
    case IPU__DATA_TYPE_INT4:
        return ipu__add_twice_uint4_t(a, b);
    case IPU__DATA_TYPE_FP16:
        // Placeholder for FP16 addition
        assert(0 && "FP16 addition not implemented");
        return 0;
    case IPU__DATA_TYPE_FP8:
        // Placeholder for FP8 addition
        assert(0 && "FP8 addition not implemented");
        return 0;
    case IPU__DATA_TYPE_FP4:
        // Placeholder for FP4 addition
        assert(0 && "FP4 addition not implemented");
        return 0;
    case IPU__DATA_TYPE_BIN:
    {
        return a ^ b;
    }
    default:
        assert(0 && "Unsupported data type in ipu__add");
        return 0;
    }
}

uint32_t ipu__mult_twice_uint4_t(uint8_t a, uint8_t b)
{
    ipu__uint8_t_as_uint4_t_t a_as_two = (ipu__uint8_t_as_uint4_t_t) a;
    ipu__uint8_t_as_uint4_t_t b_as_two = (ipu__uint8_t_as_uint4_t_t) b;

    uint16_t res_low = (a_as_two.f.low * b_as_two.f.low);
    uint16_t res_high = (b_as_two.f.high * b_as_two.f.high);
    return (res_high << IPU__UINT16T_BITS) | res_low;
}

uint32_t ipu__mult(uint8_t a, uint8_t b, ipu__data_type_t data_type)
{
    switch (data_type)
    {
    case IPU__DATA_TYPE_INT8:
        return (uint32_t)(a * b);
    case IPU__DATA_TYPE_INT4:
        return ipu__mult_twice_uint4_t(a, b);
    case IPU__DATA_TYPE_FP16:
        // Placeholder for FP16 multiplication
        assert(0 && "FP16 multiplication not implemented");
        return 0;
    case IPU__DATA_TYPE_FP8:
        // Placeholder for FP8 multiplication
        assert(0 && "FP8 multiplication not implemented");
        return 0;
    case IPU__DATA_TYPE_FP4:
        // Placeholder for FP4 multiplication
        assert(0 && "FP4 multiplication not implemented");
        return 0;
    case IPU__DATA_TYPE_BIN:
    default:
        assert(0 && "Unsupported data type in ipu__mult");
        return 0;
    }
}

void ipu__set_lr(ipu__obj_t *ipu, int lr_idx, uint32_t imm)
{
    ipu->regfile.lr_regfile.lr[lr_idx] = imm;
}

void ipu__set_cr(ipu__obj_t *ipu, int cr_idx, uint32_t imm)
{
    ipu->regfile.cr_regfile.cr[cr_idx] = imm;
}