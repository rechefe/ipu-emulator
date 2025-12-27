#include "ipu_regfile.h"
#include "xmem/xmem.h"
#include <string.h>
#include <assert.h>

void ipu__load_r_reg(ipu__obj_t *ipu, int rx_idx, int cr_idx, int lr_idx, const ipu__regfile_t *regfile_snapshot)
{
    assert(rx_idx >= 0 && rx_idx < IPU__R_REGS_NUM);
    assert(cr_idx >= 0 && cr_idx < IPU__CR_REGS_NUM);
    assert(lr_idx >= 0 && lr_idx < IPU__CR_REGS_NUM);

    // Read address from snapshot, write to current regfile
    uint32_t xmem_addr =
        regfile_snapshot->lr_regfile.lr[lr_idx] +
        regfile_snapshot->cr_regfile.cr[cr_idx];
    xmem__read_address(ipu->xmem, xmem_addr, (uint8_t *)&ipu->regfile.rx_regfile.r_regs[rx_idx], IPU__R_REG_SIZE_BYTES);
}

void ipu__store_r_reg(ipu__obj_t *ipu, int rx_idx, int cr_idx, int lr_idx, const ipu__regfile_t *regfile_snapshot)
{
    assert(rx_idx >= 0 && rx_idx < IPU__R_REGS_NUM);
    assert(cr_idx >= 0 && cr_idx < IPU__CR_REGS_NUM);
    assert(lr_idx >= 0 && lr_idx < IPU__CR_REGS_NUM);

    // Read address and data from snapshot
    uint32_t xmem_addr =
        regfile_snapshot->lr_regfile.lr[lr_idx] +
        regfile_snapshot->cr_regfile.cr[cr_idx];
    xmem__write_address(ipu->xmem, xmem_addr, (const uint8_t *)&regfile_snapshot->rx_regfile.r_regs[rx_idx], IPU__R_REG_SIZE_BYTES);
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

void ipu__set_lr(ipu__obj_t *ipu, int lr_idx, uint32_t imm)
{
    ipu->regfile.lr_regfile.lr[lr_idx] = imm;
}

void ipu__add_lr(ipu__obj_t *ipu, int lr_idx, uint32_t imm)
{
    assert(lr_idx >= 0 && lr_idx < IPU__LR_REGS_NUM);
    // Note: This function is kept for compatibility but should not be used
    // in parallel execution context. Use direct assignment with snapshot instead.
    ipu->regfile.lr_regfile.lr[lr_idx] += imm;
}

void ipu__set_cr(ipu__obj_t *ipu, int cr_idx, uint32_t imm)
{
    assert(cr_idx >= 0 && cr_idx < IPU__CR_REGS_NUM);
    ipu->regfile.cr_regfile.cr[cr_idx] = imm;
}

void ipu__get_r_register_for_mac_op(ipu__obj_t *ipu,
                                    int r_reg_index,
                                    const ipu__regfile_t *regfile_snapshot,
                                    ipu__r_reg_t *out_r_reg)
{
    (void)regfile_snapshot; // Unused parameter
    assert((r_reg_index >= 0 && r_reg_index < IPU__R_REGS_NUM) || r_reg_index == INST_PARSER__RX_REG_FIELD_MEM_BYPASS);
    if (r_reg_index == INST_PARSER__RX_REG_FIELD_MEM_BYPASS)
    {
        memcpy(out_r_reg, &ipu->misc.mem_bypass_reg, sizeof(ipu__r_reg_t));
        return;
    }
    else
    {
        memcpy(out_r_reg, &ipu->regfile.rx_regfile.r_regs[r_reg_index], sizeof(ipu__r_reg_t));
    }
}
