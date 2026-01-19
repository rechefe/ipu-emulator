#include "ipu_regfile.h"
#include "xmem/xmem.h"
#include <string.h>
#include <assert.h>

void ipu__load_r_reg(ipu__obj_t *ipu, int xmem_addr, ipu__r_reg_t *out_r_reg)
{
    assert(xmem_addr >= 0);
    xmem__read_address(ipu->xmem, xmem_addr, (uint8_t *)out_r_reg, IPU__R_REG_SIZE_BYTES);
}

void ipu__store_r_reg(ipu__obj_t *ipu, int xmem_addr, ipu__r_reg_t *r_reg)
{
    assert(xmem_addr >= 0);
    xmem__write_address(ipu->xmem, xmem_addr, (const uint8_t *)r_reg, IPU__R_REG_SIZE_BYTES);
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



void ipu__get_r_cyclic_at_idx(ipu__obj_t *ipu, int start_idx, ipu__r_reg_t *out_r_cyclic_reg)
{
    int start_idx_after_mod = start_idx % IPU__R_CYCLIC_REG_SIZE_BYTES;
    assert(start_idx_after_mod >= 0);
    assert(start_idx_after_mod <= IPU__R_CYCLIC_REG_SIZE_BYTES);

    if (start_idx_after_mod + IPU__R_REG_SIZE_BYTES > IPU__R_CYCLIC_REG_SIZE_BYTES)
    {
        memcpy(out_r_cyclic_reg->bytes,
               &ipu->regfile.mult_stage_regfile.r_cyclic_reg.bytes[start_idx_after_mod],
               IPU__R_CYCLIC_REG_SIZE_BYTES - start_idx_after_mod);
        memcpy(&out_r_cyclic_reg->bytes[IPU__R_CYCLIC_REG_SIZE_BYTES - start_idx_after_mod],
               &ipu->regfile.mult_stage_regfile.r_cyclic_reg.bytes[0],
               (start_idx_after_mod + IPU__R_REG_SIZE_BYTES) - IPU__R_CYCLIC_REG_SIZE_BYTES);
    }
    else
    {
        memcpy(out_r_cyclic_reg->bytes,
               &ipu->regfile.mult_stage_regfile.r_cyclic_reg.bytes[start_idx_after_mod],
               IPU__R_REG_SIZE_BYTES);
    }
}

void ipu__set_r_cyclic_at_idx(ipu__obj_t *ipu, int start_idx, ipu__r_reg_t *in_r_cyclic_reg)
{
    int start_idx_after_mod = start_idx % IPU__R_CYCLIC_REG_SIZE_BYTES;
    assert(start_idx_after_mod >= 0);
    assert(start_idx_after_mod <= IPU__R_CYCLIC_REG_SIZE_BYTES);

    if (start_idx_after_mod + IPU__R_REG_SIZE_BYTES > IPU__R_CYCLIC_REG_SIZE_BYTES)
    {
        memcpy(&ipu->regfile.mult_stage_regfile.r_cyclic_reg.bytes[start_idx_after_mod],
               in_r_cyclic_reg->bytes,
               IPU__R_CYCLIC_REG_SIZE_BYTES - start_idx_after_mod);
        memcpy(&ipu->regfile.mult_stage_regfile.r_cyclic_reg.bytes[0],
               &in_r_cyclic_reg->bytes[IPU__R_CYCLIC_REG_SIZE_BYTES - start_idx_after_mod],
               (start_idx_after_mod + IPU__R_REG_SIZE_BYTES) - IPU__R_CYCLIC_REG_SIZE_BYTES);
    }
    else
    {
        memcpy(&ipu->regfile.mult_stage_regfile.r_cyclic_reg.bytes[start_idx_after_mod],
               in_r_cyclic_reg->bytes,
               IPU__R_REG_SIZE_BYTES);
    }
}

// Legacy function - kept for compatibility
void ipu__pack_tf32_array(uint8_t *out_bytes, fp__tf32_t *in_fp32_array, size_t num_elements)
{
    int bits_in_out_bytes = 0;
    for (size_t i = 0; i < num_elements; i++)
    {
        fp__tf32_t val = in_fp32_array[i];

        // Pack into out_bytes bit by bit
        for (int bit_idx = 0; bit_idx < FP__TF32_WIDTH; bit_idx++)
        {
            // Extract bit from val.w
            uint32_t bit_val = (val.w >> bit_idx) & 0x1;

            int write_bit_pos_global = bits_in_out_bytes + bit_idx;
            int write_byte_pos = write_bit_pos_global / 8;
            int write_bit_pos = write_bit_pos_global % 8;

            // Clear the target bit
            out_bytes[write_byte_pos] &= ~(1 << write_bit_pos);
            // Set the target bit
            out_bytes[write_byte_pos] |= (bit_val << write_bit_pos);
        }

        bits_in_out_bytes += FP__TF32_WIDTH;
    }
}

// Legacy function - kept for compatibility
void ipu__unpack_into_tf32_array(fp__tf32_t *out_fp32_array, const uint8_t *in_bytes, size_t num_elements)
{
    int bits_in_in_bytes = 0;
    for (size_t i = 0; i < num_elements; i++)
    {
        fp__tf32_t val;
        val.w = 0;

        // Unpack from in_bytes bit by bit
        for (int bit_idx = 0; bit_idx < FP__TF32_WIDTH; bit_idx++)
        {
            int read_bit_pos_global = bits_in_in_bytes + bit_idx;
            int read_byte_pos = read_bit_pos_global / 8;
            int read_bit_pos = read_bit_pos_global % 8;

            // Extract bit from in_bytes
            uint32_t bit_val = (in_bytes[read_byte_pos] >> read_bit_pos) & 0x1;

            // Set the corresponding bit in val.w
            val.w |= (bit_val << bit_idx);
        }

        out_fp32_array[i] = val;
        bits_in_in_bytes += FP__TF32_WIDTH;
    }
}

ipu__r_reg_t * ipu__get_mult_stage_r_reg(ipu__obj_t *ipu, inst_parser__mult_stage_reg_field_t mult_stage_idx)
{
    switch (mult_stage_idx)
    {
    case INST_PARSER__MULT_STAGE_REG_FIELD_R0:
        return &ipu->regfile.mult_stage_regfile.r_regs[0];
    case INST_PARSER__MULT_STAGE_REG_FIELD_R1:
        return &ipu->regfile.mult_stage_regfile.r_regs[1];
    case INST_PARSER__MULT_STAGE_REG_FIELD_MEM_BYPASS:
        return &ipu->misc.mem_bypass_r_reg;
    default:
        assert(0 && "Invalid mult stage reg field");
        return NULL;
    }
}