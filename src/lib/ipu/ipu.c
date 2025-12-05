#include "ipu.h"
#include "logging/logger.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>

ipu__obj_t *ipu__init_ipu()
{
    ipu__obj_t *ipu = malloc(sizeof(ipu__obj_t));
    memset(&ipu->regfile, 0, sizeof(ipu__regfile_t));
    xmem__obj_t *xmem = xmem__initialize_xmem();
    ipu->xmem = xmem;
    ipu->program_counter = 0;
    memset(ipu->inst_mem, 0, sizeof(ipu->inst_mem)); // Initialize instruction mem
    return ipu;
}

int ipu__get_r_from_r_enum(int r_enum_val)
{
    assert(r_enum_val >= INST_PARSER__RX_REG_FIELD_R0 &&
           r_enum_val <= INST_PARSER__RX_REG_FIELD_R11);
    return r_enum_val - INST_PARSER__RX_REG_FIELD_R0;
}

int ipu__get_rd_from_r_enum(int r_enum_val)
{
    assert(r_enum_val >= INST_PARSER__RX_REG_FIELD_RD0 &&
           r_enum_val <= INST_PARSER__RX_REG_FIELD_RD10);
    return (r_enum_val - INST_PARSER__RX_REG_FIELD_RD0);
}

int ipu__get_rq_from_r_enum(int r_enum_val)
{
    assert(r_enum_val >= INST_PARSER__RX_REG_FIELD_RQ0 &&
           r_enum_val <= INST_PARSER__RX_REG_FIELD_RQ8);
    return r_enum_val - INST_PARSER__RX_REG_FIELD_RQ0;
}

int ipu__get_ro_from_r_enum(int r_enum_val)
{
    assert(r_enum_val == INST_PARSER__RX_REG_FIELD_RO0);
    return r_enum_val - INST_PARSER__RX_REG_FIELD_RO0;
}

inst_parser__inst_t ipu__fetch_current_instruction(ipu__obj_t *ipu)
{
    return ipu->inst_mem[ipu->program_counter];
}

static void ipu__execute_xmem_str(ipu__obj_t *ipu, inst_parser__inst_t inst, const ipu__regfile_t *regfile_snapshot)
{
    // Store R register to external memory: SR RX, [LR, CR]
    int rx_idx = ipu__get_r_from_r_enum(inst.mac_inst_token_1_rx_reg_field);
    int lr_idx = inst.xmem_inst_token_1_lr_reg_field;
    int cr_idx = inst.xmem_inst_token_2_cr_reg_field;
    ipu__store_r_reg(ipu, rx_idx, cr_idx, lr_idx, regfile_snapshot);
}

static void ipu__execute_xmem_ldr(ipu__obj_t *ipu, inst_parser__inst_t inst, const ipu__regfile_t *regfile_snapshot)
{
    // Load R register from external memory: LDR RX, [LR, CR]
    int rx_idx = ipu__get_r_from_r_enum(inst.mac_inst_token_1_rx_reg_field);
    int lr_idx = inst.xmem_inst_token_1_lr_reg_field;
    int cr_idx = inst.xmem_inst_token_2_cr_reg_field;
    ipu__load_r_reg(ipu, rx_idx, cr_idx, lr_idx, regfile_snapshot);
}

void ipu__execute_xmem_instruction(ipu__obj_t *ipu, inst_parser__inst_t inst, const ipu__regfile_t *regfile_snapshot)
{
    switch (inst.xmem_inst_token_0_xmem_inst_opcode)
    {
    case INST_PARSER__XMEM_INST_OPCODE_STR:
        ipu__execute_xmem_str(ipu, inst, regfile_snapshot);
        break;
    case INST_PARSER__XMEM_INST_OPCODE_LDR:
        ipu__execute_xmem_ldr(ipu, inst, regfile_snapshot);
        break;
    case INST_PARSER__XMEM_INST_OPCODE_XMEM_NOP:
        // No operation for XMEM
        break;
    default:
        assert(0 && "Unknown XMEM instruction opcode");
    }
}

static void ipu__execute_lr_set(ipu__obj_t *ipu, inst_parser__inst_t inst)
{
    // SET LR, immediate
    int lr_idx = inst.lr_inst_token_1_lr_reg_field;
    uint32_t imm = inst.lr_inst_token_2_lr_immediate_type;
    ipu__set_lr(ipu, lr_idx, imm);
}

static void ipu__execute_lr_incr(ipu__obj_t *ipu, inst_parser__inst_t inst, const ipu__regfile_t *regfile_snapshot)
{
    // INCR LR, immediate (add to LR) - read from snapshot
    int lr_idx = inst.lr_inst_token_1_lr_reg_field;
    uint32_t imm = inst.lr_inst_token_2_lr_immediate_type;
    uint32_t lr_value = regfile_snapshot->lr_regfile.lr[lr_idx];
    ipu->regfile.lr_regfile.lr[lr_idx] = lr_value + imm;
}

void ipu__execute_lr_instruction(ipu__obj_t *ipu, inst_parser__inst_t inst, const ipu__regfile_t *regfile_snapshot)
{
    switch (inst.lr_inst_token_0_lr_inst_opcode)
    {
    case INST_PARSER__LR_INST_OPCODE_SET:
        ipu__execute_lr_set(ipu, inst);
        break;
    case INST_PARSER__LR_INST_OPCODE_INCR:
        ipu__execute_lr_incr(ipu, inst, regfile_snapshot);
        break;
    default:
        assert(0 && "Unknown LR instruction opcode");
    }
}

static void ipu__execute_mac_ee(ipu__obj_t *ipu, inst_parser__inst_t inst, const ipu__regfile_t *regfile_snapshot)
{
    // MAC element-element: RQ[rz] += R[rx] * R[ry]
    int rq_dest = ipu__get_rq_from_r_enum(inst.mac_inst_token_1_rx_reg_field);
    int r_source_0 = ipu__get_r_from_r_enum(inst.mac_inst_token_2_rx_reg_field);
    int r_source_1 = ipu__get_r_from_r_enum(inst.mac_inst_token_3_rx_reg_field);
    // TODO: data_type should come from instruction or be configurable
    ipu__data_type_t data_type = IPU__DATA_TYPE_INT8;
    ipu__mac_element_element(ipu, rq_dest, r_source_0, r_source_1, data_type, regfile_snapshot);
}

static void ipu__execute_mac_ev(ipu__obj_t *ipu, inst_parser__inst_t inst, const ipu__regfile_t *regfile_snapshot)
{
    // MAC element-vector: RQ[rz][i] += R[rx][i] * R[ry][LR]
    int rq_dest = ipu__get_rq_from_r_enum(inst.mac_inst_token_1_rx_reg_field);
    int r_source_0 = ipu__get_r_from_r_enum(inst.mac_inst_token_2_rx_reg_field);
    int r_source_1 = ipu__get_r_from_r_enum(inst.mac_inst_token_3_rx_reg_field);
    int lr_idx = inst.mac_inst_token_4_lr_reg_field;

    ipu__data_type_t data_type = IPU__DATA_TYPE_INT8;
    ipu__mac_element_vector(ipu, rq_dest, r_source_0, r_source_1, lr_idx, data_type, regfile_snapshot);
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
    case INST_PARSER__MAC_INST_OPCODE_MAC_NOP:
        // No operation for MAC
        break;
    default:
        assert(0 && "Unknown MAC instruction opcode");
    }
}

void ipu__execute_next_instruction(ipu__obj_t *ipu)
{
    // Fetch instruction once at the start of the cycle
    inst_parser__inst_t inst = ipu__fetch_current_instruction(ipu);

    // Create a snapshot of the register file at the start of the cycle
    // All subinstructions read from this snapshot to avoid race conditions
    ipu__regfile_t regfile_snapshot = ipu->regfile;

    // Execute all subinstructions in parallel using the snapshot and fetched instruction
    ipu__execute_xmem_instruction(ipu, inst, &regfile_snapshot);
    ipu__execute_lr_instruction(ipu, inst, &regfile_snapshot);
    ipu__execute_mac_instruction(ipu, inst, &regfile_snapshot);
    ipu__execute_cond_instruction(ipu, inst, &regfile_snapshot);

    // Increment program counter if not modified by branch
    // Note: Branch instructions modify PC directly
    ipu->program_counter++;
}

static void ipu__execute_cond_beq(ipu__obj_t *ipu, uint32_t lr1, uint32_t lr2, uint32_t label)
{
    // Branch if equal: if LR1 == LR2 then PC = label
    if (lr1 == lr2)
    {
        ipu->program_counter = label;
    }
}

static void ipu__execute_cond_bne(ipu__obj_t *ipu, uint32_t lr1, uint32_t lr2, uint32_t label)
{
    // Branch if not equal: if LR1 != LR2 then PC = label
    if (lr1 != lr2)
    {
        ipu->program_counter = label;
    }
}

static void ipu__execute_cond_blt(ipu__obj_t *ipu, uint32_t lr1, uint32_t lr2, uint32_t label)
{
    // Branch if less than: if LR1 < LR2 then PC = label
    if (lr1 < lr2)
    {
        ipu->program_counter = label;
    }
}

static void ipu__execute_cond_bnz(ipu__obj_t *ipu, uint32_t lr1, uint32_t label)
{
    // Branch if not zero: if LR1 != 0 then PC = label
    if (lr1 != 0)
    {
        ipu->program_counter = label;
    }
}

static void ipu__execute_cond_bz(ipu__obj_t *ipu, uint32_t lr1, uint32_t label)
{
    // Branch if zero: if LR1 == 0 then PC = label
    if (lr1 == 0)
    {
        ipu->program_counter = label;
    }
}

static void ipu__execute_cond_b(ipu__obj_t *ipu, uint32_t label)
{
    // Unconditional branch: PC = label
    ipu->program_counter = label;
}

static void ipu__execute_cond_br(ipu__obj_t *ipu, uint32_t lr1)
{
    // Branch relative: PC = PC + LR1
    ipu->program_counter += lr1;
}

static void ipu__execute_cond_bkpt(ipu__obj_t *ipu)
{
    // Breakpoint - halt execution
    // Set PC to invalid value to stop
    ipu->program_counter = IPU__INST_MEM_SIZE;
    LOG_INFO("IPU breakpoint reached, halting execution.");
}

void ipu__execute_cond_instruction(ipu__obj_t *ipu, inst_parser__inst_t inst, const ipu__regfile_t *regfile_snapshot)
{
    int lr1_idx = inst.cond_inst_token_1_lr_reg_field;
    int lr2_idx = inst.cond_inst_token_2_lr_reg_field;
    // Read LR values from snapshot to avoid race conditions
    uint32_t lr1 = regfile_snapshot->lr_regfile.lr[lr1_idx];
    uint32_t lr2 = regfile_snapshot->lr_regfile.lr[lr2_idx];
    uint32_t label = inst.cond_inst_token_3_label_token;

    switch (inst.cond_inst_token_0_cond_inst_opcode)
    {
    case INST_PARSER__COND_INST_OPCODE_BEQ:
        ipu__execute_cond_beq(ipu, lr1, lr2, label);
        break;
    case INST_PARSER__COND_INST_OPCODE_BNE:
        ipu__execute_cond_bne(ipu, lr1, lr2, label);
        break;
    case INST_PARSER__COND_INST_OPCODE_BLT:
        ipu__execute_cond_blt(ipu, lr1, lr2, label);
        break;
    case INST_PARSER__COND_INST_OPCODE_BNZ:
        ipu__execute_cond_bnz(ipu, lr1, label);
        break;
    case INST_PARSER__COND_INST_OPCODE_BZ:
        ipu__execute_cond_bz(ipu, lr1, label);
        break;
    case INST_PARSER__COND_INST_OPCODE_B:
        ipu__execute_cond_b(ipu, label);
        break;
    case INST_PARSER__COND_INST_OPCODE_BR:
        ipu__execute_cond_br(ipu, lr1);
        break;
    case INST_PARSER__COND_INST_OPCODE_BKPT:
        ipu__execute_cond_bkpt(ipu);
        break;
    default:
        assert(0 && "Unknown COND instruction opcode");
    }
}

void ipu__load_inst_mem(ipu__obj_t *ipu, FILE *file)
{
    size_t inst_size = sizeof(inst_parser__inst_t);
    for (size_t i = 0; i < IPU__INST_MEM_SIZE; i++)
    {
        assert(fread(&ipu->inst_mem[i], 1, inst_size, file) == inst_size);
    }
}

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

static inline void ipu__mac_accumulate(ipu__obj_t *ipu, int rz, int i,
                                       uint8_t a, uint8_t b,
                                       ipu__data_type_t data_type,
                                       const ipu__regfile_t *regfile_snapshot)
{
    uint32_t product = ipu__mult(a, b, data_type);
    uint32_t acc = regfile_snapshot->rx_regfile.rq_regs[rz].words[i];
    uint32_t result = ipu__add(acc, product, data_type);
    ipu->regfile.rx_regfile.rq_regs[rz].words[i] = result;
}

void ipu__mac_element_element(ipu__obj_t *ipu,
                              int rz, int rx, int ry,
                              ipu__data_type_t data_type,
                              const ipu__regfile_t *regfile_snapshot)
{
    assert(rz >= 0 && rz < IPU__RQ_REGS_NUM);
    assert(rx >= 0 && rx < IPU__R_REGS_NUM);
    assert(ry >= 0 && ry < IPU__R_REGS_NUM);

    // Read operands from snapshot - rx and ry are R registers, not RQ
    for (int i = 0; i < IPU__R_REG_SIZE_BYTES; i++)
    {
        uint8_t a = regfile_snapshot->rx_regfile.r_regs[rx].bytes[i];
        uint8_t b = regfile_snapshot->rx_regfile.r_regs[ry].bytes[i];
        ipu__mac_accumulate(ipu, rz, i, a, b, data_type, regfile_snapshot);
    }
}

void ipu__mac_element_vector(ipu__obj_t *ipu,
                             int rq_dest, int r_source_0, int r_source_1,
                             int lr_idx,
                             ipu__data_type_t data_type,
                             const ipu__regfile_t *regfile_snapshot)
{
    assert(rq_dest >= 0 && rq_dest < IPU__RQ_REGS_NUM);
    assert(r_source_0 >= 0 && r_source_0 < IPU__R_REGS_NUM);
    assert(r_source_1 >= 0 && r_source_1 < IPU__R_REGS_NUM);
    assert(lr_idx >= 0 && lr_idx < IPU__LR_REGS_NUM);

    // The LR value is the element we choose for MAC from RY reg (read from snapshot)
    uint32_t element_index = regfile_snapshot->lr_regfile.lr[lr_idx];
    assert(element_index < IPU__R_REG_SIZE_BYTES);

    // Read operands from snapshot
    for (int i = 0; i < IPU__R_REG_SIZE_BYTES; i++)
    {
        uint8_t a = regfile_snapshot->rx_regfile.r_regs[r_source_0].bytes[i];
        uint8_t b = regfile_snapshot->rx_regfile.r_regs[r_source_1].bytes[element_index];
        ipu__mac_accumulate(ipu, rq_dest, i, a, b, data_type, regfile_snapshot);
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
    ipu__uint8_t_as_uint4_t_t a_as_two = (ipu__uint8_t_as_uint4_t_t)a;
    ipu__uint8_t_as_uint4_t_t b_as_two = (ipu__uint8_t_as_uint4_t_t)b;

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
    ipu__uint8_t_as_uint4_t_t a_as_two = (ipu__uint8_t_as_uint4_t_t)a;
    ipu__uint8_t_as_uint4_t_t b_as_two = (ipu__uint8_t_as_uint4_t_t)b;

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