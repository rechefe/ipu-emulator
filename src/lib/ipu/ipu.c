#include "ipu.h"
#include "logging/logger.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdbool.h>

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
    assert((r_enum_val >= INST_PARSER__RX_REG_FIELD_R0 &&
            r_enum_val <= INST_PARSER__RX_REG_FIELD_R11) ||
           r_enum_val == INST_PARSER__RX_REG_FIELD_MEM_BYPASS);
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
    // Store R register to external memory: STR RX, [LR, CR]
    int rx_idx = ipu__get_r_from_r_enum(inst.xmem_inst_token_1_rx_reg_field);

    // Currently stores R0 to XMEM at address LR + CR
    int lr_idx = inst.xmem_inst_token_2_lr_reg_field;
    int cr_idx = inst.xmem_inst_token_3_cr_reg_field;

    if (rx_idx != INST_PARSER__RX_REG_FIELD_MEM_BYPASS)
    {
        ipu__store_r_reg(ipu, rx_idx, cr_idx, lr_idx, regfile_snapshot);
    }
    else
    {
        assert(0 && "Storing from MEM_BYPASS register is not supported");
    }
}

static void ipu__execute_xmem_ldr(ipu__obj_t *ipu, inst_parser__inst_t inst, const ipu__regfile_t *regfile_snapshot)
{
    // Load R register from external memory: LDR RX, [LR, CR]
    int rx_idx = ipu__get_r_from_r_enum(inst.xmem_inst_token_1_rx_reg_field);

    // Currently loads into R0 from XMEM at address LR + CR
    int lr_idx = inst.xmem_inst_token_2_lr_reg_field;
    int cr_idx = inst.xmem_inst_token_3_cr_reg_field;
    if (rx_idx != INST_PARSER__RX_REG_FIELD_MEM_BYPASS)
    {
        ipu__load_r_reg(ipu, rx_idx, cr_idx, lr_idx, regfile_snapshot);
    }
    else
    {
        xmem__read_address(ipu->xmem,
                           regfile_snapshot->lr_regfile.lr[lr_idx] + regfile_snapshot->cr_regfile.cr[cr_idx],
                           (uint8_t *)&ipu->misc.mem_bypass_reg,
                           IPU__R_REG_SIZE_BYTES);
    }
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

// Structure to hold information about a single LR instruction
typedef struct
{
    bool valid;
    inst_parser__lr_inst_opcode_t opcode;
    int lr_idx;
    uint32_t immediate;
} ipu__lr_inst_info_t;

// Maximum number of LR instructions supported per cycle
#define IPU__MAX_LR_INSTS_PER_CYCLE 2

// Check if an LR instruction is valid (performs an actual register write)
static inline bool ipu__is_lr_inst_valid(inst_parser__lr_inst_opcode_t opcode, uint32_t immediate)
{
    // INCR by 0 is a NOP and doesn't count as writing to the register
    if (opcode == INST_PARSER__LR_INST_OPCODE_INCR && immediate == 0)
    {
        return false;
    }
    return true;
}

// Extract all LR instructions from an instruction word
static int ipu__extract_lr_instructions(inst_parser__inst_t inst, ipu__lr_inst_info_t *lr_insts)
{
    int count = 0;

    // LR instruction 0
    lr_insts[0].opcode = inst.lr_inst_0_token_0_lr_inst_opcode;
    lr_insts[0].lr_idx = inst.lr_inst_0_token_1_lr_reg_field;
    lr_insts[0].immediate = inst.lr_inst_0_token_2_lr_immediate_type;
    lr_insts[0].valid = ipu__is_lr_inst_valid(lr_insts[0].opcode, lr_insts[0].immediate);
    count++;

    // LR instruction 1
    lr_insts[1].opcode = inst.lr_inst_1_token_0_lr_inst_opcode;
    lr_insts[1].lr_idx = inst.lr_inst_1_token_1_lr_reg_field;
    lr_insts[1].immediate = inst.lr_inst_1_token_2_lr_immediate_type;
    lr_insts[1].valid = ipu__is_lr_inst_valid(lr_insts[1].opcode, lr_insts[1].immediate);
    count++;

    // Future expansion: Add more LR instructions here as needed
    // lr_insts[2].opcode = inst.lr_inst_2_token_0_lr_inst_opcode;
    // ...

    return count;
}

// Check for conflicts between LR instructions targeting the same register
static bool ipu__check_lr_conflicts(const ipu__lr_inst_info_t *lr_insts, int count)
{
    // Check if any two instructions write to the same LR register
    for (int i = 0; i < count; i++)
    {
        if (!lr_insts[i].valid)
            continue;

        for (int j = i + 1; j < count; j++)
        {
            if (!lr_insts[j].valid)
                continue;

            if (lr_insts[i].lr_idx == lr_insts[j].lr_idx)
            {
                LOG_ERROR("LR instruction conflict detected: LR%d written by multiple instructions in the same cycle",
                          lr_insts[i].lr_idx);
                return true; // Conflict detected
            }
        }
    }
    return false; // No conflicts
}

// Execute a single LR instruction operation
static void ipu__execute_single_lr_inst(ipu__obj_t *ipu,
                                        const ipu__lr_inst_info_t *lr_inst,
                                        const ipu__regfile_t *regfile_snapshot)
{
    switch (lr_inst->opcode)
    {
    case INST_PARSER__LR_INST_OPCODE_SET:
        // SET LR, immediate
        ipu__set_lr(ipu, lr_inst->lr_idx, lr_inst->immediate);
        break;
    case INST_PARSER__LR_INST_OPCODE_INCR:
        // INCR LR, immediate (add to LR) - read from snapshot
        {
            uint32_t lr_value = regfile_snapshot->lr_regfile.lr[lr_inst->lr_idx];
            ipu->regfile.lr_regfile.lr[lr_inst->lr_idx] = lr_value + lr_inst->immediate;
        }
        break;
    default:
        assert(0 && "Unknown LR instruction opcode");
    }
}

void ipu__execute_lr_instruction(ipu__obj_t *ipu, inst_parser__inst_t inst, const ipu__regfile_t *regfile_snapshot)
{
    // Extract all LR instructions from the instruction word
    ipu__lr_inst_info_t lr_insts[IPU__MAX_LR_INSTS_PER_CYCLE];
    int lr_inst_count = ipu__extract_lr_instructions(inst, lr_insts);

    // Check for conflicts between LR instructions
    if (ipu__check_lr_conflicts(lr_insts, lr_inst_count))
    {
        // Conflict detected - halt execution or handle as needed
        assert(0 && "Cannot execute LR instructions with register conflicts");
        return;
    }

    // Execute all LR instructions in parallel (no conflicts)
    for (int i = 0; i < lr_inst_count; i++)
    {
        if (lr_insts[i].valid)
        {
            ipu__execute_single_lr_inst(ipu, &lr_insts[i], regfile_snapshot);
        }
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
    ipu__mac_element_element(ipu, r_source_0, r_source_1, data_type, regfile_snapshot, &ipu->regfile.rx_regfile.rq_regs[rq_dest]);
}

static void ipu__execute_mac_ev(ipu__obj_t *ipu, inst_parser__inst_t inst, const ipu__regfile_t *regfile_snapshot)
{
    // MAC element-vector: RQ[rz][i] += R[rx][i] * R[ry][LR]
    int rq_dest = ipu__get_rq_from_r_enum(inst.mac_inst_token_1_rx_reg_field);

    int r_source_0 = ipu__get_r_from_r_enum(inst.mac_inst_token_2_rx_reg_field);
    int r_source_1 = ipu__get_r_from_r_enum(inst.mac_inst_token_3_rx_reg_field);
    int lr_idx = inst.mac_inst_token_4_lr_reg_field;

    ipu__data_type_t data_type = IPU__DATA_TYPE_INT8;
    ipu__mac_element_vector(ipu, r_source_0, r_source_1, lr_idx, data_type, regfile_snapshot, &ipu->regfile.rx_regfile.rq_regs[rq_dest]);
}

static void ipu__execute_mac_agg(ipu__obj_t *ipu, inst_parser__inst_t inst, const ipu__regfile_t *regfile_snapshot)
{
    int rq_dest = ipu__get_rq_from_r_enum(inst.mac_inst_token_1_rx_reg_field);

    int r_source_0 = ipu__get_r_from_r_enum(inst.mac_inst_token_2_rx_reg_field);
    int r_source_1 = ipu__get_r_from_r_enum(inst.mac_inst_token_3_rx_reg_field);

    int lr_idx = inst.mac_inst_token_4_lr_reg_field;

    ipu__r_reg_t r_source_0_value;
    ipu__r_reg_t r_source_1_value;

    ipu__get_r_register_for_mac_op(ipu, r_source_0, regfile_snapshot, &r_source_0_value);
    ipu__get_r_register_for_mac_op(ipu, r_source_1, regfile_snapshot, &r_source_1_value);

    ipu__rq_reg_t r_mult_result;
    memset(&r_mult_result, 0, sizeof(ipu__rq_reg_t));

    // Multiply R source registers element-wise
    ipu__mac_element_element(ipu, r_source_0, r_source_1, IPU__DATA_TYPE_INT8, regfile_snapshot, (ipu__rq_reg_t *)&r_mult_result);

    uint32_t sum = 0;

    for (int i = 0; i < IPU__RQ_REG_SIZE_WORDS; i++)
    {
        sum = ipu__add(sum, r_mult_result.words[i], IPU__DATA_TYPE_INT8);
    }

    uint32_t rq_store_idx = regfile_snapshot->lr_regfile.lr[lr_idx];
    assert(rq_store_idx < IPU__RQ_REG_SIZE_WORDS);

    ipu->regfile.rx_regfile.rq_regs[rq_dest].words[rq_store_idx] = sum;
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
    else
    {
        ipu->program_counter += 1;
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
    size_t i;
    for (i = 0; i < IPU__INST_MEM_SIZE; i++)
    {
        size_t bytes_read = fread(&ipu->inst_mem[i], 1, inst_size, file);
        if (bytes_read == 0)
        {
            // Reached end of file
            break;
        }
        if (bytes_read != inst_size)
        {
            // Partial read - error
            assert(0 && "Partial instruction read from file");
        }
    }
    // Fill remaining instructions with zeros (NOP)
    for (; i < IPU__INST_MEM_SIZE; i++)
    {
        memset(&ipu->inst_mem[i], 0, inst_size);
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

static inline void ipu__mac_accumulate(ipu__obj_t *ipu, int i,
                                       uint8_t a, uint8_t b,
                                       ipu__data_type_t data_type,
                                       ipu__rq_reg_t *out_rq_reg)
{
    (void)ipu; // Unused parameter
    if (i >= IPU__RQ_REG_SIZE_WORDS)
    {
        LOG_ERROR("MAC accumulate index out of bounds: i=%d, max=%d", i, IPU__RQ_REG_SIZE_WORDS);
        assert(0 && "MAC accumulate index out of bounds");
    }
    uint32_t product = ipu__mult(a, b, data_type);
    uint32_t acc = out_rq_reg->words[i];
    uint32_t result = ipu__add(acc, product, data_type);
    out_rq_reg->words[i] = result;
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

void ipu__mac_element_element(ipu__obj_t *ipu,
                              int rx, int ry,
                              ipu__data_type_t data_type,
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
                             ipu__data_type_t data_type,
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

uint32_t ipu__add_twice_uint4_t(uint8_t a, uint8_t b)
{
    ipu__uint8_t_as_uint4_t_t a_as_two = (ipu__uint8_t_as_uint4_t_t){.w = a};
    ipu__uint8_t_as_uint4_t_t b_as_two = (ipu__uint8_t_as_uint4_t_t){.w = b};

    uint16_t res_low = (a_as_two.f.low + b_as_two.f.low);
    uint16_t res_high = (b_as_two.f.high + b_as_two.f.high);
    return (res_high << IPU__UINT16T_BITS) | res_low;
}

uint32_t ipu__add_uint8_sign_extended(uint8_t a, uint8_t b, ipu__data_type_t data_type)
{
    // Sign extend uint8_t to int32_t
    int32_t a_extended = (int8_t)a;
    int32_t b_extended = (int8_t)b;

    // Convert to uint32_t and call ipu__add
    uint32_t a_as_uint32 = *(uint32_t *)&a_extended;
    uint32_t b_as_uint32 = *(uint32_t *)&b_extended;

    return ipu__add(a_as_uint32, b_as_uint32, data_type);
}

uint32_t ipu__add(uint32_t a, uint32_t b, ipu__data_type_t data_type)
{
    switch (data_type)
    {
    case IPU__DATA_TYPE_INT8:
    {
        // Treat uint32_t as int32_t during addition
        int32_t a_signed = *(int32_t *)&a;
        int32_t b_signed = *(int32_t *)&b;
        int32_t result = a_signed + b_signed;
        return *(uint32_t *)&result;
    }
    case IPU__DATA_TYPE_INT4:
    {
        // Treat uint32_t as int32_t during addition
        int32_t a_signed = *(int32_t *)&a;
        int32_t b_signed = *(int32_t *)&b;
        int32_t result_temp = ipu__add_twice_uint4_t(a_signed, b_signed);
        return *(uint32_t *)&result_temp;
    }
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
    ipu__uint8_t_as_uint4_t_t a_as_two = (ipu__uint8_t_as_uint4_t_t){.w = a};
    ipu__uint8_t_as_uint4_t_t b_as_two = (ipu__uint8_t_as_uint4_t_t){.w = b};

    uint16_t res_low = (a_as_two.f.low * b_as_two.f.low);
    uint16_t res_high = (b_as_two.f.high * b_as_two.f.high);
    return (res_high << IPU__UINT16T_BITS) | res_low;
}

uint32_t ipu__mult(uint8_t a, uint8_t b, ipu__data_type_t data_type)
{
    switch (data_type)
    {
    case IPU__DATA_TYPE_INT8:
    {
        // Treat uint8_t as int8_t (signed) for multiplication
        int8_t a_signed = (int8_t)a;
        int8_t b_signed = (int8_t)b;
        int32_t result = (int32_t)a_signed * (int32_t)b_signed;
        return *(uint32_t *)&result;
    }
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