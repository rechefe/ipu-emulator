#ifndef IPU_H
#define IPU_H

#include <stdint.h>
#include "xmem/xmem.h"
#include "assert.h"
#include "src/tools/ipu-as-py/inst_parser.h"

#define IPU__R_REG_SIZE_BYTES 128
#define IPU__WORD_SIZE_BYTES 4
#define IPU__R_REG_SIZE_WORDS (IPU__R_REG_SIZE_BYTES / IPU__WORD_SIZE_BYTES)

#define IPU__R_REGS_NUM 12

#define IPU__RD_SIZE_IN_R_REGS 2
#define IPU__RD_REGS_NUM (IPU__R_REGS_NUM / IPU__RD_SIZE_IN_R_REGS) 

#define IPU__RQ_SIZE_IN_R_REGS 4
#define IPU__RQ_REGS_NUM (IPU__R_REGS_NUM / IPU__RQ_SIZE_IN_R_REGS)

#define IPU__LR_REGS_NUM 16
#define IPU__CR_REGS_NUM 16

#define IPU__RD_REG_SIZE_BYTES (IPU__R_REG_SIZE_BYTES * IPU__RD_SIZE_IN_R_REGS)
#define IPU__RD_REG_SIZE_WORDS (IPU__RD_REG_SIZE_BYTES / IPU__WORD_SIZE_BYTES)

#define IPU__RQ_REG_SIZE_BYTES (IPU__R_REG_SIZE_BYTES * IPU__RQ_SIZE_IN_R_REGS)
#define IPU__RQ_REG_SIZE_WORDS (IPU__RQ_REG_SIZE_BYTES / IPU__WORD_SIZE_BYTES)

// Data type parameters
#define IPU__UINT4T_BITS 4
#define IPU__UINT4T_MASK ((1 << IPU__UINT4T_BITS) - 1)

#define IPU__UINT16T_BITS 16

#define IPU__INST_MEM_SIZE 1024

typedef union
{
    struct
    {
        uint8_t low : 4;
        uint8_t high : 4;
    } f;
    uint8_t w;
} ipu__uint8_t_as_uint4_t_t;

// Register types
typedef union
{
    uint8_t bytes[IPU__R_REG_SIZE_BYTES];
    uint32_t words[IPU__R_REG_SIZE_WORDS];
} ipu__r_reg_t;

typedef union
{
    uint8_t bytes[IPU__RQ_REG_SIZE_BYTES];
    uint32_t words[IPU__RQ_REG_SIZE_WORDS];
} ipu__rq_reg_t;

// IPU register file
typedef union
{
    ipu__r_reg_t r_regs[IPU__R_REGS_NUM];
    ipu__rq_reg_t rq_regs[IPU__RQ_REGS_NUM];
} ipu__rx_regfile_t;

typedef struct
{
    uint32_t lr[IPU__LR_REGS_NUM];
} ipu__lr_regfile_t;

typedef struct
{
    uint32_t cr[IPU__CR_REGS_NUM];
} ipu__cr_regfile_t;

typedef struct
{
    ipu__rx_regfile_t rx_regfile;
    ipu__lr_regfile_t lr_regfile;
    ipu__cr_regfile_t cr_regfile;
} ipu__regfile_t;

typedef enum
{
    IPU__DATA_TYPE_INT8,
    IPU__DATA_TYPE_INT4,
    IPU__DATA_TYPE_FP16,
    IPU__DATA_TYPE_FP8,
    IPU__DATA_TYPE_FP4,
    IPU__DATA_TYPE_BIN
} ipu__data_type_t;

typedef struct
{
    ipu__r_reg_t mem_bypass_reg;
} ipu__misc_t;

typedef struct
{
    ipu__regfile_t regfile;
    ipu__misc_t misc;
    uint32_t program_counter;
    xmem__obj_t *xmem;
    inst_parser__inst_t inst_mem[IPU__INST_MEM_SIZE];
} ipu__obj_t;

ipu__obj_t *ipu__init_ipu();

// Instruction execution functions
inst_parser__inst_t ipu__fetch_current_instruction(ipu__obj_t *ipu);
void ipu__execute_next_instruction(ipu__obj_t *ipu);
void ipu__execute_xmem_instruction(ipu__obj_t *ipu, inst_parser__inst_t inst, const ipu__regfile_t *regfile_snapshot);
void ipu__execute_lr_instruction(ipu__obj_t *ipu, inst_parser__inst_t inst, const ipu__regfile_t *regfile_snapshot);
void ipu__execute_mac_instruction(ipu__obj_t *ipu, inst_parser__inst_t inst, const ipu__regfile_t *regfile_snapshot);
void ipu__execute_cond_instruction(ipu__obj_t *ipu, inst_parser__inst_t inst, const ipu__regfile_t *regfile_snapshot);

// Instruction memory management
void ipu__load_inst_mem(ipu__obj_t *ipu, FILE *file);

// R registers instructions
void ipu__load_r_reg(ipu__obj_t *ipu, int rx_idx, int cr_idx, int lr_idx, const ipu__regfile_t *regfile_snapshot);
void ipu__store_r_reg(ipu__obj_t *ipu, int rx_idx, int cr_idx, int lr_idx, const ipu__regfile_t *regfile_snapshot);

void ipu__clear_reg(ipu__obj_t *ipu, int rx_idx);
void ipu__clear_rq_reg(ipu__obj_t *ipu, int rq_idx);

// MAC instructions
void ipu__get_r_register_for_mac_op(ipu__obj_t *ipu,
                                    int r_reg_index,
                                    const ipu__regfile_t *regfile_snapshot,
                                    ipu__r_reg_t *out_r_reg);

void ipu__mac_element_element(ipu__obj_t *ipu,
                              int rx, int ry,
                              ipu__data_type_t data_type,
                              const ipu__regfile_t *regfile_snapshot,
                              ipu__rq_reg_t *out_rq_reg);

/**
 * @brief This function makes the IPU do a MAC operation
 * while taking only one byte of the R[ry] reg -
 * RQ[rz][i] += (R[rx][i] * R[ry][LR[lr_idx]])
 * @param r_source_0 the index of the first R register
 * @param r_source_1 the index of the second R register
 * @param lr_idx - selects the LR - according to its value we
 * choose which byte to read from R[ry]
 * @param data_type - which data type to use
 * @param regfile_snapshot - snapshot of register file
 * @param out_rq_reg - output RQ register
 * @return none
 */
void ipu__mac_element_vector(ipu__obj_t *ipu,
                             int r_source_0, int r_source_1,
                             int lr_idx,
                             ipu__data_type_t data_type,
                             const ipu__regfile_t *regfile_snapshot,
                             ipu__rq_reg_t *out_rq_reg);

// Helper functions - for add and multiply
uint32_t ipu__add(uint32_t a, uint32_t b, ipu__data_type_t data_type);
uint32_t ipu__mult(uint8_t a, uint8_t b, ipu__data_type_t data_type);

// LR registers instructions
void ipu__set_lr(ipu__obj_t *ipu, int lr_idx, uint32_t imm);
void ipu__add_lr(ipu__obj_t *ipu, int lr_idx, uint32_t imm);

// CR registers instructions
void ipu__set_cr(ipu__obj_t *ipu, int cr_idx, uint32_t imm);

#endif // IPU_H
