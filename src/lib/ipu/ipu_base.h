#ifndef IPU_BASE_H
#define IPU_BASE_H

#include <stdint.h>
#include "xmem/xmem.h"
#include "src/tools/ipu-as-py/inst_parser.h"

// IPU register and memory size definitions
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

// Helper functions for converting register field enums to indices
int ipu__get_r_from_r_enum(int r_enum_val);
int ipu__get_rd_from_r_enum(int r_enum_val);
int ipu__get_rq_from_r_enum(int r_enum_val);
int ipu__get_ro_from_r_enum(int r_enum_val);

// IPU initialization
ipu__obj_t *ipu__init_ipu();

#endif // IPU_BASE_H
