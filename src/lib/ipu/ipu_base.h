#ifndef IPU_BASE_H
#define IPU_BASE_H

#include <stdint.h>
#include "xmem/xmem.h"
#include "ipu_math/ipu_math.h"
#include "src/tools/ipu-as-py/inst_parser.h"

// IPU register and memory size definitions
#define IPU__R_REG_SIZE_BYTES 128

// Mult regfile size definitions
#define IPU__MULT_STAGES_REGFILE_NUM_OF_R_REGS 2
#define IPU__R_CYCLIC_REG_SIZE_BYTES 512

// Accumulator regfile size definitions
#define IPU__R_ACC_TF32_VEC_SIZE_BYTES 304
#define IPU__R_ACC_TF32_VEC_NUM 2
#define IPU__R_ACC_REG_SIZE_BYTES (IPU__R_ACC_TF32_VEC_SIZE_BYTES * IPU__R_ACC_TF32_VEC_NUM)
#define IPU__RT_FROM_R_ACC_SIZE_BYTES 512

#define IPU__LR_REGS_NUM 16
#define IPU__CR_REGS_NUM 16

// CR Register indices
#define IPU__CR_DTYPE_REG 15 // CR[15] holds the data type for MAC operations

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
// IPU register file

typedef struct
{
    uint8_t bytes[IPU__R_REG_SIZE_BYTES];
} ipu__r_reg_t;

typedef struct
{
    uint8_t bytes[IPU__R_CYCLIC_REG_SIZE_BYTES];
} ipu__r_cyclic_reg_t;

typedef struct
{
    ipu__r_reg_t r_regs[IPU__MULT_STAGES_REGFILE_NUM_OF_R_REGS];
    ipu__r_cyclic_reg_t r_cyclic_reg;
} ipu__mult_stage_regfile_t;

typedef struct
{
    uint8_t bytes[IPU__R_ACC_TF32_VEC_SIZE_BYTES];
} ipu__r_acc_tf32_vec_t;

typedef struct
{
    uint8_t bytes[IPU__RT_FROM_R_ACC_SIZE_BYTES];
    uint32_t words[IPU__RT_FROM_R_ACC_SIZE_BYTES / 4];
} ipu__rt_from_r_acc_t;

typedef union
{
    uint8_t bytes[IPU__R_ACC_REG_SIZE_BYTES];
    uint32_t words[IPU__R_ACC_REG_SIZE_BYTES / 4];
    uint8_t tf32_vecs[IPU__R_ACC_TF32_VEC_NUM][IPU__R_ACC_TF32_VEC_SIZE_BYTES];
} ipu__r_acc_reg_t;

typedef struct
{
    ipu__r_reg_t r_mask;
    ipu__r_acc_reg_t r_acc;
} ipu__acc_stage_regfile_t;

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
    // Placeholder for new regfile
    ipu__mult_stage_regfile_t mult_stage_regfile;
    ipu__lr_regfile_t lr_regfile;
    ipu__cr_regfile_t cr_regfile;
} ipu__regfile_t;

typedef struct
{
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

// CR register accessors
void ipu__set_cr_dtype(ipu__obj_t *ipu, ipu_math__dtype_t dtype);
ipu_math__dtype_t ipu__get_cr_dtype(const ipu__regfile_t *regfile);

// IPU initialization
ipu__obj_t *ipu__init_ipu();

#endif // IPU_BASE_H
