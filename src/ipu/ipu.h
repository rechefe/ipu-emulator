#ifndef IPU_H
#define IPU_H

#include <stdint.h>
#include "xmem/xmem.h"
#include "assert.h"

#define IPU__R_REG_SIZE_BYTES 128
#define IPU__WORD_SIZE_BYTES 4
#define IPU__R_REG_SIZE_WORDS (IPU__R_REG_SIZE_BYTES / IPU__WORD_SIZE_BYTES)

#define IPU__R_REGS_NUM 12
#define IPU__RQ_SIZE_IN_R_REGS 4
#define IPU__RQ_REGS_NUM (IPU__R_REGS_NUM / IPU__RQ_SIZE_IN_R_REGS)

#define IPU__RQ_REG_SIZE_BYTES (IPU__R_REG_SIZE_BYTES * IPU__RQ_SIZE_IN_R_REGS)
#define IPU__RQ_REG_SIZE_WORDS (IPU__RQ_REG_SIZE_BYTES / IPU__WORD_SIZE_BYTES)

// Register types
typedef struct
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
    ipu__regfile_t regfile;
    xmem__obj_t *xmem;
} ipu__obj_t;

ipu__obj_t *ipu__init_ipu();

void ipu__load_r_reg(ipu__obj_t *ipu, int index, int xmem_addr);
void ipu__store_r_reg(ipu__obj_t *ipu, int index, int xmem_addr);
void ipu__clear_reg(ipu__obj_t *ipu, int index);
void ipu__clear_rq_reg(ipu__obj_t *ipu, int index);
void ipu__mac_element_element(ipu__obj_t *ipu,
                              int rz, int rx, int ry,
                              ipu__data_type_t data_type);

void ipu__mac_element_vector(ipu__obj_t *ipu,
                             int rz, int rx, int ry,
                             int element_index,
                             ipu__data_type_t data_type);

uint32_t ipu__add(uint32_t a, uint32_t b, ipu__data_type_t data_type);
uint32_t ipu__mult(uint8_t a, uint8_t b, ipu__data_type_t data_type);

#endif // IPU_H
