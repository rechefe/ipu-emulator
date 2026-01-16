#include "ipu_base.h"
#include "xmem/xmem.h"
#include "logging/logger.h"
#include "ipu_math/ipu_math.h"
#include <stdlib.h>
#include <string.h>
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

void ipu__set_cr_dtype(ipu__obj_t *ipu, ipu_math__dtype_t dtype)
{
    // Validate that only supported data types are used
    if (dtype != IPU_MATH__DTYPE_INT8 &&
        dtype != IPU_MATH__DTYPE_FP8_E4M3 &&
        dtype != IPU_MATH__DTYPE_FP8_E5M2)
    {
        LOG_ERROR("Unsupported data type %d. Only INT8, FP8_E4M3, and FP8_E5M2 are supported.", dtype);
        assert(0 && "Unsupported data type for CR DTYPE register");
    }
    ipu->regfile.cr_regfile.cr[IPU__CR_DTYPE_REG] = (uint32_t)dtype;
}

ipu_math__dtype_t ipu__get_cr_dtype(const ipu__regfile_t *regfile)
{
    return (ipu_math__dtype_t)regfile->cr_regfile.cr[IPU__CR_DTYPE_REG];
}
