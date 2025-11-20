#include <gtest/gtest.h>
#include <cstring>

extern "C" {
#include "ipu/ipu.h"
}

TEST(IpuInit, CreatesXmemAndZeroedRegs)
{
    ipu__obj_t *ipu = ipu__init_ipu();
    ASSERT_NE(ipu, nullptr);
    ASSERT_NE(ipu->xmem, nullptr);

    // Check that registers are zeroed
    for (int i = 0; i < IPU__R_REGS_NUM; ++i)
    {
        for (int b = 0; b < IPU__R_REG_SIZE_BYTES; ++b)
        {
            EXPECT_EQ(ipu->regfile.r_regs[i].bytes[b], 0u);
        }
    }

    free(ipu->xmem);
    free(ipu);
}

TEST(IpuLoadStoreReg, RoundTripThroughXmem)
{
    ipu__obj_t *ipu = ipu__init_ipu();
    // fill r_reg 0 with pattern
    for (int i = 0; i < IPU__R_REG_SIZE_BYTES; ++i)
        ipu->regfile.r_regs[0].bytes[i] = (uint8_t)(i & 0xFF);

    const int addr = 1024;
    ipu__store_r_reg(ipu, 0, addr);

    // clear reg 1 and load into it from memory
    ipu__clear_reg(ipu, 1);
    ipu__load_r_reg(ipu, 1, addr);

    EXPECT_EQ(0, memcmp(&ipu->regfile.r_regs[0], &ipu->regfile.r_regs[1], IPU__R_REG_SIZE_BYTES));

    free(ipu->xmem);
    free(ipu);
}

TEST(IpuMacElementVector, ProducesExpectedAccumulation)
{
    ipu__obj_t *ipu = ipu__init_ipu();

    // Prepare rx and ry registers with simple data
    for (int i = 0; i < IPU__R_REG_SIZE_BYTES; ++i)
    {
        ipu->regfile.r_regs[2].bytes[i] = 2; // rx
        ipu->regfile.r_regs[3].bytes[i] = 3; // ry
    }

    // ensure rq reg zero
    ipu__clear_rq_reg(ipu, 0);

    ipu__mac_element_vector(ipu, 0, 2, 3, 0, IPU__DATA_TYPE_INT8);

    // each byte product = 2 * 3 = 6, accumulation initially 0
    for (int i = 0; i < IPU__R_REG_SIZE_BYTES; ++i)
    {
        EXPECT_EQ(ipu->regfile.rq_regs[0].words[i], 6u);
    }

    free(ipu->xmem);
    free(ipu);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
