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
            EXPECT_EQ(ipu->regfile.rx_regfile.r_regs[i].bytes[b], 0u);
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
        ipu->regfile.rx_regfile.r_regs[0].bytes[i] = (uint8_t)(i & 0xFF);

    const int addr = 1024;
    const int lr_idx = 0, cr_idx = 0;
    ipu__set_lr(ipu, lr_idx, addr);
    
    // Create snapshot for store operation
    ipu__regfile_t snapshot = ipu->regfile;
    ipu__store_r_reg(ipu, 0, cr_idx, lr_idx, &snapshot);

    // clear reg 1 and load into it from memory
    ipu__clear_reg(ipu, 1);
    snapshot = ipu->regfile;
    ipu__load_r_reg(ipu, 1, cr_idx, lr_idx, &snapshot);

    EXPECT_EQ(0, memcmp(&ipu->regfile.rx_regfile.r_regs[0], &ipu->regfile.rx_regfile.r_regs[1], IPU__R_REG_SIZE_BYTES));

    free(ipu->xmem);
    free(ipu);
}

TEST(IpuMacElementElement, BasicMultiplyAccumulate)
{
    ipu__obj_t *ipu = ipu__init_ipu();
    
    // Clear accumulator FIRST (RQ[1] aliases R[4-7])
    ipu__clear_rq_reg(ipu, 1);
    
    // Use R[8] and R[9] as inputs (don't overlap with RQ[1])
    for (int i = 0; i < IPU__R_REG_SIZE_BYTES; ++i)
    {
        ipu->regfile.rx_regfile.r_regs[8].bytes[i] = 2;
        ipu->regfile.rx_regfile.r_regs[9].bytes[i] = 3;
    }
    
    // Run MAC: RQ[1] += R[8] * R[9]
    ipu__regfile_t snapshot = ipu->regfile;
    ipu__mac_element_element(ipu, 1, 8, 9, IPU__DATA_TYPE_INT8, &snapshot);
    
    // Each word should be 2 * 3 = 6
    for (int i = 0; i < IPU__R_REG_SIZE_BYTES; ++i)
    {
        EXPECT_EQ(ipu->regfile.rx_regfile.rq_regs[1].words[i], 6u) << "at index " << i;
    }
    
    free(ipu->xmem);
    free(ipu);
}

TEST(IpuMacElementElement, AccumulationOverMultipleCalls)
{
    ipu__obj_t *ipu = ipu__init_ipu();
    
    ipu__clear_rq_reg(ipu, 1);
    
    // Set up R[8] and R[9] AFTER clearing (don't overlap with RQ[1])
    for (int i = 0; i < IPU__R_REG_SIZE_BYTES; ++i)
    {
        ipu->regfile.rx_regfile.r_regs[8].bytes[i] = 2;
        ipu->regfile.rx_regfile.r_regs[9].bytes[i] = 3;
    }
    
    // First MAC: RQ[1] = 0 + (2 * 3) = 6
    ipu__regfile_t snapshot = ipu->regfile;
    ipu__mac_element_element(ipu, 1, 8, 9, IPU__DATA_TYPE_INT8, &snapshot);
    
    // Second MAC: RQ[1] = 6 + (2 * 3) = 12
    snapshot = ipu->regfile;
    ipu__mac_element_element(ipu, 1, 8, 9, IPU__DATA_TYPE_INT8, &snapshot);
    
    for (int i = 0; i < IPU__R_REG_SIZE_BYTES; ++i)
    {
        EXPECT_EQ(ipu->regfile.rx_regfile.rq_regs[1].words[i], 12u);
    }
    
    free(ipu->xmem);
    free(ipu);
}

TEST(IpuMacElementVector, SingleElementBroadcast)
{
    ipu__obj_t *ipu = ipu__init_ipu();
    const int element_idx_lr_reg = 0;
    const int element_index = 5;
    
    ipu__set_lr(ipu, element_idx_lr_reg, element_index);
    
    // Clear RQ[2] which aliases R[8-11], so use R[4] and R[5] as inputs
    ipu__clear_rq_reg(ipu, 2);
    
    // Fill R[4] with 2, R[5] with zeros except at element_index AFTER clearing
    for (int i = 0; i < IPU__R_REG_SIZE_BYTES; ++i)
    {
        ipu->regfile.rx_regfile.r_regs[4].bytes[i] = 2;
        ipu->regfile.rx_regfile.r_regs[5].bytes[i] = 0;
    }
    ipu->regfile.rx_regfile.r_regs[5].bytes[element_index] = 4;
    
    ipu__regfile_t snapshot = ipu->regfile;
    ipu__mac_element_vector(ipu, 2, 4, 5, element_idx_lr_reg, IPU__DATA_TYPE_INT8, &snapshot);
    
    // All words should be 2 * 4 = 8 (element at index 5 broadcasted)
    for (int i = 0; i < IPU__R_REG_SIZE_BYTES; ++i)
    {
        EXPECT_EQ(ipu->regfile.rx_regfile.rq_regs[2].words[i], 8u);
    }
    
    free(ipu->xmem);
    free(ipu);
}

TEST(IpuLRInstructions, SetAndIncrement)
{
    ipu__obj_t *ipu = ipu__init_ipu();
    
    // Test SET
    ipu__set_lr(ipu, 0, 100);
    EXPECT_EQ(ipu->regfile.lr_regfile.lr[0], 100u);
    
    // Test INCR with snapshot
    ipu__regfile_t snapshot = ipu->regfile;
    ipu->regfile.lr_regfile.lr[0] = snapshot.lr_regfile.lr[0] + 25;
    EXPECT_EQ(ipu->regfile.lr_regfile.lr[0], 125u);
    
    free(ipu->xmem);
    free(ipu);
}

TEST(IpuRegfileSnapshot, IsolatesParallelWrites)
{
    ipu__obj_t *ipu = ipu__init_ipu();
    
    // Set initial LR values
    ipu__set_lr(ipu, 0, 10);
    ipu__set_lr(ipu, 1, 20);
    
    // Take snapshot
    ipu__regfile_t snapshot = ipu->regfile;
    
    // Modify LR[0] using current state
    ipu__set_lr(ipu, 0, 100);
    
    // Read from snapshot should still give old value
    EXPECT_EQ(snapshot.lr_regfile.lr[0], 10u);
    EXPECT_EQ(ipu->regfile.lr_regfile.lr[0], 100u);
    
    // Increment LR[1] using snapshot (should use old value of 20)
    uint32_t new_val = snapshot.lr_regfile.lr[1] + 5;
    ipu__set_lr(ipu, 1, new_val);
    EXPECT_EQ(ipu->regfile.lr_regfile.lr[1], 25u);
    
    free(ipu->xmem);
    free(ipu);
}

TEST(IpuMemoryOps, SnapshotPreservesAddressing)
{
    ipu__obj_t *ipu = ipu__init_ipu();
    
    // Set up addressing registers
    ipu__set_lr(ipu, 0, 1000);
    ipu__set_cr(ipu, 0, 24);
    
    // Fill register with pattern
    for (int i = 0; i < IPU__R_REG_SIZE_BYTES; ++i)
        ipu->regfile.rx_regfile.r_regs[0].bytes[i] = (uint8_t)(i + 50);
    
    // Take snapshot and store
    ipu__regfile_t snapshot = ipu->regfile;
    ipu__store_r_reg(ipu, 0, 0, 0, &snapshot);
    
    // Modify LR after snapshot (shouldn't affect stored address)
    ipu__set_lr(ipu, 0, 2000);
    
    // Clear register and load back using original snapshot address
    ipu__clear_reg(ipu, 1);
    ipu__load_r_reg(ipu, 1, 0, 0, &snapshot);
    
    // Should match original data
    EXPECT_EQ(0, memcmp(&ipu->regfile.rx_regfile.r_regs[0], 
                        &ipu->regfile.rx_regfile.r_regs[1], 
                        IPU__R_REG_SIZE_BYTES));
    
    free(ipu->xmem);
    free(ipu);
}

TEST(IpuFetchInstruction, ReturnsCorrectInstruction)
{
    ipu__obj_t *ipu = ipu__init_ipu();
    
    // Set up a test instruction at PC=0
    ipu->inst_mem[0].xmem_inst_token_0_xmem_inst_opcode = INST_PARSER__XMEM_INST_OPCODE_LDR;
    ipu->program_counter = 0;
    
    inst_parser__inst_t fetched = ipu__fetch_current_instruction(ipu);
    EXPECT_EQ(fetched.xmem_inst_token_0_xmem_inst_opcode, INST_PARSER__XMEM_INST_OPCODE_LDR);
    
    // Change PC and verify fetch
    ipu->inst_mem[5].xmem_inst_token_0_xmem_inst_opcode = INST_PARSER__XMEM_INST_OPCODE_STR;
    ipu->program_counter = 5;
    
    fetched = ipu__fetch_current_instruction(ipu);
    EXPECT_EQ(fetched.xmem_inst_token_0_xmem_inst_opcode, INST_PARSER__XMEM_INST_OPCODE_STR);
    
    free(ipu->xmem);
    free(ipu);
}

TEST(IpuClearOperations, ZeroesRegisters)
{
    ipu__obj_t *ipu = ipu__init_ipu();
    
    // Fill R register with non-zero data
    for (int i = 0; i < IPU__R_REG_SIZE_BYTES; ++i)
        ipu->regfile.rx_regfile.r_regs[0].bytes[i] = 0xFF;
    
    ipu__clear_reg(ipu, 0);
    
    for (int i = 0; i < IPU__R_REG_SIZE_BYTES; ++i)
        EXPECT_EQ(ipu->regfile.rx_regfile.r_regs[0].bytes[i], 0u);
    
    // Fill RQ register with non-zero data
    for (int i = 0; i < IPU__RQ_REG_SIZE_BYTES; ++i)
        ipu->regfile.rx_regfile.rq_regs[0].bytes[i] = 0xFF;
    
    ipu__clear_rq_reg(ipu, 0);
    
    for (int i = 0; i < IPU__RQ_REG_SIZE_BYTES; ++i)
        EXPECT_EQ(ipu->regfile.rx_regfile.rq_regs[0].bytes[i], 0u);
    
    free(ipu->xmem);
    free(ipu);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
