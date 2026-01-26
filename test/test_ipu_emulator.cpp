#include "ipu_test_helper.h"

using namespace ipu_test;

// ============================================================================
// Basic Register Operations Tests
// ============================================================================

/**
 * @brief Test setting and reading LR registers via assembly
 */
TEST_F(IpuEmulatorTest, RegisterOperations_SetLrRegister)
{
    std::string asm_code = R"(
set lr13 0x1000;;
bkpt;;
)";

    ASSERT_TRUE(helper.LoadProgramFromAssembly(asm_code));
    int cycles = helper.Run();

    ASSERT_GT(cycles, 0) << "Program should execute successfully";
    EXPECT_EQ(helper.GetLr(13), 0x1000);
}

/**
 * @brief Test incrementing LR register
 */
TEST_F(IpuEmulatorTest, RegisterOperations_IncrementLrRegister)
{
    std::string asm_code = R"(
set lr11 10;;
incr lr11 5;;
incr lr11 3;;
bkpt;;
)";

    ASSERT_TRUE(helper.LoadProgramFromAssembly(asm_code));
    helper.Run();

    EXPECT_EQ(helper.GetLr(11), 18); // 10 + 5 + 3
}

/**
 * @brief Test direct register access via helper methods
 */
TEST_F(IpuEmulatorTest, RegisterOperations_DirectAccess)
{
    // Set LR registers directly
    helper.SetLr(11, 0xDEADBEEF);
    helper.SetLr(5, 0x12345678);

    EXPECT_EQ(helper.GetLr(11), 0xDEADBEEF);
    EXPECT_EQ(helper.GetLr(5), 0x12345678);

    // Set CR registers directly
    helper.SetCr(0, 0xABCDEF00);
    EXPECT_EQ(helper.GetCr(0), 0xABCDEF00);
}

// ============================================================================
// Memory Operations Tests
// ============================================================================

/**
 * @brief Test loading data from external memory
 */
TEST_F(IpuEmulatorTest, Memory_LoadFromMemory)
{
    // Prepare test data in memory
    std::vector<uint8_t> test_data(IPU__R_REG_SIZE_BYTES);
    for (int i = 0; i < IPU__R_REG_SIZE_BYTES; i++)
    {
        test_data[i] = i & 0xFF;
    }
    helper.WriteXmem(0x1000, test_data);

    std::string asm_code = R"(
set lr13 0x1000;;
ldr_mult_reg r1 lr13 cr0;;
bkpt;;
)";

    ASSERT_TRUE(helper.LoadProgramFromAssembly(asm_code));
    helper.Run();

    // Verify the address register was set correctly
    EXPECT_EQ(helper.GetLr(13), 0x1000);

    // Verify that r1 received the correct data from memory
    auto r1_data = helper.GetRBytes(1, 0, IPU__R_REG_SIZE_BYTES);
    for (int i = 0; i < IPU__R_REG_SIZE_BYTES; i++)
    {
        EXPECT_EQ(r1_data[i], test_data[i])
            << "r1 byte " << i << " should match test data";
    }
}

/**
 * @brief Test storing data to memory
 */
TEST_F(IpuEmulatorTest, Memory_StoreToMemory)
{
    // First, load some data and perform operations to populate the accumulator
    std::vector<uint8_t> r1_data(IPU__R_REG_SIZE_BYTES, 2);            // Fill r1 with 2s
    std::vector<uint8_t> cyclic_data(IPU__R_CYCLIC_REG_SIZE_BYTES, 3); // Fill cyclic with 3s

    helper.WriteXmem(0x1000, r1_data);
    helper.WriteXmem(0x2000, cyclic_data);

    std::string asm_code = R"(
# Load r1 with data (all 2s)
set lr13 0x1000;;
ldr_mult_reg r1 lr13 cr0;;

# Load cyclic register with data (all 3s)
set lr14 0x2000;;
set lr15 0;;
ldr_cyclic_mult_reg lr14 cr0 lr15;;

# Reset accumulator
reset_acc;;

# Perform element-wise multiplication: 2 * 3 = 6 per element
mult.ee r1 lr0 lr0 lr0;
acc;;
# Store accumulator result to memory (512 bytes)

set lr0 0x3000;;
str_acc_reg lr0 cr0;;

bkpt;;
)";

    ASSERT_TRUE(helper.LoadProgramFromAssembly(asm_code));
    helper.Run();

    // Validate that cyclic reg loaded correctly

    // Read back the stored accumulator data
    auto acc_result_bytes = helper.ReadXmem(0x3000, IPU__R_ACC_REG_SIZE_BYTES);

    // Reinterpret as uint32_t words (accumulator stores 128 uint32_t values)
    auto* acc_result_words = reinterpret_cast<uint32_t*>(acc_result_bytes.data());

    // Verify accumulator has accumulated values (should have 6 per uint32_t word)
    // First 128 words should have values from accumulation
    for (size_t i = 0; i < 128; i++)
    {
        EXPECT_EQ(acc_result_words[i], 6)
            << "Accumulator word " << i << " should be 6";
    }
}

/**
 * @brief Test floating point conversions to/from external memory
 */
TEST_F(IpuEmulatorTest, Memory_Fp8Conversions)
{
    // Write FP32 values as FP8_E4M3
    std::vector<float> input_values = {1.0f, 2.0f, 3.0f, 4.0f, 0.5f, -1.0f};
    helper.WriteXmemFp32AsE4M3(0x3000, input_values);

    // Read them back
    auto output_values = helper.ReadXmemE4M3AsFp32(0x3000, input_values.size());

    // Check round-trip conversion (allowing for FP8 precision loss)
    for (size_t i = 0; i < input_values.size(); i++)
    {
        EXPECT_NEAR(output_values[i], input_values[i], 0.1f)
            << "Value " << i << " should match approximately";
    }
}

/**
 * @brief Test cyclic register load
 */
TEST_F(IpuEmulatorTest, Memory_CyclicRegisterLoad)
{
    // Prepare cyclic data
    std::vector<uint8_t> cyclic_data(128);
    for (int i = 0; i < 128; i++)
    {
        cyclic_data[i] = (i * 2) & 0xFF;
    }
    helper.WriteXmem(0x5000, cyclic_data);

    std::string asm_code = R"(
set lr0 0x5000;;  # Base address
set lr1 0;;       # Index
ldr_cyclic_mult_reg lr0 cr0 lr1;;
bkpt;;
)";

    ASSERT_TRUE(helper.LoadProgramFromAssembly(asm_code));
    helper.Run();

    // Verify instruction executed
    EXPECT_EQ(helper.GetLr(0), 0x5000);

    std::vector<uint8_t> cyclic_reg_data = helper.GetCyclicBytes(0, 128);
    for (int i = 0; i < 128; i++)
    {
        EXPECT_EQ(cyclic_reg_data[i], cyclic_data[i])
            << "Cyclic register byte " << i << " should match loaded data";
    }
}

// ============================================================================
// Control Flow Tests
// ============================================================================

/**
 * @brief Test unconditional branch
 */
TEST_F(IpuEmulatorTest, ControlFlow_UnconditionalBranch)
{
    std::string asm_code = R"(
set lr0 1;;
b skip_section;;
set lr0 2;;  # Should be skipped
skip_section:
set lr1 3;;
bkpt;;
)";

    ASSERT_TRUE(helper.LoadProgramFromAssembly(asm_code));
    helper.Run();

    EXPECT_EQ(helper.GetLr(0), 1); // Should not have been changed to 2
    EXPECT_EQ(helper.GetLr(1), 3); // Should have executed after branch
}

/**
 * @brief Test conditional branch (BNE)
 */
TEST_F(IpuEmulatorTest, ControlFlow_ConditionalBranchNotEqual)
{
    std::string asm_code = R"(
set lr0 10;;
set lr1 20;;
bne lr0 lr1 not_equal_branch;;
set lr2 0;;  # Should be skipped
bkpt;;
not_equal_branch:
set lr2 1;;  # Should be executed
bkpt;;
)";

    ASSERT_TRUE(helper.LoadProgramFromAssembly(asm_code));
    helper.Run();

    EXPECT_EQ(helper.GetLr(2), 1); // Should have taken the branch
}

/**
 * @brief Test simple loop
 */
TEST_F(IpuEmulatorTest, SimpleLoop)
{
    std::string asm_code = R"(
set lr0 0;;      # Counter
set lr1 10;;     # Target
set lr2 0;;      # Zero register for comparison
loop_start:
incr lr0 1;;
bne lr0 lr1 loop_start;;
bkpt;;
)";

    ASSERT_TRUE(helper.LoadProgramFromAssembly(asm_code));
    int cycles = helper.Run(1000);

    ASSERT_GT(cycles, 0) << "Loop should complete";
    EXPECT_EQ(helper.GetLr(0), 10);
}

/**
 * @brief Test FP8 memory operations
 */
TEST_F(IpuEmulatorTest, Fp8MemoryOperations)
{
    // Write FP32 values as FP8_E4M3
    std::vector<float> input_values = {1.0f, 2.0f, 3.0f, 4.0f, 0.5f, -1.0f};
    helper.WriteXmemFp32AsE4M3(0x3000, input_values);

    // Read them back
    auto output_values = helper.ReadXmemE4M3AsFp32(0x3000, input_values.size());

    // Check round-trip conversion (allowing for FP8 precision loss)
    for (size_t i = 0; i < input_values.size(); i++)
    {
        EXPECT_NEAR(output_values[i], input_values[i], 0.1f)
            << "Value " << i << " should match approximately";
    }
}

/**
 * @brief Test program counter manipulation
 */
TEST_F(IpuEmulatorTest, ProgramCounterTest)
{
    std::string asm_code = R"(
set lr0 100;;
set lr1 200;;
bkpt;;
)";

    ASSERT_TRUE(helper.LoadProgramFromAssembly(asm_code));

    uint32_t initial_pc = helper.GetPc();
    EXPECT_EQ(initial_pc, 0);

    helper.Step(); // Execute first instruction
    EXPECT_EQ(helper.GetPc(), 1);

    helper.Step(); // Execute second instruction
    EXPECT_EQ(helper.GetPc(), 2);
}

/**
 * @brief Test direct register access
 */
TEST_F(IpuEmulatorTest, DirectRegisterAccess)
{
    // Set LR registers directly
    helper.SetLr(0, 0xDEADBEEF);
    helper.SetLr(5, 0x12345678);

    EXPECT_EQ(helper.GetLr(0), 0xDEADBEEF);
    EXPECT_EQ(helper.GetLr(5), 0x12345678);

    // Set CR registers directly
    helper.SetCr(0, 0xABCDEF00);
    EXPECT_EQ(helper.GetCr(0), 0xABCDEF00);
}

/**
 * @brief Test unconditional branch
 */
TEST_F(IpuEmulatorTest, UnconditionalBranch)
{
    std::string asm_code = R"(
set lr0 1;;
b skip_section;;
set lr0 2;;  # Should be skipped
skip_section:
set lr1 3;;
bkpt;;
)";

    ASSERT_TRUE(helper.LoadProgramFromAssembly(asm_code));
    helper.Run();

    EXPECT_EQ(helper.GetLr(0), 1); // Should not have been changed to 2
    EXPECT_EQ(helper.GetLr(1), 3); // Should have executed after branch
}

/**
 * @brief Test accumulator reset
 */
TEST_F(IpuEmulatorTest, AccumulatorReset)
{
    std::string asm_code = R"(
reset_acc;;
bkpt;;
)";

    for (int i = 0; i < int(IPU__R_ACC_REG_SIZE_WORDS); i++)
    {
        helper.SetAccWord(i, 12345); // Pre-set some non-zero value
    }

    ASSERT_TRUE(helper.LoadProgramFromAssembly(asm_code));
    helper.Run();

    // Verify accumulator is reset (check first few words)
    for (int i = 0; i < int(IPU__R_ACC_REG_SIZE_WORDS); i++)
    {
        EXPECT_EQ(helper.GetAccWord(i), 0) << "Accumulator word " << i << " should be zero";
    }
}
