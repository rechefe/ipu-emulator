#ifdef __cplusplus
#include <gtest/gtest.h>

extern "C"
{
#endif
#include "fp/fp.h"
#include "xmem/xmem.h"
#ifdef __cplusplus
}
#endif

#define EPSILON 0.01f

class FP8E4M3Test : public ::testing::Test
{
};
class FP8E5M2Test : public ::testing::Test
{
};
class FP16Test : public ::testing::Test
{
};
class FP4Test : public ::testing::Test
{
};
class FileLoadingTest : public ::testing::Test
{
};

TEST_F(FP8E4M3Test, ZeroConversion)
{
    fp__fp8_e4m3_t zero = fp__fp32_to_fp8_e4m3(0.0f);
    EXPECT_NEAR(fp__fp8_e4m3_to_fp32(zero), 0.0f, EPSILON);
}

TEST_F(FP8E4M3Test, SmallPositiveValue)
{
    fp__fp8_e4m3_t small = fp__fp32_to_fp8_e4m3(1.0f);
    float result = fp__fp8_e4m3_to_fp32(small);
    EXPECT_NEAR(result, 1.0f, EPSILON);
}

TEST_F(FP8E4M3Test, NegativeValue)
{
    fp__fp8_e4m3_t neg = fp__fp32_to_fp8_e4m3(-2.0f);
    float result = fp__fp8_e4m3_to_fp32(neg);
    EXPECT_LT(result, 0.0f);
}

TEST_F(FP8E4M3Test, MultiplicationMaxValues)
{
    // Max FP8_E4M3: exp=15, man=7 -> approximately 240.0f
    fp__fp8_e4m3_t max_val = fp__fp32_to_fp8_e4m3(240.0f);
    fp__fp8_e4m3_t min_val = fp__fp32_to_fp8_e4m3(0.001f);
    
    // Convert back to verify exact values
    float max_fp32 = fp__fp8_e4m3_to_fp32(max_val);
    float min_fp32 = fp__fp8_e4m3_to_fp32(min_val);
    
    // Multiply max * min
    float result = fp__fp8_e4m3_mult(max_val, min_val);
    float expected = max_fp32 * min_fp32;
    
    // Exact equality check
    EXPECT_EQ(result, expected);
}

TEST_F(FP8E4M3Test, MultiplicationMinMaxValues)
{
    // Min positive FP8_E4M3 (exp=0, man=1)
    fp__fp8_e4m3_t a = fp__fp32_to_fp8_e4m3(0.001953125f); // 2^-9
    fp__fp8_e4m3_t b = fp__fp32_to_fp8_e4m3(0.001953125f);
    
    float result = fp__fp8_e4m3_mult(a, b);
    float expected = fp__fp8_e4m3_to_fp32(a) * fp__fp8_e4m3_to_fp32(b);
    
    // Exact equality check
    EXPECT_EQ(result, expected);
}

TEST_F(FP8E4M3Test, Addition)
{
    fp__fp8_e4m3_t a = fp__fp32_to_fp8_e4m3(2.0f);
    fp__fp8_e4m3_t b = fp__fp32_to_fp8_e4m3(3.0f);
    float result = fp__fp8_e4m3_add(a, b);
    EXPECT_NEAR(result, 5.0f, 1.0f);
}

TEST_F(FP8E5M2Test, ZeroConversion)
{
    fp__fp8_e5m2_t zero = fp__fp32_to_fp8_e5m2(0.0f);
    EXPECT_NEAR(fp__fp8_e5m2_to_fp32(zero), 0.0f, EPSILON);
}

TEST_F(FP8E5M2Test, ValueConversion)
{
    fp__fp8_e5m2_t val = fp__fp32_to_fp8_e5m2(3.5f);
    float result = fp__fp8_e5m2_to_fp32(val);
    EXPECT_NEAR(result, 3.5f, 1.0f);
}

TEST_F(FP16Test, ZeroConversion)
{
    fp__fp16_t zero = fp__fp32_to_fp16(0.0f);
    EXPECT_NEAR(fp__fp16_to_fp32(zero), 0.0f, EPSILON);
}

TEST_F(FP16Test, PositiveConversion)
{
    fp__fp16_t val = fp__fp32_to_fp16(123.456f);
    float result = fp__fp16_to_fp32(val);
    EXPECT_NEAR(result, 123.456f, 0.1f);
}

TEST_F(FP16Test, NegativeConversion)
{
    fp__fp16_t neg = fp__fp32_to_fp16(-42.0f);
    float result = fp__fp16_to_fp32(neg);
    EXPECT_NEAR(result, -42.0f, 0.1f);
}

TEST_F(FP16Test, Multiplication)
{
    fp__fp16_t a = fp__fp32_to_fp16(4.0f);
    fp__fp16_t b = fp__fp32_to_fp16(5.0f);
    float result = fp__fp16_mult(a, b);
    EXPECT_NEAR(result, 20.0f, 0.5f);
}

TEST_F(FP16Test, Addition)
{
    fp__fp16_t a = fp__fp32_to_fp16(10.0f);
    fp__fp16_t b = fp__fp32_to_fp16(15.0f);
    float result = fp__fp16_add(a, b);
    EXPECT_NEAR(result, 25.0f, 0.5f);
}

TEST_F(FP4Test, ZeroConversion)
{
    fp__fp4_t zero = fp__fp32_to_fp4(0.0f);
    EXPECT_NEAR(fp__fp4_to_fp32(zero), 0.0f, EPSILON);
}

TEST_F(FP4Test, PositiveValue)
{
    fp__fp4_t val = fp__fp32_to_fp4(2.0f);
    float result = fp__fp4_to_fp32(val);
    EXPECT_GT(result, 0.0f);
}

TEST_F(FileLoadingTest, LoadFP32File)
{
    const char *test_file = "/tmp/test_fp32_gtest.bin";
    FILE *f = fopen(test_file, "wb");
    ASSERT_NE(f, nullptr);

    float test_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    fwrite(test_data, sizeof(float), 5, f);
    fclose(f);

    xmem__obj_t xmem;
    memset(&xmem, 0, sizeof(xmem));

    int result = fp__load_fp32_file_to_xmem(&xmem, test_file, 0, 0x1000, 0, 0);
    EXPECT_EQ(result, 5);

    remove(test_file);
}

TEST_F(FileLoadingTest, NullXmemPointer)
{
    int result = fp__load_fp32_file_to_xmem(NULL, "test.bin", 0, 0, 0, 0);
    EXPECT_EQ(result, -1);
}

TEST_F(FileLoadingTest, NullFilePointer)
{
    xmem__obj_t xmem;
    int result = fp__load_fp32_file_to_xmem(&xmem, NULL, 0, 0, 0, 0);
    EXPECT_EQ(result, -1);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}