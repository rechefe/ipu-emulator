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

TEST_F(FP8E4M3Test, RoundTripConversionNoDegeneration)
{
    // Test that values that can be represented in FP8_E4M3 without degradation
    // return the exact same float value after round-trip conversion (FP8 -> FP32 -> FP8)

    float representable_values[] = {
        0.0f,
        1.0f,
        2.0f,
        3.0f,
        4.0f,
        5.0f,
        8.0f,
        16.0f,
        32.0f,
        64.0f,
        128.0f,
        240.0f, // Max representable value
        -1.0f,
        -2.0f,
        -4.0f,
        -8.0f,
        -16.0f,
        -32.0f,
        -64.0f,
        -128.0f,
        -240.0f};

    int num_values = sizeof(representable_values) / sizeof(representable_values[0]);
    int round_trip_count = 0;
    int exact_match_count = 0;

    // Test round-trip conversion for representable values
    for (int i = 0; i < num_values; ++i)
    {
        // Convert to FP8_E4M3
        fp__fp8_e4m3_t fp8_val = fp__fp32_to_fp8_e4m3(representable_values[i]);

        // Convert to FP32
        float fp32_val = fp__fp8_e4m3_to_fp32(fp8_val);

        // Check if the float values are exactly the same
        EXPECT_EQ(fp32_val, representable_values[i])
            << "Round-trip conversion failed for value " << representable_values[i]
            << " (FP32: " << fp32_val << ", final FP32: " << representable_values[i] << ")";

        if (representable_values[i] == fp32_val)
        {
            exact_match_count++;
        }

        round_trip_count++;
    }

    EXPECT_EQ(exact_match_count, round_trip_count);
    printf("Round-trip conversion test: %d/%d representable values preserved as exact float values\n",
           exact_match_count, round_trip_count);
}

TEST_F(FP8E4M3Test, MultiplicationExactEqualityLargeSet)
{
    // Test exact equality on a large set of FP8_E4M3 inputs
    // Generate test values covering the entire range of representable values
    float test_values[] = {
        0.0f,   // Zero
        0.001f, // Very small positive
        0.01f,
        0.1f,
        0.5f,
        1.0f,
        2.0f,
        3.0f,
        4.0f,
        5.0f,
        10.0f,
        50.0f,
        100.0f,
        150.0f,
        200.0f,
        240.0f,  // Max FP8_E4M3
        -0.001f, // Negative values
        -0.1f,
        -1.0f,
        -2.5f,
        -10.0f,
        -100.0f,
        -240.0f};

    int num_test_values = sizeof(test_values) / sizeof(test_values[0]);
    int test_count = 0;

    // Test all combinations of values
    for (int i = 0; i < num_test_values; ++i)
    {
        for (int j = 0; j < num_test_values; ++j)
        {
            fp__fp8_e4m3_t a = fp__fp32_to_fp8_e4m3(test_values[i]);
            fp__fp8_e4m3_t b = fp__fp32_to_fp8_e4m3(test_values[j]);

            // Convert back to fp32 to get the actual values being multiplied
            float a_fp32 = fp__fp8_e4m3_to_fp32(a);
            float b_fp32 = fp__fp8_e4m3_to_fp32(b);

            // Compute result using the multiplication function
            float result = fp__fp8_e4m3_mult(a, b);

            // Compute expected result
            float expected = a_fp32 * b_fp32;

            // Exact equality check - the result must match exactly
            EXPECT_EQ(result, expected)
                << "Multiplication failed for values: " << a_fp32 << " * " << b_fp32
                << " = " << result << " (expected " << expected << ")";

            test_count++;
        }
    }

    // Verify we actually ran the tests
    EXPECT_GT(test_count, 0);
    printf("Tested %d multiplication pairs for exact equality\n", test_count);
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

TEST_F(FP8E5M2Test, RoundTripConversionNoDegeneration)
{
    // Test that values that can be represented in FP8_E5M2 without degradation
    // return the exact same float value after round-trip conversion
    
    float representable_values[] = {
        0.0f,
        0.5f,
        1.0f,
        2.0f,
        4.0f,
        8.0f,
        16.0f,
        32.0f,
        64.0f,
        128.0f,
        256.0f,
        512.0f,
        1024.0f,
        2048.0f,
        -0.5f,
        -1.0f,
        -2.0f,
        -4.0f,
        -8.0f,
        -16.0f,
        -32.0f,
        -64.0f,
        -128.0f,
        -256.0f,
        -512.0f,
        -1024.0f
    };

    int num_values = sizeof(representable_values) / sizeof(representable_values[0]);
    int exact_match_count = 0;

    for (int i = 0; i < num_values; ++i)
    {
        fp__fp8_e5m2_t fp8_val = fp__fp32_to_fp8_e5m2(representable_values[i]);
        float fp32_val = fp__fp8_e5m2_to_fp32(fp8_val);

        EXPECT_EQ(fp32_val, representable_values[i])
            << "Round-trip conversion failed for FP8_E5M2 value " << representable_values[i];

        if (representable_values[i] == fp32_val)
        {
            exact_match_count++;
        }
    }

    EXPECT_EQ(exact_match_count, num_values);
    printf("FP8_E5M2 round-trip test: %d/%d values preserved\n", exact_match_count, num_values);
}

TEST_F(FP8E5M2Test, MultiplicationExactEqualityLargeSet)
{
    float test_values[] = {
        0.0f, 0.5f, 1.0f, 2.0f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f, 128.0f,
        256.0f, 512.0f, -0.5f, -1.0f, -2.0f, -4.0f, -8.0f, -16.0f, -32.0f, -64.0f
    };

    int num_test_values = sizeof(test_values) / sizeof(test_values[0]);
    int test_count = 0;

    for (int i = 0; i < num_test_values; ++i)
    {
        for (int j = 0; j < num_test_values; ++j)
        {
            fp__fp8_e5m2_t a = fp__fp32_to_fp8_e5m2(test_values[i]);
            fp__fp8_e5m2_t b = fp__fp32_to_fp8_e5m2(test_values[j]);

            float a_fp32 = fp__fp8_e5m2_to_fp32(a);
            float b_fp32 = fp__fp8_e5m2_to_fp32(b);

            float result = fp__fp8_e5m2_mult(a, b);
            float expected = a_fp32 * b_fp32;

            EXPECT_EQ(result, expected)
                << "FP8_E5M2 multiplication failed for " << a_fp32 << " * " << b_fp32;

            test_count++;
        }
    }

    EXPECT_GT(test_count, 0);
    printf("FP8_E5M2 tested %d multiplication pairs\n", test_count);
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

TEST_F(FP16Test, RoundTripConversionNoDegeneration)
{
    // Test that values that can be represented in FP16 without degradation
    // return the exact same float value after round-trip conversion
    
    float representable_values[] = {
        0.0f,
        0.5f,
        1.0f,
        2.0f,
        4.0f,
        8.0f,
        16.0f,
        32.0f,
        64.0f,
        128.0f,
        256.0f,
        512.0f,
        1024.0f,
        2048.0f,
        4096.0f,
        8192.0f,
        16384.0f,
        32768.0f,
        65504.0f,     // Max FP16
        -0.5f,
        -1.0f,
        -2.0f,
        -4.0f,
        -8.0f,
        -16.0f,
        -32.0f,
        -64.0f,
        -128.0f,
        -256.0f,
        -512.0f,
        -1024.0f,
        -2048.0f,
        -4096.0f,
        -8192.0f,
        -16384.0f,
        -32768.0f,
        -65504.0f
    };

    int num_values = sizeof(representable_values) / sizeof(representable_values[0]);
    int exact_match_count = 0;

    for (int i = 0; i < num_values; ++i)
    {
        fp__fp16_t fp16_val = fp__fp32_to_fp16(representable_values[i]);
        float fp32_val = fp__fp16_to_fp32(fp16_val);

        EXPECT_EQ(fp32_val, representable_values[i])
            << "Round-trip conversion failed for FP16 value " << representable_values[i];

        if (representable_values[i] == fp32_val)
        {
            exact_match_count++;
        }
    }

    EXPECT_EQ(exact_match_count, num_values);
    printf("FP16 round-trip test: %d/%d values preserved\n", exact_match_count, num_values);
}

TEST_F(FP16Test, MultiplicationExactEqualityLargeSet)
{
    float test_values[] = {
        0.0f, 0.5f, 1.0f, 2.0f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f, 128.0f,
        256.0f, 512.0f, 1024.0f, 2048.0f, 4096.0f, -0.5f, -1.0f, -2.0f, -4.0f, -8.0f,
        -16.0f, -32.0f, -64.0f, -128.0f, -256.0f
    };

    int num_test_values = sizeof(test_values) / sizeof(test_values[0]);
    int test_count = 0;

    for (int i = 0; i < num_test_values; ++i)
    {
        for (int j = 0; j < num_test_values; ++j)
        {
            fp__fp16_t a = fp__fp32_to_fp16(test_values[i]);
            fp__fp16_t b = fp__fp32_to_fp16(test_values[j]);

            float a_fp32 = fp__fp16_to_fp32(a);
            float b_fp32 = fp__fp16_to_fp32(b);

            float result = fp__fp16_mult(a, b);
            float expected = a_fp32 * b_fp32;

            EXPECT_EQ(result, expected)
                << "FP16 multiplication failed for " << a_fp32 << " * " << b_fp32;

            test_count++;
        }
    }

    EXPECT_GT(test_count, 0);
    printf("FP16 tested %d multiplication pairs\n", test_count);
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

TEST_F(FP4Test, RoundTripConversionNoDegeneration)
{
    // Test that values that can be represented in FP4 without degradation
    // return the exact same float value after round-trip conversion
    // FP4 has very limited precision (2 exp bits, 1 mantissa bit)
    // So only a few values are actually representable
    
    float representable_values[] = {
        0.0f,
        1.0f,
        2.0f,
        -1.0f,
        -2.0f
    };

    int num_values = sizeof(representable_values) / sizeof(representable_values[0]);
    int exact_match_count = 0;

    for (int i = 0; i < num_values; ++i)
    {
        fp__fp4_t fp4_val = fp__fp32_to_fp4(representable_values[i]);
        float fp32_val = fp__fp4_to_fp32(fp4_val);

        EXPECT_EQ(fp32_val, representable_values[i])
            << "Round-trip conversion failed for FP4 value " << representable_values[i];

        if (representable_values[i] == fp32_val)
        {
            exact_match_count++;
        }
    }

    EXPECT_EQ(exact_match_count, num_values);
    printf("FP4 round-trip test: %d/%d values preserved\n", exact_match_count, num_values);
}

TEST_F(FP4Test, MultiplicationExactEqualityLargeSet)
{
    float test_values[] = {
        0.0f, 0.5f, 1.0f, 2.0f, 4.0f, 8.0f, 16.0f, -0.5f, -1.0f, -2.0f, -4.0f, -8.0f, -16.0f
    };

    int num_test_values = sizeof(test_values) / sizeof(test_values[0]);
    int test_count = 0;

    for (int i = 0; i < num_test_values; ++i)
    {
        for (int j = 0; j < num_test_values; ++j)
        {
            fp__fp4_t a = fp__fp32_to_fp4(test_values[i]);
            fp__fp4_t b = fp__fp32_to_fp4(test_values[j]);

            float a_fp32 = fp__fp4_to_fp32(a);
            float b_fp32 = fp__fp4_to_fp32(b);

            float result = fp__fp4_mult(a, b);
            float expected = a_fp32 * b_fp32;

            EXPECT_EQ(result, expected)
                << "FP4 multiplication failed for " << a_fp32 << " * " << b_fp32;

            test_count++;
        }
    }

    EXPECT_GT(test_count, 0);
    printf("FP4 tested %d multiplication pairs\n", test_count);
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