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
class FileLoadingTest : public ::testing::Test
{
};
class ExplicitTest : public ::testing::Test
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

TEST_F(FP8E4M3Test, RoundTripSpecificValue)
{
    // Test roundtrip conversion for the specific value 0.0078125 (2^-7)
    // This is the minimum normal exponent value in FP8_E4M3
    // It encodes as exp=0, man=0 (subnormal with no mantissa bits)
    float test_value = 0.0078125;

    fp__fp8_e4m3_t fp8_val = fp__fp32_to_fp8_e4m3(test_value);

    EXPECT_EQ(fp8_val.w, 0x04)
        << "FP8_E4M3 representation incorrect for specific value " << test_value;

    float result = fp__fp8_e4m3_to_fp32(fp8_val);

    EXPECT_EQ(result, 0.0078125f)
        << "Round-trip conversion should preserve value " << test_value;
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
        -1024.0f};

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
        256.0f, 512.0f, -0.5f, -1.0f, -2.0f, -4.0f, -8.0f, -16.0f, -32.0f, -64.0f};

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

TEST_F(ExplicitTest, SampleTest)
{
    float f1 = 3.5f;
    float f2 = 2.25f;

    fp__fp8_e4m3_t fp8_1 = fp__fp32_to_fp8_e4m3(f1);
    fp__fp8_e4m3_t fp8_2 = fp__fp32_to_fp8_e4m3(f2);

    float result = fp__fp8_e4m3_mult(fp8_1, fp8_2);

    EXPECT_EQ(result, 7.875f)
        << "Explicit test multiplication failed for " << f1 << " * " << f2;

    fp__fp8_e4m3_t result_mult = fp__fp32_to_fp8_e4m3(result);
    float final_result = fp__fp8_e4m3_to_fp32(result_mult);

    EXPECT_EQ(final_result, 7.5f);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}