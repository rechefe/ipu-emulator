#include <gtest/gtest.h>
#include <cstring>
#include <stdlib.h>

extern "C" {
#include "xmem/xmem.h"
}

TEST(XmemInitialize, ZeroedMemory)
{
    xmem__obj_t *xmem = xmem__initialize_xmem();
    ASSERT_NE(xmem, nullptr);

    uint8_t buf[16];
    xmem__read_address(xmem, 0, buf, sizeof(buf));
    for (size_t i = 0; i < sizeof(buf); ++i)
    {
        EXPECT_EQ(buf[i], 0u);
    }

    free(xmem);
}

TEST(XmemWriteRead, BasicRoundTrip)
{
    xmem__obj_t *xmem = xmem__initialize_xmem();
    uint8_t data[4] = {1, 2, 3, 4};
    xmem__write_address(xmem, 10, data, sizeof(data));

    uint8_t out[4] = {0};
    xmem__read_address(xmem, 10, out, sizeof(out));
    EXPECT_EQ(0, memcmp(data, out, sizeof(data)));

    free(xmem);
}

TEST(XmemLoadArray, LoadsCorrectly)
{
    xmem__obj_t *xmem = xmem__initialize_xmem();
    uint8_t arr[5] = {10, 11, 12, 13, 14};
    const int start = 200;
    xmem__load_array_to(xmem, arr, 5, start);

    uint8_t out[5] = {0};
    xmem__read_address(xmem, start, out, 5);
    EXPECT_EQ(0, memcmp(arr, out, 5));

    free(xmem);
}

TEST(XmemLoadMatrix, RowAlignmentBehavior)
{
    xmem__obj_t *xmem = xmem__initialize_xmem();
    const int rows = 2;
    const int cols = 3;
    uint8_t matrix[rows * cols];
    for (int i = 0; i < rows * cols; ++i)
        matrix[i] = (uint8_t)(i + 1);

    const int start = 0;
    xmem__load_matrix_to(xmem, matrix, rows, cols, start);

    // First row is at start + ALIGN(0)
    uint8_t out0[3] = {0};
    xmem__read_address(xmem, start + XMEM__ALIGN_ADDR(0), out0, cols);
    EXPECT_EQ(0, memcmp(out0, matrix, cols));

    // Second row is at start + ALIGN(3)
    int offset1 = XMEM__ALIGN_ADDR(3);
    uint8_t out1[3] = {0};
    xmem__read_address(xmem, start + offset1, out1, cols);
    EXPECT_EQ(0, memcmp(out1, matrix + cols, cols));

    free(xmem);
}

TEST(XmemBounds, LastByteAccess)
{
    xmem__obj_t *xmem = xmem__initialize_xmem();
    uint8_t v = 0xAA;
    int last = XMEM__XMEM_SIZE_BYTES - 1;
    xmem__write_address(xmem, last, &v, 1);
    uint8_t out = 0;
    xmem__read_address(xmem, last, &out, 1);
    EXPECT_EQ(out, v);
    free(xmem);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
