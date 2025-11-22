#include <gtest/gtest.h>
#include "lib/parser/parser.h"
#include "lib/validator/validator.h"
#include "lib/encoder/encoder.h"

using namespace ipu_as;

TEST(Encoder, AddAndJmp)
{
    std::vector<std::string> lines = {
        "start:",
        "  add r1, r2, r3",
        "  jmp start",
    };
    std::unordered_map<std::string, size_t> labels;
    auto parsed = parse_lines_with_labels(lines, labels);
    ASSERT_EQ(parsed.size(), 2u);
    EXPECT_EQ(labels["start"], 0u);

    // decode each instruction
    std::vector<DecodedInst> dis;
    for (auto &i : parsed)
    {
        auto d = decode_inst(i);
        ASSERT_TRUE(d.has_value());
        dis.push_back(*d);
    }

    std::string err;
    auto bin = encode_decoded(dis, labels, err);
    ASSERT_TRUE(bin.has_value());
    auto &b = *bin;
    // first instruction: opcode byte, opcount
    ASSERT_GE(b.size(), 2u + 3*4);
    EXPECT_EQ(b[0], 1); // opcode add -> 1
    EXPECT_EQ(b[1], 3); // three operands
    // third instruction (jmp) second inst offset: starts at byte index for second instruction
    // locate opcode of second instruction
    size_t pos = 0;
    // skip first instruction
    pos += 2 + 3*4;
    EXPECT_EQ(b[pos], 10); // jmp opcode id 10
    EXPECT_EQ(b[pos+1], 1);
    // operand for jmp is label -> type 5 in top byte little-endian of 4-byte operand
    uint32_t opv = (uint32_t)b[pos+2] | ((uint32_t)b[pos+3]<<8) | ((uint32_t)b[pos+4]<<16) | ((uint32_t)b[pos+5]<<24);
    uint8_t t = (uint8_t)((opv >> 24) & 0xFF);
    uint32_t val = opv & 0x00FFFFFFu;
    EXPECT_EQ(t, 5u);
    EXPECT_EQ(val, 0u);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
