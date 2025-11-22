#include <gtest/gtest.h>
#include "lib/parser/parser.h"

using namespace ipu_as;

TEST(ParserLine, SimpleInstruction)
{
    auto r = parse_line("ADD R1, R2, R3");
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->op, "add");
    ASSERT_EQ(r->operands.size(), 3u);
    EXPECT_EQ(r->operands[0], "R1");
    EXPECT_EQ(r->operands[1], "R2");
    EXPECT_EQ(r->operands[2], "R3");
}

TEST(ParserLine, ExtraSpacesAndTabs)
{
    auto r = parse_line("   mul\t r0 ,\tr1,   r2   ");
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->op, "mul");
    ASSERT_EQ(r->operands.size(), 3u);
    EXPECT_EQ(r->operands[0], "r0");
}

TEST(ParserLine, CommentsAndBlank)
{
    EXPECT_FALSE(parse_line("").has_value());
    EXPECT_FALSE(parse_line("   \t  ").has_value());
    EXPECT_FALSE(parse_line("# just a comment").has_value());
    EXPECT_FALSE(parse_line("; another comment").has_value());
    EXPECT_FALSE(parse_line("// comment here").has_value());

    auto r = parse_line("add r1, r2  // trailing comment");
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->op, "add");
    ASSERT_EQ(r->operands.size(), 2u);
}

TEST(ParserLines, Multiple)
{
    std::vector<std::string> lines = {
        "add r1, r2, r3",
        "  // comment",
        "mul r4, r5",
        "",
        "sub r6, r7"
    };
    auto out = parse_lines(lines);
    ASSERT_EQ(out.size(), 3u);
    EXPECT_EQ(out[0].op, "add");
    EXPECT_EQ(out[1].op, "mul");
    EXPECT_EQ(out[2].op, "sub");
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
