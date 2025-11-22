#include <gtest/gtest.h>
#include "lib/parser/parser.h"
#include "lib/validator/validator.h"

using namespace ipu_as;

TEST(Validator, ValidAdd)
{
    auto inst = parse_line("ADD R1, R2, R3");
    ASSERT_TRUE(inst.has_value());
    auto res = validate_inst(*inst);
    EXPECT_TRUE(res.ok) << res.message;
}

TEST(Validator, UnknownOp)
{
    auto inst = parse_line("FOO R1");
    ASSERT_TRUE(inst.has_value());
    auto res = validate_inst(*inst);
    EXPECT_FALSE(res.ok);
    EXPECT_NE(res.message.find("unknown opcode"), std::string::npos);
}

TEST(Validator, WrongOperandCount)
{
    auto inst = parse_line("ADD R1, R2");
    ASSERT_TRUE(inst.has_value());
    auto res = validate_inst(*inst);
    EXPECT_FALSE(res.ok);
    EXPECT_NE(res.message.find("expected"), std::string::npos);
}

TEST(Validator, WrongOperandType)
{
    // ADD expects registers: second operand is immediate here
    auto inst = parse_line("ADD R1, 42, R3");
    ASSERT_TRUE(inst.has_value());
    auto res = validate_inst(*inst);
    EXPECT_FALSE(res.ok);
    EXPECT_NE(res.message.find("must be register"), std::string::npos);
}

TEST(Validator, LoadImmediate)
{
    auto inst = parse_line("LOAD R1, 0x10");
    ASSERT_TRUE(inst.has_value());
    auto res = validate_inst(*inst);
    EXPECT_TRUE(res.ok) << res.message;
}

TEST(Validator, StoreWrongType)
{
    // store expects (reg, imm) so providing two regs should fail
    auto inst = parse_line("STORE R1, R2");
    ASSERT_TRUE(inst.has_value());
    auto res = validate_inst(*inst);
    EXPECT_FALSE(res.ok);
}

TEST(Validator, Multiple)
{
    std::vector<std::string> lines = {"add r1, r2, r3", "foo r1", "mul r0, r1"};
    auto parsed = parse_lines(lines);
    auto results = validate_insts(parsed);
    ASSERT_EQ(results.size(), 3u);
    EXPECT_TRUE(results[0].ok);
    EXPECT_FALSE(results[1].ok);
    EXPECT_FALSE(results[2].ok); // mul expects 3 operands
}

TEST(Validator, RQRegisters)
{
    auto inst = parse_line("DOTRQ RQ1, RQ2");
    ASSERT_TRUE(inst.has_value());
    auto res = validate_inst(*inst);
    EXPECT_TRUE(res.ok) << res.message;

    auto inst2 = parse_line("DOTRQ RQ999, RQ2");
    ASSERT_TRUE(inst2.has_value());
    auto res2 = validate_inst(*inst2);
    EXPECT_FALSE(res2.ok);
    EXPECT_NE(res2.message.find("out of range"), std::string::npos);
}

TEST(Validator, LRCRRegisters)
{
    auto i1 = parse_line("SETLR LR3");
    ASSERT_TRUE(i1.has_value());
    EXPECT_TRUE(validate_inst(*i1).ok);

    auto i2 = parse_line("SETCR CR0");
    ASSERT_TRUE(i2.has_value());
    EXPECT_TRUE(validate_inst(*i2).ok);
}

TEST(Validator, DecodeAdd)
{
    auto inst = parse_line("ADD R4, R5, R6");
    ASSERT_TRUE(inst.has_value());
    auto d = decode_inst(*inst);
    ASSERT_TRUE(d.has_value());
    EXPECT_EQ(d->op, "add");
    ASSERT_EQ(d->operands.size(), 3u);
    EXPECT_EQ(d->operands[0].kind, DecodedKind::DK_REG);
    EXPECT_EQ(d->operands[0].reg.kind, RegKind::R);
    EXPECT_EQ(d->operands[0].reg.index, 4);
}

TEST(Validator, DecodeLoadImmediate)
{
    auto inst = parse_line("LOAD R1, 0x10");
    ASSERT_TRUE(inst.has_value());
    auto d = decode_inst(*inst);
    ASSERT_TRUE(d.has_value());
    ASSERT_EQ(d->operands.size(), 2u);
    EXPECT_EQ(d->operands[1].kind, DecodedKind::DK_IMM);
    EXPECT_EQ(d->operands[1].imm, 0x10);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
