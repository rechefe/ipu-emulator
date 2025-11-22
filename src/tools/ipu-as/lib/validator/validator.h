#ifndef IPU_AS_VALIDATOR_H
#define IPU_AS_VALIDATOR_H

#include <string>
#include <vector>
#include "../parser/parser.h"
#include <optional>
#include <cstdint>

#include <variant>

namespace ipu_as {

enum class OperandType { REG_R, REG_RQ, REG_LR, REG_CR, IMM, ANY };

enum class RegKind { R, RQ, LR, CR };

struct RegRef {
    RegKind kind;
    int index;
};

// Decoded operand value (either a register reference or an immediate)
enum class DecodedKind { DK_REG, DK_IMM };

struct DecodedOperand {
    DecodedKind kind;
    // valid when kind == DK_REG
    RegRef reg;
    // valid when kind == DK_IMM
    int64_t imm;
};

struct DecodedInst {
    std::string op;
    std::vector<DecodedOperand> operands;
};

struct ValidationResult {
    bool ok;
    std::string message; // empty when ok == true
};

// Validate a single parsed instruction. Returns {true, ""} on success,
// or {false, "reason"} on failure.
ValidationResult validate_inst(const Inst &inst);

// Validate multiple instructions.
std::vector<ValidationResult> validate_insts(const std::vector<Inst> &insts);

// Helper: convert string token into operand type
// (exposed for tests if needed)
// Return parsed RegRef if token is a register like R0, RQ3, LR1, CR2 (case-insensitive)
std::optional<RegRef> parse_register_token(const std::string &tok);
bool is_register_token(const std::string &tok);
bool is_immediate_token(const std::string &tok);

// Decode and validate an instruction. Returns a DecodedInst on success,
// or std::nullopt if the instruction is invalid.
std::optional<DecodedInst> decode_inst(const Inst &inst);

} // namespace ipu_as

#endif // IPU_AS_VALIDATOR_H
