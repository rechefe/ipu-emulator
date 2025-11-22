#include "validator.h"
#include <unordered_map>
#include <algorithm>
#include <cstdlib>
#include <cctype>
#include <optional>
#include <cstdint>

extern "C"
{
    #include "ipu/ipu.h"
}

namespace ipu_as
{

    static const std::unordered_map<std::string, std::vector<OperandType>> kOpOperands = {
        {"add", {OperandType::REG_R, OperandType::REG_R, OperandType::REG_R}},
        {"sub", {OperandType::REG_R, OperandType::REG_R, OperandType::REG_R}},
        {"mul", {OperandType::REG_R, OperandType::REG_R, OperandType::REG_R}},
        {"mov", {OperandType::REG_R, OperandType::REG_R}},
        {"load", {OperandType::REG_R, OperandType::IMM}},
        {"store", {OperandType::REG_R, OperandType::IMM}},
        // example instructions for other register kinds
        {"dotrq", {OperandType::REG_RQ, OperandType::REG_RQ}},
        {"setlr", {OperandType::REG_LR}},
        {"setcr", {OperandType::REG_CR}},
    };

    static bool is_register_token_impl(const std::string &s)
    {
        if (s.size() < 2)
            return false;
        if (s[0] != 'r' && s[0] != 'R')
            return false;
        for (size_t i = 1; i < s.size(); ++i)
        {
            if (!std::isdigit((unsigned char)s[i]))
                return false;
        }
        return true;
    }

    static bool is_immediate_token_impl(const std::string &s)
    {
        if (s.empty())
            return false;
        char *end = nullptr;
        // strtol with base 0 supports 0x prefix
        errno = 0;
        long val = std::strtol(s.c_str(), &end, 0);
        (void)val;
        if (end == s.c_str())
            return false;
        // ensure whole string parsed
        return (*end == '\0');
    }

    bool is_register_token(const std::string &tok) { return is_register_token_impl(tok); }
    bool is_immediate_token(const std::string &tok) { return is_immediate_token_impl(tok); }

    static std::optional<RegRef> parse_register_token_impl(const std::string &s)
    {
        if (s.size() < 2)
            return std::nullopt;
        // lowercase copy
        std::string t;
        t.reserve(s.size());
        for (char c : s)
            t.push_back(std::tolower((unsigned char)c));

        // try prefixes
        if (t.rfind("rq", 0) == 0)
        {
            std::string num = t.substr(2);
            if (num.empty())
                return std::nullopt;
            char *end = nullptr;
            long v = std::strtol(num.c_str(), &end, 10);
            if (*end != '\0' || v < 0)
                return std::nullopt;
            return RegRef{RegKind::RQ, (int)v};
        }
        if (t.rfind("lr", 0) == 0)
        {
            std::string num = t.substr(2);
            if (num.empty())
                return std::nullopt;
            char *end = nullptr;
            long v = std::strtol(num.c_str(), &end, 10);
            if (*end != '\0' || v < 0)
                return std::nullopt;
            return RegRef{RegKind::LR, (int)v};
        }
        if (t.rfind("cr", 0) == 0)
        {
            std::string num = t.substr(2);
            if (num.empty())
                return std::nullopt;
            char *end = nullptr;
            long v = std::strtol(num.c_str(), &end, 10);
            if (*end != '\0' || v < 0)
                return std::nullopt;
            return RegRef{RegKind::CR, (int)v};
        }
        // default R: starts with 'r' followed by digits but not 'rq'
        if (t[0] == 'r')
        {
            std::string num = t.substr(1);
            if (num.empty())
                return std::nullopt;
            char *end = nullptr;
            long v = std::strtol(num.c_str(), &end, 10);
            if (*end != '\0' || v < 0)
                return std::nullopt;
            return RegRef{RegKind::R, (int)v};
        }
        return std::nullopt;
    }

    std::optional<RegRef> parse_register_token(const std::string &tok) { return parse_register_token_impl(tok); }

    ValidationResult validate_inst(const Inst &inst)
    {
        // op already lowercased by parser
        auto it = kOpOperands.find(inst.op);
        if (it == kOpOperands.end())
            return {false, "unknown opcode: " + inst.op};
        const auto &expected = it->second;
        if (inst.operands.size() != expected.size())
        {
            return {false, "wrong operand count for '" + inst.op + "': expected " + std::to_string(expected.size()) + ", got " + std::to_string(inst.operands.size())};
        }
        // Check each operand type
        for (size_t i = 0; i < expected.size(); ++i)
        {
            const std::string &tok = inst.operands[i];
            OperandType t = expected[i];
            if (tok.empty())
                return {false, "empty operand at index " + std::to_string(i)};
            if (t == OperandType::IMM)
            {
                if (!is_immediate_token_impl(tok))
                    return {false, "operand " + std::to_string(i) + " for '" + inst.op + "' must be immediate"};
                continue;
            }
            if (t == OperandType::ANY)
                continue;

            // Expect a register of a specific kind
            auto r = parse_register_token_impl(tok);
            if (!r)
                return {false, "operand " + std::to_string(i) + " for '" + inst.op + "' must be register"};
            RegKind need;
            switch (t)
            {
            case OperandType::REG_R:
                need = RegKind::R;
                break;
            case OperandType::REG_RQ:
                need = RegKind::RQ;
                break;
            case OperandType::REG_LR:
                need = RegKind::LR;
                break;
            case OperandType::REG_CR:
                need = RegKind::CR;
                break;
            default:
                need = RegKind::R; // fallback
            }
            if (r->kind != need)
                return {false, "operand " + std::to_string(i) + " for '" + inst.op + "' must be register of correct kind"};

            // range checks
            // default limits (can be made configurable later)
            const int kMaxR = IPU__R_REGS_NUM - 1;
            const int kMaxRQ = IPU__RQ_REGS_NUM - 1;
            const int kMaxLR = IPU__LR_REGS_NUM - 1;
            const int kMaxCR = IPU__CR_REGS_NUM - 1;
            switch (r->kind)
            {
            case RegKind::R:
                if (r->index < 0 || r->index > kMaxR)
                    return {false, "register index out of range for R: " + std::to_string(r->index)};
                break;
            case RegKind::RQ:
                if (r->index < 0 || r->index > kMaxRQ)
                    return {false, "register index out of range for RQ: " + std::to_string(r->index)};
                break;
            case RegKind::LR:
                if (r->index < 0 || r->index > kMaxLR)
                    return {false, "register index out of range for LR: " + std::to_string(r->index)};
                break;
            case RegKind::CR:
                if (r->index < 0 || r->index > kMaxCR)
                    return {false, "register index out of range for CR: " + std::to_string(r->index)};
                break;
            }
        }
        return {true, ""};
    }

    std::vector<ValidationResult> validate_insts(const std::vector<Inst> &insts)
    {
        std::vector<ValidationResult> out;
        out.reserve(insts.size());
        for (const auto &i : insts)
            out.push_back(validate_inst(i));
        return out;
    }

    std::optional<DecodedInst> decode_inst(const Inst &inst)
    {
        auto vr = validate_inst(inst);
        if (!vr.ok)
            return std::nullopt;

        DecodedInst d;
        d.op = inst.op;
        d.operands.reserve(inst.operands.size());
        for (size_t i = 0; i < inst.operands.size(); ++i)
        {
            const auto &tok = inst.operands[i];
            auto expected = kOpOperands.at(inst.op)[i];
            DecodedOperand dob;
            if (expected == OperandType::IMM)
            {
                dob.kind = DecodedKind::DK_IMM;
                char *end = nullptr;
                long long v = std::strtoll(tok.c_str(), &end, 0);
                dob.imm = (int64_t)v;
            }
            else
            {
                dob.kind = DecodedKind::DK_REG;
                auto r = parse_register_token_impl(tok);
                // validation already ensured parse succeeds
                dob.reg = *r;
            }
            d.operands.push_back(dob);
        }
        return d;
    }

} // namespace ipu_as
