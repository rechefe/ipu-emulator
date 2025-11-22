#include "encoder.h"
#include <cstring>

namespace ipu_as {

static const std::unordered_map<std::string, uint8_t> opcode_map = {
    {"add", 1}, {"sub", 2}, {"mul", 3}, {"mov", 4}, {"load", 5}, {"store", 6},
    {"dotrq", 7}, {"setlr", 8}, {"setcr", 9}, {"jmp", 10}, {"b", 11}, {"beq", 12}, {"bne", 13},
};

static uint32_t pack_operand_uint32(uint8_t type, uint32_t value)
{
    return ((uint32_t)type << 24) | (value & 0x00FFFFFFu);
}

static void push_u32le(std::vector<uint8_t> &out, uint32_t v)
{
    uint8_t b[4];
    b[0] = (uint8_t)(v & 0xFF);
    b[1] = (uint8_t)((v >> 8) & 0xFF);
    b[2] = (uint8_t)((v >> 16) & 0xFF);
    b[3] = (uint8_t)((v >> 24) & 0xFF);
    out.insert(out.end(), b, b + 4);
}

std::optional<std::vector<uint8_t>> encode_decoded(const std::vector<DecodedInst> &insts, const LabelMap &labels, std::string &err)
{
    std::vector<uint8_t> out;
    out.reserve(insts.size() * 8);

    for (size_t idx = 0; idx < insts.size(); ++idx)
    {
        const auto &di = insts[idx];
        // find opcode id

        auto it = opcode_map.find(di.op);
        if (it == opcode_map.end())
        {
            err = "unknown opcode in encoding: " + di.op;
            return std::nullopt;
        }
        uint8_t opcode = it->second;
        uint8_t opcount = (uint8_t)di.operands.size();
        out.push_back(opcode);
        out.push_back(opcount);

        for (const auto &op : di.operands)
        {
            uint32_t packed = 0;
            switch (op.kind)
            {
            case DecodedKind::DK_REG:
            {
                uint8_t t = 0; // R
                switch (op.reg.kind)
                {
                case RegKind::R: t = 0; break;
                case RegKind::RQ: t = 1; break;
                case RegKind::LR: t = 2; break;
                case RegKind::CR: t = 3; break;
                }
                uint32_t val = (uint32_t)op.reg.index & 0x00FFFFFFu;
                packed = pack_operand_uint32(t, val);
                break;
            }
            case DecodedKind::DK_IMM:
            {
                uint8_t t = 4;
                uint32_t val = (uint32_t)op.imm & 0x00FFFFFFu;
                packed = pack_operand_uint32(t, val);
                break;
            }
            case DecodedKind::DK_LABEL:
            {
                uint8_t t = 5;
                auto itl = labels.find(op.label);
                if (itl == labels.end())
                {
                    err = "unknown label in encoding: " + op.label;
                    return std::nullopt;
                }
                uint32_t val = (uint32_t)itl->second & 0x00FFFFFFu;
                packed = pack_operand_uint32(t, val);
                break;
            }
            }
            push_u32le(out, packed);
        }
    }
    return out;
}

std::optional<std::vector<uint8_t>> encode_from_parsed(const std::vector<Inst> &insts, const LabelMap &labels, std::string &err)
{
    std::vector<DecodedInst> dis;
    dis.reserve(insts.size());
    for (const auto &i : insts)
    {
        auto d = decode_inst(i);
        if (!d)
        {
            err = "decode failed for opcode: " + i.op;
            return std::nullopt;
        }
        dis.push_back(*d);
    }
    return encode_decoded(dis, labels, err);
}

} // namespace ipu_as
