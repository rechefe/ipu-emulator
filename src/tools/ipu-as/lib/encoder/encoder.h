#ifndef IPU_AS_ENCODER_H
#define IPU_AS_ENCODER_H

#include <vector>
#include <string>
#include <optional>
#include <cstdint>
#include <unordered_map>
#include "../validator/validator.h"

namespace ipu_as
{

    // Encoded output is a flat byte vector. The format per instruction is:
    // [opcode:1][op_count:1][operand0:4][operand1:4]...
    // Each operand is a 32-bit little-endian value: [type:8 | value:24]
    // type codes: 0=R,1=RQ,2=LR,3=CR,4=IMM,5=LABEL

    using LabelMap = std::unordered_map<std::string, size_t>;

    // Encode already-decoded instructions. Label operands must be resolved using
    // `labels` (mapping label -> instruction index); for label operands the value
    // encoded is the instruction index (uint32). Returns std::nullopt on failure
    // and sets `err` with a message.
    std::optional<std::vector<uint8_t>> encode_decoded(const std::vector<DecodedInst> &insts, const LabelMap &labels, std::string &err);

    // Convenience: decode+encode from parsed `Inst` list and label map. Uses
    // `decode_inst` internally. Returns std::nullopt on failure and sets `err`.
    std::optional<std::vector<uint8_t>> encode_from_parsed(const std::vector<Inst> &insts, const LabelMap &labels, std::string &err);

} // namespace ipu_as

#endif // IPU_AS_ENCODER_H
