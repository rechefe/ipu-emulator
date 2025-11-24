#ifndef IPU_AS_PARSER_H
#define IPU_AS_PARSER_H

#include <string>
#include <vector>
#include <optional>
#include <unordered_map>

namespace ipu_as
{

    struct Inst
    {
        std::string op;
        std::vector<std::string> operands;
        // optional label attached to this instruction (if label and instr on same line)
        std::optional<std::string> label;
    };

    // Parse a single line of assembly. Returns empty optional for blank/comment lines or parse error.
    std::optional<Inst> parse_line(const std::string &line);

    // Parse multiple lines (file contents). Returns vector of parsed instructions (skips blanks/comments).
    std::vector<Inst> parse_lines(const std::vector<std::string> &lines);

    // Parse multiple lines and collect labels. Returns vector of parsed instructions
    // and fills `labels` mapping label name -> instruction index in the returned vector.
    std::vector<Inst> parse_lines_with_labels(const std::vector<std::string> &lines, std::unordered_map<std::string, size_t> &labels);

} // namespace ipu_as

#endif // IPU_AS_PARSER_H
