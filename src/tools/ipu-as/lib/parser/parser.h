#ifndef IPU_AS_PARSER_H
#define IPU_AS_PARSER_H

#include <string>
#include <vector>
#include <optional>

namespace ipu_as {

struct Inst {
    std::string op;
    std::vector<std::string> operands;
};

// Parse a single line of assembly. Returns empty optional for blank/comment lines or parse error.
std::optional<Inst> parse_line(const std::string &line);

// Parse multiple lines (file contents). Returns vector of parsed instructions (skips blanks/comments).
std::vector<Inst> parse_lines(const std::vector<std::string> &lines);

} // namespace ipu_as

#endif // IPU_AS_PARSER_H
