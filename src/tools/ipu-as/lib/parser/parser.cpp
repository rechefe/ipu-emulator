#include "parser.h"
#include <algorithm>
#include <cctype>

using namespace ipu_as;

static inline std::string trim(const std::string &s)
{
    size_t a = 0;
    while (a < s.size() && std::isspace((unsigned char)s[a]))
        ++a;
    size_t b = s.size();
    while (b > a && std::isspace((unsigned char)s[b - 1]))
        --b;
    return s.substr(a, b - a);
}

static inline void split_tokens(const std::string &s, std::vector<std::string> &out)
{
    std::string cur;
    for (size_t i = 0; i < s.size(); ++i)
    {
        char c = s[i];
        if (c == ',' || std::isspace((unsigned char)c))
        {
            if (!cur.empty())
            {
                out.push_back(cur);
                cur.clear();
            }
            continue;
        }
        cur.push_back(c);
    }
    if (!cur.empty())
        out.push_back(cur);
}

std::optional<Inst> ipu_as::parse_line(const std::string &line)
{
    std::string s = trim(line);
    if (s.empty())
        return std::nullopt;
    // allow comments starting with '#', ';' or '//' (line-start or after tokens)
    // find comment start
    size_t comment_pos = std::string::npos;
    for (size_t i = 0; i + 1 < s.size(); ++i)
    {
        if (s[i] == '/' && s[i + 1] == '/')
        {
            comment_pos = i;
            break;
        }
    }
    if (comment_pos == std::string::npos)
    {
        for (size_t i = 0; i < s.size(); ++i)
        {
            if (s[i] == '#' || s[i] == ';')
            {
                comment_pos = i;
                break;
            }
        }
    }
    if (comment_pos != std::string::npos)
        s = trim(s.substr(0, comment_pos));
    if (s.empty())
        return std::nullopt;

    // tokenise op and operands
    std::vector<std::string> tokens;
    split_tokens(s, tokens);
    if (tokens.empty())
        return std::nullopt;
    Inst inst;
    // lowercase op
    inst.op = tokens[0];
    std::transform(inst.op.begin(), inst.op.end(), inst.op.begin(), [](unsigned char c)
                   { return std::tolower(c); });
    for (size_t i = 1; i < tokens.size(); ++i)
        inst.operands.push_back(tokens[i]);
    return inst;
}

std::vector<Inst> ipu_as::parse_lines(const std::vector<std::string> &lines)
{
    std::vector<Inst> out;
    for (auto &l : lines)
    {
        auto r = parse_line(l);
        if (r)
            out.push_back(*r);
    }
    return out;
}
