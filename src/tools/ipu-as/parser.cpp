#include "parser.h"
#include <algorithm>
#include <cctype>
#include <unordered_map>
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
    inst.label = std::nullopt;
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
    std::unordered_map<std::string, size_t> labels;
    return parse_lines_with_labels(lines, labels);
}

std::vector<Inst> ipu_as::parse_lines_with_labels(const std::vector<std::string> &lines, std::unordered_map<std::string, size_t> &labels)
{
    std::vector<Inst> out;
    labels.clear();
    for (const auto &orig : lines)
    {
        std::string s = trim(orig);
        if (s.empty())
            continue;
        // strip comments first (reuse logic)
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
            continue;

        // detect leading label: token ending with ':' before any whitespace
        // find first non-space token
        size_t pos = 0;
        while (pos < s.size() && std::isspace((unsigned char)s[pos]))
            ++pos;
        size_t colon = std::string::npos;
        for (size_t i = pos; i < s.size(); ++i)
        {
            if (s[i] == ':') { colon = i; break; }
            if (std::isspace((unsigned char)s[i])) break; // not a label
        }
        if (colon != std::string::npos)
        {
            // extract label name
            std::string lbl = trim(s.substr(pos, colon - pos));
            if (!lbl.empty())
            {
                // validate label name: starts with letter or underscore, then alnum/_
                bool ok = true;
                if (!(std::isalpha((unsigned char)lbl[0]) || lbl[0] == '_'))
                    ok = false;
                for (size_t i = 1; i < lbl.size() && ok; ++i)
                {
                    if (!(std::isalnum((unsigned char)lbl[i]) || lbl[i] == '_'))
                        ok = false;
                }
                if (ok)
                {
                    // map label to next instruction index
                    labels[lbl] = out.size();
                }
            }
            // remainder after colon
            std::string rem = trim(s.substr(colon + 1));
            if (rem.empty())
                continue; // label-only line
            // parse remainder as instruction
            auto r = parse_line(rem);
            if (r)
            {
                // if an instruction exists on same line, attach the label to it
                r->label = std::optional<std::string>(trim(s.substr(pos, colon - pos)));
                out.push_back(*r);
            }
            continue;
        }

        // normal instruction line
        auto r = parse_line(s);
        if (r)
            out.push_back(*r);
    }
    return out;
}
