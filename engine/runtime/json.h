// A small host-only JSON reader for runtime manifests and test fixtures. Objects (keys in file order), arrays, strings (\u escapes and
// surrogate pairs to UTF-8), numbers (double: integers exact up to 2^53), true, false, null. Host only; malformed input
// throws std::runtime_error.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace g6v {
struct Json {
    enum Type { Null, Bool, Num, Str, Arr, Obj };
    Type t = Null;
    bool b = false;
    double n = 0;
    std::string s;
    std::vector<Json> a;
    std::vector<std::pair<std::string, Json>> o;

    const Json* find(const std::string& k) const {
        for (const auto& kv : o) if (kv.first == k) return &kv.second;
        return nullptr;
    }
    const Json& at(const std::string& k) const {
        const Json* v = t == Obj ? find(k) : nullptr;
        if (!v) throw std::runtime_error("json: no key \"" + k + "\"");
        return *v;
    }
    const Json& at(size_t i) const {
        if (t != Arr || i >= a.size()) throw std::runtime_error("json: index out of range");
        return a[i];
    }
    size_t size() const { return t == Arr ? a.size() : t == Obj ? o.size() : 0; }
    double num() const { if (t != Num) throw std::runtime_error("json: not a number"); return n; }
    int64_t i64() const {
        const double v = num();
        if (v != std::floor(v) || std::fabs(v) > 9007199254740992.0) throw std::runtime_error("json: not an integer");
        return int64_t(v);
    }
    const std::string& str() const { if (t != Str) throw std::runtime_error("json: not a string"); return s; }

    static Json parse(const std::string& text) {
        size_t i = 0;
        Json v = value(text, i, 0);
        ws(text, i);
        if (i != text.size()) fail("trailing characters");
        return v;
    }

private:
    [[noreturn]] static void fail(const char* what) { throw std::runtime_error(std::string("json: ") + what); }
    static void ws(const std::string& s, size_t& i) {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
    }
    static void utf8(std::string& out, uint32_t c) {
        if (c < 0x80) out += char(c);
        else if (c < 0x800) { out += char(0xC0 | c >> 6); out += char(0x80 | (c & 0x3F)); }
        else if (c < 0x10000) { out += char(0xE0 | c >> 12); out += char(0x80 | (c >> 6 & 0x3F)); out += char(0x80 | (c & 0x3F)); }
        else { out += char(0xF0 | c >> 18); out += char(0x80 | (c >> 12 & 0x3F)); out += char(0x80 | (c >> 6 & 0x3F)); out += char(0x80 | (c & 0x3F)); }
    }
    static uint32_t hex4(const std::string& s, size_t& i) {
        if (i + 4 > s.size()) fail("short \\u escape");
        uint32_t c = 0;
        for (int k = 0; k < 4; ++k) {
            const char h = s[i++];
            c = c * 16 + (h >= '0' && h <= '9' ? h - '0' : h >= 'a' && h <= 'f' ? h - 'a' + 10 : h >= 'A' && h <= 'F' ? h - 'A' + 10 : (fail("bad \\u escape"), 0));
        }
        return c;
    }
    static std::string string(const std::string& s, size_t& i) {
        if (s[i] != '"') fail("expected a string");
        ++i;
        std::string out;
        for (;;) {
            if (i >= s.size()) fail("unterminated string");
            const char c = s[i++];
            if (c == '"') return out;
            if (uint8_t(c) < 0x20) fail("control character in a string");
            if (c != '\\') { out += c; continue; }
            if (i >= s.size()) fail("unterminated escape");
            const char e = s[i++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    uint32_t cp = hex4(s, i);
                    if (cp >= 0xD800 && cp < 0xDC00) {
                        if (i + 2 > s.size() || s[i] != '\\' || s[i + 1] != 'u') fail("lone surrogate");
                        i += 2;
                        const uint32_t lo = hex4(s, i);
                        if (lo < 0xDC00 || lo >= 0xE000) fail("bad surrogate pair");
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    } else if (cp >= 0xDC00 && cp < 0xE000) fail("lone surrogate");
                    utf8(out, cp);
                    break;
                }
                default: fail("bad escape");
            }
        }
    }
    static Json value(const std::string& s, size_t& i, int depth) {
        if (depth > 64) fail("nested too deeply");
        ws(s, i);
        if (i >= s.size()) fail("unexpected end");
        Json v;
        const char c = s[i];
        if (c == '{') {
            v.t = Obj; ++i; ws(s, i);
            if (i < s.size() && s[i] == '}') { ++i; return v; }
            for (;;) {
                ws(s, i);
                if (i >= s.size()) fail("unexpected end");
                std::string k = string(s, i);
                ws(s, i);
                if (i >= s.size() || s[i] != ':') fail("expected ':'");
                ++i;
                v.o.emplace_back(std::move(k), value(s, i, depth + 1));
                ws(s, i);
                if (i < s.size() && s[i] == ',') { ++i; continue; }
                if (i < s.size() && s[i] == '}') { ++i; return v; }
                fail("expected ',' or '}'");
            }
        }
        if (c == '[') {
            v.t = Arr; ++i; ws(s, i);
            if (i < s.size() && s[i] == ']') { ++i; return v; }
            for (;;) {
                v.a.push_back(value(s, i, depth + 1));
                ws(s, i);
                if (i < s.size() && s[i] == ',') { ++i; continue; }
                if (i < s.size() && s[i] == ']') { ++i; return v; }
                fail("expected ',' or ']'");
            }
        }
        if (c == '"') { v.t = Str; v.s = string(s, i); return v; }
        if (s.compare(i, 4, "true") == 0) { v.t = Bool; v.b = true; i += 4; return v; }
        if (s.compare(i, 5, "false") == 0) { v.t = Bool; i += 5; return v; }
        if (s.compare(i, 4, "null") == 0) { i += 4; return v; }
        // number: -? digits [. digits] [e [+-] digits]
        const size_t start = i;
        if (s[i] == '-') ++i;
        auto digits = [&] { const size_t d = i; while (i < s.size() && s[i] >= '0' && s[i] <= '9') ++i; return i > d; };
        if (!digits()) fail("bad value");
        if (i < s.size() && s[i] == '.') { ++i; if (!digits()) fail("bad number"); }
        if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
            ++i;
            if (i < s.size() && (s[i] == '+' || s[i] == '-')) ++i;
            if (!digits()) fail("bad number");
        }
        v.t = Num;
        v.n = std::strtod(s.substr(start, i - start).c_str(), nullptr);
        return v;
    }
};
} // namespace g6v
