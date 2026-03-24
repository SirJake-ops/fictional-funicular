//
// Created by jake on 3/22/26.
//
#pragma once

#include <vector>
#include <string>
#include <cstdint>

namespace token {
    class Tokenizer {
    public:
        explicit Tokenizer() = default;
        Tokenizer(const Tokenizer&) = default;
        Tokenizer(Tokenizer&&) = default;
        ~Tokenizer() = default;
        Tokenizer& operator=(const Tokenizer&) = default;


        std::vector<std::int64_t> encode(const std::string &input);
        std::string decode(const std::vector<int64_t> &tokens);
    };
}



