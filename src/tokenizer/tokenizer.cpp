//
// Created by jake on 3/22/26.
//

#include "fictional_funicular/tokenizer/tokenizer.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>


std::vector<std::int64_t> token::Tokenizer::encode(const std::string& input) {
    std::vector<std::int64_t> tokens;
    const std::size_t input_length = input.length();
    tokens.reserve(input_length);
    tokens.push_back(static_cast<std::int64_t>(input_length));

    std::size_t i = 0;
    while (i < input_length) {
        std::uint64_t token_chunk = 0;

        const std::size_t copy_length = std::min(static_cast<std::size_t>(8), input_length - i);

        std::memcpy(&token_chunk, input.data() + i, copy_length);

        tokens.push_back(token_chunk);
        i += copy_length;
    }

    return tokens;
}

std::string token::Tokenizer::decode(const std::vector<std::int64_t>& tokens) {
    if (tokens.empty()) return {};
    const std::size_t token_length = static_cast<std::size_t>(tokens.at(0));
    std::string output;

    output.reserve(token_length);

    for (std::size_t i = 1; i < tokens.size() && output.size() < token_length; ++i) {
        const std::uint64_t token_chunk = static_cast<std::uint64_t>(tokens.at(i));
        const std::size_t remaining = token_length - output.size();
        const std::size_t chunk_size = std::min(static_cast<std::size_t>(8), remaining);
        for (std::size_t j = 0; j < chunk_size; ++j) {
            auto c = static_cast<char>((token_chunk >> (8*j)) & 0xFF);
            output.push_back(c);
        }
    }
    return output;
}
