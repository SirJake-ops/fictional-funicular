//
// Created by jake on 3/22/26.
//

#include "../../include/fictional_funicular/tokenizer/tokenizer.h"

#include <cstdint>
#include <string>
#include <vector>


std::vector<std::int64_t> token::Tokenizer::encode(const std::string& input) {
    std::vector<std::int64_t> tokens;
    for (const auto& c : input) {
        tokens.push_back(static_cast<std::int64_t>(c));
    }

    return tokens;
}
