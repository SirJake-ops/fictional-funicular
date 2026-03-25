//
// Created by jake on 1/13/26.
//

#include "fictional_funicular/http/routes.h"

#include <filesystem>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "fictional_funicular/inference/model_inference.h"
#include "onnxruntime_cxx_api.h"

namespace {
constexpr std::size_t kDefaultMaxNewTokens = 16;

model_inference::ModelInference &get_model(const std::filesystem::path &model_path) {
    static std::filesystem::path loaded_model_path;
    static std::unique_ptr<model_inference::ModelInference> model;

    if (!model || loaded_model_path != model_path) {
        model = std::make_unique<model_inference::ModelInference>(model_path);
        loaded_model_path = model_path;
    }

    return *model;
}

std::string json_escape(const std::string &input) {
    std::ostringstream escaped;

    for (const unsigned char c : input) {
        switch (c) {
            case '\\':
                escaped << "\\\\";
                break;
            case '"':
                escaped << "\\\"";
                break;
            case '\n':
                escaped << "\\n";
                break;
            case '\r':
                escaped << "\\r";
                break;
            case '\t':
                escaped << "\\t";
                break;
            default:
                if (c < 0x20) {
                    escaped << "\\u"
                            << "00"
                            << "0123456789abcdef"[c >> 4]
                            << "0123456789abcdef"[c & 0x0F];
                } else {
                    escaped << static_cast<char>(c);
                }
                break;
        }
    }

    return escaped.str();
}

std::string encode_token_ids_json(const std::vector<std::int64_t> &token_ids) {
    std::ostringstream json;
    json << "[";

    for (std::size_t i = 0; i < token_ids.size(); ++i) {
        if (i != 0) {
            json << ",";
        }
        json << token_ids[i];
    }

    json << "]";
    return json.str();
}

std::size_t parse_max_new_tokens(const httplib::Request &req) {
    if (!req.has_param("max_tokens")) {
        return kDefaultMaxNewTokens;
    }

    const auto parsed_value = std::stoul(req.get_param_value("max_tokens"));
    if (parsed_value == 0) {
        throw std::invalid_argument("max_tokens must be greater than 0");
    }

    return parsed_value;
}

std::string generation_result_to_json(const load_routes::GenerationResult &result) {
    std::ostringstream json;
    json << "{"
         << "\"prompt\":\"" << json_escape(result._prompt) << "\","
         << "\"generated_text\":\"" << json_escape(result._generated_text) << "\","
         << "\"response_text\":\"" << json_escape(result._prompt + result._generated_text) << "\","
         << "\"prompt_token_ids\":" << encode_token_ids_json(result._prompt_token_ids) << ","
         << "\"generated_token_ids\":" << encode_token_ids_json(result._generated_token_ids) << ","
         << "\"prompt_token_count\":" << result._prompt_token_ids.size() << ","
         << "\"generated_token_count\":" << result._generated_token_ids.size() << ","
         << "\"cache_layers\":" << result._cache_layers << ","
         << "\"cache_sequence_length\":" << result._cache_sequence_length
         << "}";
    return json.str();
}
} // namespace

std::vector<float> load_routes::run_inference_with_model(
    const std::vector<std::int64_t> &input_ids,
    const std::filesystem::path &model_path) {
    auto &model = get_model(model_path);
    return model.run_inference(input_ids, static_cast<int>(model.get_required_cache_layer_count()));
}

std::vector<float> load_routes::run_inference(
    const std::vector<std::int64_t> &input_ids) {
    return run_inference_with_model(input_ids, "models/model.onnx");
}

load_routes::GenerationResult load_routes::generate_with_model(
    const std::string &prompt,
    const std::size_t max_new_tokens,
    const std::filesystem::path &model_path) {
    if (prompt.empty()) {
        throw std::invalid_argument("Prompt body must not be empty");
    }

    auto &model = get_model(model_path);
    token::Tokenizer tokenizer;
    const auto prompt_token_ids = tokenizer.encode(prompt);
    const auto cache_layer_count = static_cast<int>(model.get_required_cache_layer_count());

    model.reset_cache();

    auto logits = model.run_inference(prompt_token_ids, cache_layer_count);
    std::vector<std::int64_t> generated_token_ids;
    generated_token_ids.reserve(max_new_tokens);

    for (std::size_t i = 0; i < max_new_tokens; ++i) {
        const auto next_token = static_cast<std::int64_t>(get_next_token(logits));
        generated_token_ids.push_back(next_token);

        if (i + 1 == max_new_tokens) {
            break;
        }

        logits = model.run_inference({next_token}, cache_layer_count);
    }

    return GenerationResult{
        ._prompt = prompt,
        ._generated_text = tokenizer.decode(generated_token_ids),
        ._prompt_token_ids = prompt_token_ids,
        ._generated_token_ids = generated_token_ids,
        ._cache_layers = model.get_required_cache_layer_count(),
        ._cache_sequence_length = model.get_cached_sequence_length(),
    };
}

load_routes::GenerationResult load_routes::generate(const std::string &prompt,
                                                    const std::size_t max_new_tokens) {
    return generate_with_model(prompt, max_new_tokens, "models/model.onnx");
}

int load_routes::get_next_token(const std::vector<float> &logits,
                                const std::size_t vocab_size) {
    if (vocab_size == 0 || logits.size() < vocab_size) {
        throw std::invalid_argument("Logits must contain at least one full vocabulary window");
    }

    const std::size_t last_token_start = logits.size() - vocab_size;

    int best_token{0};
    float best_score = logits.at(last_token_start);

    for (std::size_t i = 1; i < vocab_size; i++) {
        if (logits.at(last_token_start + i) > best_score) {
            best_score = logits.at(last_token_start + i);
            best_token = static_cast<int>(i);
        }
    }
    return best_token;
}

void load_routes::register_routes(httplib::Server &server,
                                  const InferenceRunner &runner,
                                  const GenerationRunner &generation_runner) {
    server.Get("/hi", handle_hi_request);
    server.Get("/run_model", [runner](const httplib::Request &req, httplib::Response &res) {
        Routes::get_route_instance().handle_run_model_request(req, res, runner);
    });
    server.Post("/generate", [generation_runner](const httplib::Request &req, httplib::Response &res) {
        Routes::get_route_instance().handle_generate_request(req, res, generation_runner);
    });
    server.Get("/stop", [&](const httplib::Request &, httplib::Response &) { server.stop(); });
}

void load_routes::handle_hi_request(const httplib::Request &, httplib::Response &res) {
    res.status = 200;
    res.set_content("Hello from the class", "text/plain");
}

void load_routes::handle_run_model_request(const httplib::Request &req,
                                           httplib::Response &res,
                                           const InferenceRunner &runner) {
    Routes::get_route_instance().handle_run_model_request(req, res, runner);
}

void load_routes::handle_generate_request(const httplib::Request &req,
                                          httplib::Response &res,
                                          const GenerationRunner &runner) {
    Routes::get_route_instance().handle_generate_request(req, res, runner);
}

void load_routes::Routes::handle_run_model_request(const httplib::Request &req,
                                                   httplib::Response &res,
                                                   const InferenceRunner &runner) {
    try {
        const std::vector<std::int64_t> input_ids = _tokenizer.encode(req.body);
        const auto output = runner(input_ids);
        const int next_token = get_next_token(output);
        const std::string decoded = _tokenizer.decode({next_token});

        res.status = 200;
        res.set_content("next_token_id: " + std::to_string(next_token) + "\n" +
                        "decoded: " + decoded + "\n" +
                        "input_length: " + std::to_string(input_ids.size()),
                        "text/plain");
    } catch (const Ort::Exception &e) {
        res.status = 500;
        res.set_content(std::string("Model failed to run: ") + e.what(), "text/plain");
    } catch (const std::exception &e) {
        res.status = 500;
        res.set_content(e.what(), "text/plain");
    }
}

void load_routes::Routes::handle_generate_request(const httplib::Request &req,
                                                  httplib::Response &res,
                                                  const GenerationRunner &runner) {
    try {
        const auto max_new_tokens = parse_max_new_tokens(req);
        const auto result = runner(req.body, max_new_tokens);

        res.status = 200;
        res.set_content(generation_result_to_json(result), "application/json");
    } catch (const Ort::Exception &e) {
        res.status = 500;
        res.set_content(std::string("Model failed to run: ") + e.what(), "text/plain");
    } catch (const std::invalid_argument &e) {
        res.status = 400;
        res.set_content(e.what(), "text/plain");
    } catch (const std::exception &e) {
        res.status = 500;
        res.set_content(e.what(), "text/plain");
    }
}

void load_routes::Routes::start(const char *host, const int &port) {
    register_routes(_server);
    if (!_server.listen(host, port)) {
        std::cerr << "Server failed to listen on " << host << ":" << port << std::endl;
    }
}
