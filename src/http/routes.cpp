//
// Created by jake on 1/13/26.
//

#include "fictional_funicular/http/routes.h"
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "fictional_funicular/inference/model_inference.h"
#include "onnxruntime_cxx_api.h"

namespace {
std::filesystem::path resolve_model_path(std::filesystem::path model_path) {
    if (model_path.is_absolute() && std::filesystem::exists(model_path)) {
        return model_path;
    }

    const auto project_root = std::filesystem::path{LLM_INFERENCE_ENGINE_PROJECT_ROOT};
    const auto models_dir = project_root / "models";

    if (!model_path.empty()) {
        const auto from_cwd = std::filesystem::absolute(model_path);
        if (std::filesystem::exists(from_cwd)) {
            return from_cwd;
        }

        const auto from_models_dir = models_dir / model_path.filename();
        if (std::filesystem::exists(from_models_dir)) {
            return from_models_dir;
        }
    }

    if (std::filesystem::exists(models_dir)) {
        for (const auto &entry : std::filesystem::directory_iterator(models_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".onnx") {
                return entry.path();
            }
        }
    }

    throw std::runtime_error(
        "Missing ONNX model. Expected " + (models_dir / model_path.filename()).string() +
        " or any .onnx file inside " + models_dir.string());
}
}  // namespace

std::vector<float> load_routes::run_inference_with_model(
    const std::vector<std::int64_t> &input_ids,
    const std::filesystem::path &model_path) {
    const auto resolved_model_path = resolve_model_path(model_path);

    static std::filesystem::path loaded_model_path;
    static std::unique_ptr<model_inference::ModelInference> model;

    if (!model || loaded_model_path != resolved_model_path) {
        model = std::make_unique<model_inference::ModelInference>(
            resolved_model_path.string());
        loaded_model_path = resolved_model_path;
    }

    return model->run_inference(input_ids, 12);
}

std::vector<float> load_routes::run_inference(
    const std::vector<std::int64_t> &input_ids) {
    return run_inference_with_model(input_ids, std::filesystem::relative("models/on"));
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
                                  const InferenceRunner &runner) {
    server.Get("/hi", handle_hi_request);

    server.Get("/run_model", [runner](const httplib::Request &req, httplib::Response &res) {
        Routes::get_route_instance().handle_run_model_request(req, res, runner);
    });

    server.Get("/stop", [&](const httplib::Request &, httplib::Response &) { server.stop(); });
}

void load_routes::handle_hi_request(const httplib::Request &, httplib::Response &res) {
    res.status = 200;
    res.set_content("Hello from the class", "text/plain");
}

void load_routes::Routes::handle_run_model_request(const httplib::Request &req,
                                                   httplib::Response &res,
                                                   const InferenceRunner &runner) {
    try {

        std::string prompt_input = req.body;

        const std::vector<std::int64_t> input_ids = _tokenizer.encode(prompt_input);

        const auto output = runner(input_ids);
        const int next_token = get_next_token(output);
        std::string decoded = _tokenizer.decode({next_token});


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

void load_routes::Routes::start(const char *host, const int &port) {
    register_routes(svr_);
    if (!svr_.listen(host, port)) {
        std::cerr << "Server failed to listen on " << host << ":" << port << std::endl;
    }
}
