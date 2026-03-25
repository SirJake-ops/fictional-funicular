//
// Created by jake on 1/13/26.
//

#ifndef LLM_INFERENCE_ENGINE_ROUTES_H
#define LLM_INFERENCE_ENGINE_ROUTES_H
#include "httplib.h"
#include <filesystem>
#include <functional>
#include <vector>

#include "fictional_funicular/tokenizer/tokenizer.h"


enum class REST;
namespace load_routes {
    using InferenceRunner =
        std::function<std::vector<float>(const std::vector<std::int64_t> &)>;
    struct GenerationResult {
        std::string _prompt;
        std::string _generated_text;
        std::vector<std::int64_t> _prompt_token_ids;
        std::vector<std::int64_t> _generated_token_ids;
        std::size_t _cache_layers{0};
        std::size_t _cache_sequence_length{0};
    };
    using GenerationRunner =
        std::function<GenerationResult(const std::string &, std::size_t)>;

    std::vector<float> run_inference_with_model(
        const std::vector<std::int64_t> &input_ids,
        const std::filesystem::path &model_path = "models/model.onnx");

    std::vector<float> run_inference(const std::vector<std::int64_t> &input_ids);
    GenerationResult generate_with_model(const std::string &prompt,
                                         std::size_t max_new_tokens,
                                         const std::filesystem::path &model_path = "models/model.onnx");
    GenerationResult generate(const std::string &prompt, std::size_t max_new_tokens);

    int get_next_token(const std::vector<float> &logits,
                       std::size_t vocab_size = 50257);

    void handle_hi_request(const httplib::Request &req, httplib::Response &res);

    void handle_run_model_request(const httplib::Request &req, httplib::Response &res,
                                  const InferenceRunner &runner = run_inference);
    void handle_generate_request(const httplib::Request &req, httplib::Response &res,
                                 const GenerationRunner &runner = generate);

    void register_routes(httplib::Server &server,
                         const InferenceRunner &runner = run_inference,
                         const GenerationRunner &generation_runner = generate);

    class Routes {
    public:
        static Routes &get_route_instance() {
            static Routes instance;
            return instance;
        }

        Routes(Routes &) = delete;
        Routes &operator=(const Routes &) = delete;

        void start(const char *host, const int &port);
        void handle_run_model_request(const httplib::Request &req,
                                      httplib::Response &res,
                                      const InferenceRunner &runner = run_inference);
        void handle_generate_request(const httplib::Request &req,
                                     httplib::Response &res,
                                     const GenerationRunner &runner = generate);

    private:
        Routes() = default;
        httplib::Server _server;
        httplib::ErrorLogger _logger;
        token::Tokenizer _tokenizer;
    };
} // namespace load_routes

#endif // LLM_INFERENCE_ENGINE_ROUTES_H
