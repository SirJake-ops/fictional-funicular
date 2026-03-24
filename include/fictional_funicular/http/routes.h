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

    std::vector<float> run_inference_with_model(
        const std::vector<std::int64_t> &input_ids,
        const std::filesystem::path &model_path = "models/decoder_model.onnx");

    std::vector<float> run_inference(const std::vector<std::int64_t> &input_ids);

    int get_next_token(const std::vector<float> &logits,
                       std::size_t vocab_size = 50257);

    void handle_hi_request(const httplib::Request &req, httplib::Response &res);

    void handle_run_model_request(const httplib::Request &req, httplib::Response &res,
                                  const InferenceRunner &runner = run_inference);

    void register_routes(httplib::Server &server,
                         const InferenceRunner &runner = run_inference);

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

    private:
        Routes() = default;
        httplib::Server svr_;
        httplib::ErrorLogger logger_;
        token::Tokenizer _tokenizer;
    };
} // namespace load_routes

#endif // LLM_INFERENCE_ENGINE_ROUTES_H
