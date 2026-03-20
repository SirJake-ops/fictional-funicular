//
// Created by jake on 1/13/26.
//

#include "../include/routes.h"
#include <filesystem>
#include <iostream>
#include <stdexcept>

#include "../include/load_model_inference.h"
#include "onnxruntime_cxx_api.h"

std::vector<float> run_inference(const std::vector<std::int64_t> &input_ids) {
    static const std::filesystem::path model_path = "models/decoder_model.onnx";

    if (!std::filesystem::exists(model_path)) {
        throw std::runtime_error("Missing model file at " + model_path.string());
    }

    static model_inference::ModelInference model(model_path.string());
    auto out_put = model.run_inference(input_ids);
    return out_put;
}

int get_next_token(const std::vector<float> &logits, const std::size_t vocab_size = 50257) {
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

void load_routes::Routes::start(const char *host, const int &port) {
    { // GET Requests
        get_hi();
        run_model();
        stop_server();
    }
    if (!svr_.listen(host, port)) {
        std::cerr << "Server failed to listen on " << host << ":" << port << std::endl;
    }
}

void load_routes::Routes::get_hi() {
    try {
        svr_.Get("/hi", [](const httplib::Request &req, httplib::Response &res) {
            res.set_content("Hello from the class", "text/plain");
        });
    } catch (...) {
        std::cerr << "Request not valid" << std::endl;
    }
}

void load_routes::Routes::run_model() {
    try {
        svr_.Get("/run_model", [](const httplib::Request &req, httplib::Response &res) {
            try {
                const std::vector<std::int64_t> input_ids = {15496, 995};
                const auto output = run_inference(input_ids);
                const int next_token = get_next_token(output);

                res.set_content("next_token_id: " + std::to_string(next_token) + "\n" +
                                        "input_length: " + std::to_string(input_ids.size()),
                                "text/plain");
            } catch (const Ort::Exception &e) {
                res.status = 500;
                res.set_content(std::string("Model failed to run: ") + e.what(), "text/plain");
            } catch (const std::exception &e) {
                res.status = 500;
                res.set_content(e.what(), "text/plain");
            }
        });
    } catch (const Ort::Exception &e) {
        std::cerr << "Model failed to run: " << e.what() << std::endl;
    }
}

void load_routes::Routes::stop_server() {
    try {
        svr_.Get("/stop", [&](const httplib::Request &req, httplib::Response &res) { svr_.stop(); });
    } catch (...) {
        std::cerr << "Server cannot stop." << std::endl;
    }
}
