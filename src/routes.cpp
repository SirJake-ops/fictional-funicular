//
// Created by jake on 1/13/26.
//

#include "../include/routes.h"
#include <iostream>
#include <sstream>
#include <string>

#include "../include/load_model_inference.h"
#include "httplib.h"
#include "onnxruntime_cxx_api.h"


std::vector<float> run_inference(const std::vector<std::int64_t> &input_ids) {
    static model_inference::ModelInference model("models/decoder_model.onnx");
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

std::vector<int> generate_text(std::vector<std::int64_t> input_ids, int max_tokens = 50257) {
    static model_inference::ModelInference model("models/decoder_model.onnx");
    std::vector<int> generated;

    for (int i = 0; i < max_tokens; i++) {
        auto output = model.run_inference(input_ids);

        int next_token = get_next_token(output);
        generated.push_back(next_token);
        input_ids.push_back(next_token);
        if (next_token == 50256)
            break;
    }
    return generated;
}


void load_routes::Routes::start(const char *host, const int &port) {
    { // GET Requests
        get_hi();
        run_model();
        generate();
        stop_server();
    }
    svr_.listen(host, port);
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
            const std::vector<std::int64_t> input_ids = {15496, 995};
            const auto output = run_inference(input_ids);
            const int next_token = get_next_token(output);

            res.set_content("next_token_id: " + std::to_string(next_token) + "\n" +
                                    "input_length: " + std::to_string(input_ids.size()),
                            "text/plain");
        });
    } catch (const Ort::Exception e) {
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

void load_routes::Routes::generate() {
    try {
        svr_.Get("/generate", [](const httplib::Request &req, httplib::Response &res) {
            std::vector<std::int64_t> input_ids;
            if (req.has_param("input_ids")) {
                std::string ids_str = req.get_param_value("input_ids");
                std::stringstream ss(ids_str);

                std::string token;
                while (std::getline(ss, token, ',')) {
                    input_ids.push_back(std::stoll(token));
                }
            } else {
                input_ids = {15496, 995};
            }
            auto generated = generate_text(input_ids);
        });
    } catch (Ort::Exception e) {
        std::cerr << "Model failed to generate response: " << e.what() << std::endl;
    }
}
