//
// Created by jake on 1/14/26.
//

#ifndef LLM_INFERENCE_ENGINE_LOAD_MODEL_INFERENCE_H
#define LLM_INFERENCE_ENGINE_LOAD_MODEL_INFERENCE_H
#include "onnxruntime_cxx_api.h"

namespace model_inference {
    class ModelInference {
    private:
        Ort::Env env_;
        Ort::Session session_;
        Ort::SessionOptions ort_session_options_;

    public:
        explicit ModelInference(const std::string &path_to_model) :
            env_(ORT_LOGGING_LEVEL_WARNING, "GPT2Inference"), session_(nullptr) {
            ort_session_options_.SetIntraOpNumThreads(4);
            ort_session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

            session_ = Ort::Session(env_, path_to_model.c_str(), ort_session_options_);
        }

        void printModelInfo() const {
            std::size_t num_of_inputs = session_.GetInputCount();
            // TODO: This method is for showing general info, need to figure out what I want to display.
        }

        std::vector<float> run_inference(const std::vector<std::int64_t> &inputs);
    };
} // namespace model_inference

#endif // LLM_INFERENCE_ENGINE_LOAD_MODEL_INFERENCE_H
