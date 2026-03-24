//
// Created by jake on 1/14/26.
//

#include "fictional_funicular/inference/model_inference.h"

#include <filesystem>
#include <stdexcept>
#include <vector>

namespace {
std::filesystem::path resolve_model_path(const std::filesystem::path &model_path) {
    if (model_path.is_absolute() && std::filesystem::exists(model_path)) {
        return model_path;
    }

    const auto project_root = std::filesystem::path{LLM_INFERENCE_ENGINE_PROJECT_ROOT};

    if (!model_path.empty()) {
        const auto from_cwd = std::filesystem::absolute(model_path);
        if (std::filesystem::exists(from_cwd)) {
            return from_cwd;
        }

        const auto from_project_root = project_root / model_path;
        if (std::filesystem::exists(from_project_root)) {
            return from_project_root;
        }

        const auto from_models_dir = project_root / "models" / model_path.filename();
        if (std::filesystem::exists(from_models_dir)) {
            return from_models_dir;
        }

        throw std::runtime_error(
            "Model file does not exist: " + model_path.string());
    }

    const auto models_dir = project_root / "models";
    if (std::filesystem::exists(models_dir)) {
        for (const auto &entry : std::filesystem::directory_iterator(models_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".onnx") {
                return entry.path();
            }
        }
    }

    throw std::runtime_error("Model file does not exist and no .onnx file was found in " +
                             models_dir.string());
}
} // namespace

model_inference::ModelInference::ModelInference(const std::filesystem::path &path_to_model)
    : env_(ORT_LOGGING_LEVEL_WARNING, "GPT2Inference"),
      session_(nullptr) {
    ort_session_options_.SetIntraOpNumThreads(4);
    ort_session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    const auto resolved_model_path = resolve_model_path(path_to_model);
    session_ = Ort::Session(env_, resolved_model_path.c_str(), ort_session_options_);
}

std::vector<float> model_inference::ModelInference::run_inference(
    const std::vector<std::int64_t> &input_ids, const int &number_of_layers) {

    if (_layer_cache.size() != static_cast<std::size_t>(number_of_layers)) {
        _layer_cache.clear();
        _layer_cache.resize(static_cast<std::size_t>(number_of_layers));
    }

    if (input_ids.empty()) {
        return {};
    }

    const Ort::AllocatorWithDefaultOptions allocator;

    const std::vector<std::int64_t> input_shape = {
        1, static_cast<std::int64_t>(input_ids.size())
    };
    std::vector<std::int64_t> input_ids_copy = input_ids;
    std::vector<std::int64_t> attention_mask(input_ids.size(), 1);
    std::vector<std::int64_t> position_ids(input_ids.size());
    for (std::size_t i = 0; i < position_ids.size(); ++i) {
        position_ids[i] = static_cast<std::int64_t>(i);
    }

    const auto memory_info =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<std::int64_t>(
        memory_info, input_ids_copy.data(), input_ids_copy.size(),
        input_shape.data(), input_shape.size());

    Ort::Value mask_tensor = Ort::Value::CreateTensor(
        memory_info, attention_mask.data(), attention_mask.size(),
        input_shape.data(), input_shape.size());

    Ort::Value position_tensor = Ort::Value::CreateTensor<std::int64_t>(
        memory_info, position_ids.data(), position_ids.size(),
        input_shape.data(), input_shape.size());

    std::vector<std::string> input_name_storage;
    std::vector<const char *> input_names;
    std::vector<std::string> output_name_storage;
    std::vector<const char *> output_names;
    std::vector<Ort::Value> input_tensors;

    input_name_storage.reserve(3 + number_of_layers * 2);
    input_tensors.reserve(3 + number_of_layers * 2);
    input_names.reserve(3 + number_of_layers * 2);

    input_name_storage.push_back("input_ids");
    input_tensors.push_back(std::move(input_tensor));

    input_name_storage.push_back("attention_mask");
    input_tensors.push_back(std::move(mask_tensor));

    input_name_storage.push_back("position_ids");
    input_tensors.push_back(std::move(position_tensor));

    for (int i = 0; i < number_of_layers; ++i) {
        if (!_layer_cache.at(i)._keys.empty() && !_layer_cache.at(i)._values.empty()) {
            input_name_storage.push_back("past_key_values." + std::to_string(i) + ".key");
            input_tensors.push_back(std::move(_layer_cache[i]._keys.back()));

            input_name_storage.push_back("past_key_values." + std::to_string(i) + ".value");
            input_tensors.push_back(std::move(_layer_cache[i]._values.back()));
        }
    }

    for (const auto &name : input_name_storage) {
        input_names.push_back(name.c_str());
    }

    const std::size_t output_count = session_.GetOutputCount();
    output_name_storage.reserve(output_count);
    output_names.reserve(output_count);
    for (std::size_t i = 0; i < output_count; ++i) {
        output_name_storage.push_back(session_.GetOutputNameAllocated(i, allocator).get());
    }
    for (const auto &name : output_name_storage) {
        output_names.push_back(name.c_str());
    }

    auto output_tensors =
            session_.Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(),
                         input_tensors.size(), output_names.data(), output_names.size());

    const std::size_t expected_cache_outputs = 1 + static_cast<std::size_t>(number_of_layers) * 2;
    if (output_tensors.size() >= expected_cache_outputs) {
        for (int i = 0; i < number_of_layers; ++i) {
            const std::size_t key_index = 1 + static_cast<std::size_t>(i) * 2;
            const std::size_t value_index = 2 + static_cast<std::size_t>(i) * 2;

            _layer_cache[i]._keys.clear();
            _layer_cache[i]._values.clear();

            _layer_cache[i]._keys.push_back(std::move(output_tensors[key_index]));
            _layer_cache[i]._values.push_back(std::move(output_tensors[value_index]));
        }
    }

    float *output_data = output_tensors[0].GetTensorMutableData<float>();
    const auto output_shape =
            output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

    std::size_t output_size = 1;
    for (const auto &dim: output_shape) {
        output_size *= dim;
    }

    return std::vector<float>(output_data, output_data + output_size);
}
