//
// Created by jake on 1/14/26.
//

#include "fictional_funicular/inference/model_inference.h"

#include <filesystem>
#include <string>
#include <string_view>
#include <stdexcept>
#include <vector>

namespace {
constexpr std::int64_t kDefaultAttentionHeadCount = 16;
constexpr std::int64_t kDefaultAttentionHeadSize = 64;

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

std::size_t inspect_required_cache_layer_count(Ort::Session &session) {
    const Ort::AllocatorWithDefaultOptions allocator;
    std::size_t layer_count = 0;

    for (std::size_t i = 0; i < session.GetInputCount(); ++i) {
        const auto input_name = session.GetInputNameAllocated(i, allocator);
        const std::string_view name{input_name.get()};
        if (name.starts_with("past_key_values.") && name.ends_with(".key")) {
            ++layer_count;
        }
    }

    return layer_count;
}

std::int64_t get_past_sequence_length(const cache::KVCache &layer_cache) {
    if (layer_cache._keys_cache.empty()) {
        return 0;
    }

    const auto shape = layer_cache._keys_cache.front().GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() < 3) {
        return 0;
    }

    return shape[2] < 0 ? 0 : shape[2];
}
} // namespace

model_inference::ModelInference::ModelInference(const std::filesystem::path &path_to_model)
    : _env(ORT_LOGGING_LEVEL_WARNING, "GPT2Inference"),
      _session(nullptr) {
    _ort_session_options.SetIntraOpNumThreads(4);
    _ort_session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    const auto resolved_model_path = resolve_model_path(path_to_model);
    _session = Ort::Session(_env, resolved_model_path.c_str(), _ort_session_options);
    _required_cache_layer_count = inspect_required_cache_layer_count(_session);
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
    const std::size_t required_cache_layers = _required_cache_layer_count;
    const std::size_t active_cache_layers =
        std::max(required_cache_layers, static_cast<std::size_t>(number_of_layers));
    const std::int64_t past_sequence_length =
        _layer_cache.empty() ? 0 : get_past_sequence_length(_layer_cache.front());

    const std::vector<std::int64_t> input_shape = {
        1, static_cast<std::int64_t>(input_ids.size())
    };
    std::vector<std::int64_t> input_ids_copy = input_ids;
    std::vector<std::int64_t> attention_mask(
        static_cast<std::size_t>(past_sequence_length + static_cast<std::int64_t>(input_ids.size())), 1);
    const std::vector<std::int64_t> attention_mask_shape = {
        1, static_cast<std::int64_t>(attention_mask.size())
    };
    std::vector<std::int64_t> position_ids(input_ids.size());
    for (std::size_t i = 0; i < position_ids.size(); ++i) {
        position_ids[i] = past_sequence_length + static_cast<std::int64_t>(i);
    }
    const std::vector<std::int64_t> empty_cache_shape = {
        1, kDefaultAttentionHeadCount, past_sequence_length, kDefaultAttentionHeadSize
    };
    std::vector<std::vector<float>> empty_cache_buffers(active_cache_layers * 2);

    const auto memory_info =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<std::int64_t>(
        memory_info, input_ids_copy.data(), input_ids_copy.size(),
        input_shape.data(), input_shape.size());

    Ort::Value mask_tensor = Ort::Value::CreateTensor(
        memory_info, attention_mask.data(), attention_mask.size(),
        attention_mask_shape.data(), attention_mask_shape.size());

    Ort::Value position_tensor = Ort::Value::CreateTensor<std::int64_t>(
        memory_info, position_ids.data(), position_ids.size(),
        input_shape.data(), input_shape.size());

    std::vector<std::string> input_name_storage;
    std::vector<const char *> input_names;
    std::vector<std::string> output_name_storage;
    std::vector<const char *> output_names;
    std::vector<Ort::Value> input_tensors;

    input_name_storage.reserve(3 + active_cache_layers * 2);
    input_tensors.reserve(3 + active_cache_layers * 2);
    input_names.reserve(3 + active_cache_layers * 2);

    input_name_storage.push_back("input_ids");
    input_tensors.push_back(std::move(input_tensor));

    input_name_storage.push_back("attention_mask");
    input_tensors.push_back(std::move(mask_tensor));

    input_name_storage.push_back("position_ids");
    input_tensors.push_back(std::move(position_tensor));

    for (std::size_t i = 0; i < active_cache_layers; ++i) {
        input_name_storage.push_back("past_key_values." + std::to_string(i) + ".key");
        if (i < _layer_cache.size() && !_layer_cache[i]._keys_cache.empty()) {
            input_tensors.push_back(std::move(_layer_cache[i]._keys_cache.back()));
        } else {
            input_tensors.push_back(Ort::Value::CreateTensor<float>(
                memory_info, empty_cache_buffers[i * 2].data(), empty_cache_buffers[i * 2].size(),
                empty_cache_shape.data(), empty_cache_shape.size()));
        }

        input_name_storage.push_back("past_key_values." + std::to_string(i) + ".value");
        if (i < _layer_cache.size() && !_layer_cache[i]._values_cache.empty()) {
            input_tensors.push_back(std::move(_layer_cache[i]._values_cache.back()));
        } else {
            input_tensors.push_back(Ort::Value::CreateTensor<float>(
                memory_info, empty_cache_buffers[i * 2 + 1].data(), empty_cache_buffers[i * 2 + 1].size(),
                empty_cache_shape.data(), empty_cache_shape.size()));
        }
    }

    for (const auto &name : input_name_storage) {
        input_names.push_back(name.c_str());
    }

    const std::size_t output_count = _session.GetOutputCount();
    output_name_storage.reserve(output_count);
    output_names.reserve(output_count);
    for (std::size_t i = 0; i < output_count; ++i) {
        output_name_storage.push_back(_session.GetOutputNameAllocated(i, allocator).get());
    }
    for (const auto &name : output_name_storage) {
        output_names.push_back(name.c_str());
    }

    auto output_tensors =
            _session.Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(),
                         input_tensors.size(), output_names.data(), output_names.size());

    const std::size_t cache_layers_to_store = std::min(
        static_cast<std::size_t>(number_of_layers), required_cache_layers);
    const std::size_t expected_cache_outputs = 1 + required_cache_layers * 2;
    if (output_tensors.size() >= expected_cache_outputs) {
        for (std::size_t i = 0; i < cache_layers_to_store; ++i) {
            const std::size_t key_index = 1 + static_cast<std::size_t>(i) * 2;
            const std::size_t value_index = 2 + static_cast<std::size_t>(i) * 2;

            _layer_cache[i]._keys_cache.clear();
            _layer_cache[i]._values_cache.clear();

            _layer_cache[i]._keys_cache.push_back(std::move(output_tensors[key_index]));
            _layer_cache[i]._values_cache.push_back(std::move(output_tensors[value_index]));
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

void model_inference::ModelInference::reset_cache() {
    for (auto &layer_cache : _layer_cache) {
        layer_cache.clear();
    }
}

std::size_t model_inference::ModelInference::get_required_cache_layer_count() const {
    return _required_cache_layer_count;
}

std::size_t model_inference::ModelInference::get_cached_sequence_length() const {
    if (_layer_cache.empty()) {
        return 0;
    }

    return static_cast<std::size_t>(get_past_sequence_length(_layer_cache.front()));
}
