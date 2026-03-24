//
// Created by jake on 1/14/26.
//

#include "fictional_funicular/inference/model_inference.h"

#include <vector>

std::vector<float> model_inference::ModelInference::run_inference(
    const std::vector<std::int64_t> &input_ids, const int &number_of_layers) {
    if (_layer_cache.size() != static_cast<std::size_t>(number_of_layers)) {
        _layer_cache.clear();
        _layer_cache.resize(static_cast<std::size_t>(number_of_layers));
    }
    const Ort::AllocatorWithDefaultOptions allocator;

    const std::vector<std::int64_t> input_shape = {
        1, static_cast<std::int64_t>(input_ids.size())
    };
    std::vector<std::int64_t> input_ids_copy = input_ids;
    std::vector<std::int64_t> attention_mask(input_ids.size(), 1);

    const auto memory_info =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<std::int64_t>(
        memory_info, input_ids_copy.data(), input_ids_copy.size(),
        input_shape.data(), input_shape.size());

    Ort::Value mask_tensor = Ort::Value::CreateTensor(
        memory_info, attention_mask.data(), attention_mask.size(),
        input_shape.data(), input_shape.size());

    std::vector<std::string> input_name_storage;
    std::vector<const char *> input_names;
    std::vector<std::string> output_name_storage;
    std::vector<const char *> output_names;
    std::vector<Ort::Value> input_tensors;

    input_name_storage.reserve(2 + number_of_layers * 2);
    input_tensors.reserve(2 + number_of_layers * 2);
    input_names.reserve(2 + number_of_layers * 2);

    input_name_storage.push_back("input_ids");
    input_tensors.push_back(std::move(input_tensor));

    input_name_storage.push_back("attention_mask");
    input_tensors.push_back(std::move(mask_tensor));

    for (int i = 0; i < number_of_layers; ++i) {
        if (!_layer_cache.at(i)._keys.empty() && !_layer_cache.at(i)._values.empty()) {
            input_name_storage.push_back("past_key_" + std::to_string(i));
            input_tensors.push_back(std::move(_layer_cache[i]._keys.back()));

            input_name_storage.push_back("past_value_" + std::to_string(i));
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
