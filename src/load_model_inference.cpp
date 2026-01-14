//
// Created by jake on 1/14/26.
//


#include "../include/load_model_inference.h"

std::vector<float> model_inference::ModelInference::run_inference(const std::vector<std::int64_t> &input_ids) {
    const Ort::AllocatorWithDefaultOptions allocator;

    std::vector<std::int64_t> input_shape = {1, static_cast<std::int64_t>(input_ids.size())};

    const auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    const Ort::Value input_tensor = Ort::Value::CreateTensor<std::int64_t>(
            memory_info, const_cast<std::int64_t *>(input_ids.data()), input_ids.size(),
            const_cast<std::int64_t *>(input_shape.data()), input_shape.size());

    const auto input_name = session_.GetInputNameAllocated(0, allocator);
    const auto output_name = session_.GetOutputNameAllocated(0, allocator);

    const char *input_names[] = {input_name.get()};
    const char *output_names[] = {output_name.get()};


    auto output_tensors = session_.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

    float *output_data = output_tensors[0].GetTensorMutableData<float>();
    const auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

    std::size_t output_size = 1;
    for (const auto &dim: output_shape) {
        output_size *= dim;
    }

    return std::vector<float>(output_data, output_data + output_size);
}
