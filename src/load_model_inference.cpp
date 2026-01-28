//
// Created by jake on 1/14/26.
//

#include "../include/load_model_inference.h"
#include <vector>

std::vector<float> model_inference::ModelInference::run_inference(
    const std::vector<std::int64_t> &input_ids) {
  const Ort::AllocatorWithDefaultOptions allocator;

  const std::vector<std::int64_t> input_shape = {
      1, static_cast<std::int64_t>(input_ids.size())};
  std::vector<std::int64_t> input_ids_copy = input_ids;

  const auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::vector<std::int64_t> attention_mask(input_ids.size(), 1);

  auto mask_tensor = Ort::Value::CreateTensor(
      memory_info, attention_mask.data(), attention_mask.size(),
      input_shape.data(), input_shape.size());

  Ort::Value input_tensor = Ort::Value::CreateTensor<std::int64_t>(
      memory_info, input_ids_copy.data(), input_ids_copy.size(),
      input_shape.data(), input_shape.size());

  const auto input_name_1 = session_.GetInputNameAllocated(0, allocator);
  const auto input_name_2 = session_.GetInputNameAllocated(1, allocator);
  const auto output_name = session_.GetOutputNameAllocated(0, allocator);

  const char *input_names[] = {input_name_1.get(), input_name_2.get()};
  const char *output_names[] = {output_name.get()};

  std::vector<Ort::Value> input_tensors;
  input_tensors.push_back(std::move(input_tensor));
  input_tensors.push_back(std::move(mask_tensor));

  auto output_tensors =
      session_.Run(Ort::RunOptions{nullptr}, input_names, input_tensors.data(),
                   input_tensors.size(), output_names, 1);

  float *output_data = output_tensors[0].GetTensorMutableData<float>();
  const auto output_shape =
      output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

  std::size_t output_size = 1;
  for (const auto &dim : output_shape) {
    output_size *= dim;
  }

  return std::vector<float>(output_data, output_data + output_size);
}
