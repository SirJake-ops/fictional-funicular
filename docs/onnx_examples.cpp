//
// Created by jake on 1/13/26.
//
#include "onnxruntime_cxx_api.h"
#include <vector>
// Load the model and create InferenceSession

Ort::Env env;
std::string model_path = "path/to/your/onnx/model";
// Ort::Session session(env, model_path, Ort::SessionOptions{ nullptr });
// Load and preprocess the input image to inputTensor

// Run inference
// std::vector outputTensors =
// session.Run(Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor,
  // inputNames.size(), outputNames.data(), outputNames.size());
// const float* outputDataPtr = outputTensors[0].GetTensorMutableData();
// std::cout << outputDataPtr[0] << std::endl;
