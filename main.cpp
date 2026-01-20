#include <iostream>
#include <onnxruntime_cxx_api.h>
#include "include/routes.h"

int main() {
    load_routes::Routes::set_model_path("./models/decoder_model.onnx");
    load_routes::Routes::get_route_instance().start("localhost", 1234);
    auto Print = [](const std::string &msg) { std::cout << msg << std::endl; };
    Print("Hello, world!");
}
