#include "include/routes.h"
#include <iostream>
#include <onnxruntime_cxx_api.h>

int main() {
  load_routes::Routes::get_route_instance().start("127.0.0.1", 1234);
  auto Print = [](const std::string &msg) { std::cout << msg << std::endl; };
  Print("Hello, world!");
}
