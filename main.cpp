#include <iostream>
#include <onnxruntime_cxx_api.h>
#include "httplib.h"

int main() {

    using namespace httplib;

    Server svr;

    svr.Get("/hi", [](const Request &req, Response &res) { res.set_content("Hello World!", "text/plain"); });


    svr.Get("/stop", [&](const Request &req, Response &res) { svr.stop(); });

    svr.listen("localhost", 1234);


    auto Print = [](const std::string &msg) { std::cout << msg << std::endl; };
    Print("Hello, world!");
}
