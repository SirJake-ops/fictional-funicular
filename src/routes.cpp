//
// Created by jake on 1/13/26.
//

#include "../include/routes.h"
#include <iostream>

void load_routes::Routes::start(const char *host, const int &port) {
    {//GET Requests
        get_hi();
        stop_server();
    }
    _svr.listen(host, port);
}

void load_routes::Routes::get_hi() {
    try {
        _svr.Get("/hi", [](const httplib::Request &req, httplib::Response &res) {
            res.set_content("Hello from the class", "text/plain");
        });
    } catch (...) {
        std::cerr << "Request not valid" << std::endl;
    }
}
void load_routes::Routes::stop_server() {
    try {
        _svr.Get("/stop", [&](const httplib::Request &req, httplib::Response &res) { _svr.stop(); });
    } catch (...) {
        std::cerr << "Server cannot stop." << std::endl;
    }
}
