//
// Created by jake on 1/13/26.
//

#ifndef LLM_INFERENCE_ENGINE_ROUTES_H
#define LLM_INFERENCE_ENGINE_ROUTES_H
#include "httplib.h"


enum class REST;
namespace load_routes {
    class Routes {
    public:
        static Routes &get_route_instance() {
            static Routes instance;
            return instance;
        }

        Routes(Routes &) = delete;
        Routes &operator=(const Routes &) = delete;

        void start(const char *host, const int &port);
        void get_hi();
        void stop_server();

    private:
        Routes() = default;
        httplib::Server _svr;
        httplib::ErrorLogger _logger;
    };
} // namespace load_routes

#endif // LLM_INFERENCE_ENGINE_ROUTES_H
