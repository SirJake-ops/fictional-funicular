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

        static void set_model_path(const std::string &model_path);
        void start(const char *host, const int &port);
        void get_hi();
        void run_model();
        void generate();
        void stop_server();

    private:
        Routes() = default;
        httplib::Server svr_;
        httplib::ErrorLogger logger_;
        std::string path_to_model_;
    };
} // namespace load_routes

#endif // LLM_INFERENCE_ENGINE_ROUTES_H
