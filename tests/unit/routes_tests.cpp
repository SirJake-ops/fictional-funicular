#include "fictional_funicular/http/routes.h"
#include "onnxruntime_cxx_api.h"

#include <gtest/gtest.h>

#include <stdexcept>

TEST(RouteTest, HiHandlerReturnsExpectedBody) {
  httplib::Request req;
  httplib::Response res;

  load_routes::handle_hi_request(req, res);

  EXPECT_EQ(res.status, 200);
  EXPECT_EQ(res.body, "Hello from the class");
}

TEST(RouteTest, RunModelHandlerFormatsInferenceOutput) {
  httplib::Request req;
  httplib::Response res;
  req.body = "hello";

  load_routes::Routes::get_route_instance().handle_run_model_request(
      req, res, [](const std::vector<std::int64_t> &input_ids) {
        EXPECT_FALSE(input_ids.empty());
        EXPECT_EQ(input_ids.front(), 5);
        std::vector<float> logits(50257 * 2, -1.0f);
        logits[50257 + 1] = 10.0f;
        return logits;
      });

  EXPECT_EQ(res.status, 200);
  EXPECT_NE(res.body.find("next_token_id: 1"), std::string::npos);
  EXPECT_NE(res.body.find("input_length: 2"), std::string::npos);
}

TEST(RouteTest, RunModelHandlerReturnsStdExceptionMessage) {
  httplib::Request req;
  httplib::Response res;

  load_routes::Routes::get_route_instance().handle_run_model_request(
      req, res, [](const std::vector<std::int64_t> &) -> std::vector<float> {
        throw std::runtime_error("model missing");
      });

  EXPECT_EQ(res.status, 500);
  EXPECT_EQ(res.body, "model missing");
}

TEST(RouteTest, RunModelHandlerReturnsOrtExceptionMessage) {
  httplib::Request req;
  httplib::Response res;

  load_routes::Routes::get_route_instance().handle_run_model_request(
      req, res, [](const std::vector<std::int64_t> &) -> std::vector<float> {
        throw Ort::Exception("ORT exploded", ORT_RUNTIME_EXCEPTION);
      });

  EXPECT_EQ(res.status, 500);
  EXPECT_NE(res.body.find("Model failed to run: ORT exploded"),
            std::string::npos);
}
