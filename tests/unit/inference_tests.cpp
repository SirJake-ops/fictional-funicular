#include "fictional_funicular/http/routes.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <stdexcept>
#include <vector>
#include <cstdint>

static httplib::Request make_request(std::string body = {}) {
  httplib::Request req;
  req.body = std::move(body);
  return req;
}

namespace {
  constexpr std::size_t kVocabularySize = 50257;
  const load_routes::InferenceRunner k_successful_runner =
      [](const std::vector<std::int64_t> &) {
    std::vector<float> logits(50257 * 2, -1.0f);
    logits[50257 + 1] = 10.0f;
    return logits;
  };
}

TEST(TokenGenerationTest, PicksBestTokenFromLastVocabularyWindow) {
  const std::vector<float> logits = {
      9.0f, 8.0f, 7.0f,
      0.2f, 0.9f, 0.1f,
  };

  EXPECT_EQ(load_routes::get_next_token(logits, 3), 1);
}

TEST(InferenceTest, MissingModelPathThrowsHelpfulError) {
  const auto missing_path =
      std::filesystem::path("models/definitely_missing_decoder_model.onnx");

  EXPECT_THROW(
      {
        try {
          (void)load_routes::run_inference_with_model({15496, 995}, missing_path);
        } catch (const std::runtime_error &e) {
          EXPECT_NE(std::string(e.what()).find(missing_path.string()),
                    std::string::npos);
          throw;
        }
      },
      std::runtime_error);
}

TEST(TokenGenerationTest, RejectsLogitVectorsShorterThanVocabularySize) {
  EXPECT_THROW(load_routes::get_next_token({0.1f, 0.2f, 0.3f}, 4),
               std::invalid_argument);
}

TEST(RunModelTest, HandleRunModelRequest) {
  httplib::Request req;
  httplib::Response res;
  req.body = "Hello";

  load_routes::Routes::get_route_instance().handle_run_model_request(req, res, [](const std::vector<std::int64_t>& inputs) {
    EXPECT_FALSE(inputs.empty());
    std::vector<float> logits(50257 * 2, -1.0f);
    logits[50257 + 1] = 10.0f;
    return logits;
  });


  EXPECT_EQ(res.status, 200);
  EXPECT_NE(res.body.find("next_token_id: 1"), std::string::npos);
}

TEST(RoutesRegisteredTest, RoutesRegisteredOnStartup) {
  httplib::Server server;
  load_routes::register_routes(server, k_successful_runner);

  std::thread server_thread([&]() {
    server.listen("127.0.0.1", 18081);
  });

  std::this_thread::sleep_for(std::chrono::seconds(1));
  httplib::Client client("127.0.0.1", 18081);
  auto res = client.Get("/hi");

  ASSERT_TRUE(res);
  ASSERT_EQ(res->status, 200);


  client.Get("/stop");
  server_thread.join();
}
