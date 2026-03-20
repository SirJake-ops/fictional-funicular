#include "fictional_funicular/http/routes.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <stdexcept>

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
