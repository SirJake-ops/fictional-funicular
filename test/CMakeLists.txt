cmake_minimum_required(VERSION 3.14)
project(LLM_Inference_Engine_Test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(ROOT_DIR ${CMAKE_SOURCE_DIR}/..)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/release-1.12.1.zip
)
FetchContent_MakeAvailable(googletest)

file(GLOB_RECURSE TEST_SOURCES ${ROOT_DIR}/test/test-src/*.cpp)


enable_testing()
add_executable(LLM_Inference_Engine_Test main_test.cpp ${TEST_SOURCES} main_test.cpp)
target_link_libraries(LLM_Inference_Engine_Test gtest)
include(GoogleTest)
gtest_discover_tests(LLM_Inference_Engine_Test)
