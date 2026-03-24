cmake_minimum_required(VERSION 4.1)
include(GoogleTest)

find_package(GTest CONFIG QUIET)

if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )

    FetchContent_MakeAvailable(googletest)
endif()

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/unit/*.cpp"
)

add_executable(LLM_Inference_Engine_Test
        tests_main.cpp
    ${TEST_SOURCES}
)

target_compile_features(LLM_Inference_Engine_Test PRIVATE cxx_std_20)

target_link_libraries(LLM_Inference_Engine_Test
    PRIVATE
        GTest::gtest
        llm_inference_engine_lib
)

gtest_discover_tests(LLM_Inference_Engine_Test)
