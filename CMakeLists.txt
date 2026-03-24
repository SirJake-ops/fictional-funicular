cmake_minimum_required(VERSION 3.20)

project(LLM_Inference_Engine VERSION 0.1.0 LANGUAGES CXX)

include(CTest)

set(ONNXRUNTIME_LIB
    "${CMAKE_CURRENT_SOURCE_DIR}/models/libonnxruntime.so"
    CACHE FILEPATH "Path to the ONNX Runtime shared library"
)

if(NOT (EXISTS "${ONNXRUNTIME_LIB}" OR IS_SYMLINK "${ONNXRUNTIME_LIB}"))
    message(FATAL_ERROR "Expected ONNX Runtime shared library at ${ONNXRUNTIME_LIB}")
endif()

get_filename_component(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_LIB}" DIRECTORY)

add_library(onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_LIB}"
)

add_library(llm_inference_engine_lib STATIC)

target_sources(llm_inference_engine_lib
    PRIVATE
        src/tokenizer/tokenizer.cpp
        src/inference/model_inference.cpp
        src/http/routes.cpp
        src/common/globals.cpp
)

target_compile_features(llm_inference_engine_lib PUBLIC cxx_std_20)

target_compile_definitions(llm_inference_engine_lib
    PUBLIC
        LLM_INFERENCE_ENGINE_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
)

target_include_directories(llm_inference_engine_lib
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/http"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnx"
)

target_link_libraries(llm_inference_engine_lib
    PUBLIC
        onnxruntime
)

add_executable(LLM_Inference_Engine src/app/main.cpp)

target_link_libraries(LLM_Inference_Engine
    PRIVATE
        llm_inference_engine_lib
)

set_target_properties(LLM_Inference_Engine PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}"
)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
