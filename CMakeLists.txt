cmake_minimum_required(VERSION 3.20)
project(LLM_Inference_Engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/include/*.h)
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/*.cpp)

add_executable(LLM_Inference_Engine
        main.cpp
        ${SOURCES}
        ${HEADERS}
        include/routes.h
        docs/onnx_examples.cpp
        src/routes.cpp
        shared/globals.cpp
        include/load_model_inference.h
        src/load_model_inference.cpp
)

target_include_directories(LLM_Inference_Engine PRIVATE
        ${CMAKE_SOURCE_DIR}/extern/http/
        ${CMAKE_SOURCE_DIR}/extern/onnx/
)

target_link_libraries(LLM_Inference_Engine PRIVATE
        onnxruntime
)
