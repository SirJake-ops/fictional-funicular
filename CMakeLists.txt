cmake_minimum_required(VERSION 3.20)
project(LLM_Inference_Engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ONNXRUNTIME_LIB "${CMAKE_SOURCE_DIR}/models/libonnxruntime.so" CACHE FILEPATH
        "Path to the ONNX Runtime shared library")

file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/include/*.h)
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE DOCS CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/docs/*.cpp)
file(GLOB_RECURSE SHARED CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/shared/*.cpp)


add_executable(LLM_Inference_Engine
        main.cpp
        ${SOURCES}
        ${HEADERS}
        ${DOCS}
        ${SHARED}
)

target_include_directories(LLM_Inference_Engine PRIVATE
        ${CMAKE_SOURCE_DIR}/extern/http/
        ${CMAKE_SOURCE_DIR}/extern/onnx/
)

if(NOT (EXISTS "${ONNXRUNTIME_LIB}" OR IS_SYMLINK "${ONNXRUNTIME_LIB}"))
    message(FATAL_ERROR "Expected ONNX Runtime shared library at ${ONNXRUNTIME_LIB}")
endif()

target_link_libraries(LLM_Inference_Engine PRIVATE
        "${ONNXRUNTIME_LIB}"
)

set_target_properties(LLM_Inference_Engine PROPERTIES
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/models"
)
