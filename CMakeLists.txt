cmake_minimum_required(VERSION 4.1)
project(LLM_Inference_Engine)

set(CMAKE_CXX_STANDARD 20)

file(GLOB_RECURSE HEADERS include/*.h)
file(GLOB_RECURSE SOURCES src/*.cpp)

add_executable(LLM_Inference_Engine
        ${HEADERS}
        ${SOURCES}
        main.cpp)
