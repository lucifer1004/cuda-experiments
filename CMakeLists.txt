cmake_minimum_required(VERSION 3.22)
project(cuda-experiments LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES
    70 # Volta
    75 # Turing
    80 # Ampere (T100)
    86 # Ampere (RTX 30)
    89 # Ada Lovelace
    90 # Hopper
)

include_directories(.pixi/env/include)
include_directories(cutlass/include)
add_subdirectory(src)
