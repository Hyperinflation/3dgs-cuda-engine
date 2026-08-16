cmake_minimum_required(VERSION 3.18)
project(3dgs_cuda_engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Support multiple NVIDIA GPU architectures: sm_75 (T4/RTX 2080), sm_80 (A100), sm_86 (RTX 3090), sm_89 (RTX 4090)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

# 1. Build Shared Library
add_library(3dgs_cuda SHARED
    rasterizer.cu
    forward.cu
    backward.cu
)
target_include_directories(3dgs_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# 2. Build Test Harness Executable
add_executable(test_3dgs_cuda
    test_kernel.cu
    rasterizer.cu
    forward.cu
    backward.cu
)
target_include_directories(test_3dgs_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
