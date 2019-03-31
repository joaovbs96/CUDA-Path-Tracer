cmake_minimum_required(VERSION 3.0.0)
project(CUDA-Path-Tracer VERSION 0.1.0)

include(CTest)
enable_testing()

find_package(CUDA REQUIRED)

if (CUDA_TOOLKIT_ROOT_DIR)
	include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
endif()

# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -gencode arch=compute_60,code=sm_60 -use_fast_math

    # For GTX 10XX, use arch=compute_60,code=sm_60
    # For GTX 9XX use arch=compute_XX,code=sm_XX
)

# Specify include directories
# include_directories()

# Specify library paths
# link_directories()

cuda_add_executable(
  CUDA-Path-Tracer

  # main kernel
  src/main.cu
)